"""OOTDiffusion trainer with real mask+pose conditioning from stratified-category dataset."""

import argparse
import os
import random
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid

from common import add_common_args, cleanup_dist, latest_checkpoint, setup_dist, wrap_ddp
from utils import _local_load_image

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


_FC_MC_RE = re.compile(r"_(?:fc|mc)_")


def _maybe_init_wandb(args, is_main):
    if not is_main or args.disable_wandb or os.environ.get("DISABLE_WANDB", "0") == "1":
        return None
    if wandb is None:
        return None
    try:
        return wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    except Exception:
        return None


def _to_wandb_image(batch: torch.Tensor, caption: str):
    if wandb is None:
        return None
    vis = (batch.clamp(-1, 1) + 1.0) * 0.5
    grid = make_grid(vis, nrow=min(4, vis.shape[0]))
    return wandb.Image(grid, caption=caption)


class OOTCategoryMaskPoseDataset(Dataset):
    """root/category/gender/{cloth,initial_person,mask,pose,tryon}."""

    def __init__(self, root_dir: str, category: str = "all", gender: str = "all", size: int = 0):
        self.root_dir = root_dir
        self.category = category
        self.gender = gender
        self.size = size
        self.samples = []

        if size and size > 0:
            self.img_tf = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            self.mask_tf = transforms.Compose([
                transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ])
        else:
            self.img_tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            self.mask_tf = transforms.Compose([transforms.ToTensor()])

        categories = ("dresses", "upper_body", "lower_body", "uncertain") if category == "all" else (category,)
        genders = ("female", "male") if gender == "all" else (gender,)

        for cat in categories:
            for g in genders:
                self._collect(cat, g)

        if not self.samples:
            raise RuntimeError(f"No valid samples found under {root_dir} for category={category}, gender={gender}")
        print(f"[OOTMaskPoseDataset] Loaded {len(self.samples)} samples")

    def _collect(self, category: str, gender: str):
        leaf = os.path.join(self.root_dir, category, gender)
        cloth_dir = os.path.join(leaf, "cloth_image")
        person_dir = os.path.join(leaf, "initial_person_image")
        mask_dir = os.path.join(leaf, "mask_image")
        pose_dir = os.path.join(leaf, "pose_image")
        tryon_dir = os.path.join(leaf, "tryon_image")
        for d in (cloth_dir, person_dir, mask_dir, pose_dir, tryon_dir):
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Missing directory: {d}")

        cloth_files = sorted([f for f in os.listdir(cloth_dir) if f.lower().endswith(".png")])
        person_stems = {os.path.splitext(f)[0] for f in os.listdir(person_dir) if f.lower().endswith(".png")}
        mask_set = set([f for f in os.listdir(mask_dir) if f.lower().endswith(".png")])
        pose_set = set([f for f in os.listdir(pose_dir) if f.lower().endswith(".png")])
        tryon_set = set([f for f in os.listdir(tryon_dir) if f.lower().endswith(".png")])

        for fname in cloth_files:
            if fname not in tryon_set or fname not in mask_set or fname not in pose_set:
                continue
            stem = os.path.splitext(fname)[0]
            m = _FC_MC_RE.search(stem)
            if m is None:
                continue
            person_stem = stem[:m.start()]
            if person_stem not in person_stems:
                continue
            self.samples.append((
                os.path.join(person_dir, person_stem + ".png"),
                os.path.join(cloth_dir, fname),
                os.path.join(mask_dir, fname),
                os.path.join(pose_dir, fname),
                os.path.join(tryon_dir, fname),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        person_p, cloth_p, mask_p, pose_p, tryon_p = self.samples[idx]
        person = self.img_tf(_local_load_image(person_p))
        cloth = self.img_tf(_local_load_image(cloth_p))
        pose = self.img_tf(_local_load_image(pose_p))
        tryon = self.img_tf(_local_load_image(tryon_p))
        mask = self.mask_tf(_local_load_image(mask_p).convert("L"))
        mask = (mask > 0.5).float()
        return {
            "person": person,
            "cloth": cloth,
            "pose": pose,
            "mask": mask,
            "ground_truth": tryon,
        }


def _collate(batch):
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0].keys()}


class OOTDiffusionModel(nn.Module):
    def __init__(self, model_name, outfitting_dropout=0.1):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
        self.scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.denoising_unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
        self.outfitting_unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
        self.vae.requires_grad_(False)
        self.outfitting_dropout = outfitting_dropout
        self.outfit_adapter = nn.Conv2d(4, 4, kernel_size=1)

    @property
    def cross_attention_dim(self):
        dim = self.denoising_unet.config.cross_attention_dim
        return int(dim[0] if isinstance(dim, (tuple, list)) else dim)

    def encode(self, image):
        latents = self.vae.encode(image).latent_dist.sample()
        return latents * self.vae.config.scaling_factor

    def empty_text(self, batch_size, device, dtype):
        return torch.zeros(batch_size, 77, self.cross_attention_dim, device=device, dtype=dtype)

    def forward(self, noisy_target, person_lat, cloth_lat, timesteps):
        text = self.empty_text(noisy_target.shape[0], noisy_target.device, noisy_target.dtype)
        if self.training and random.random() < self.outfitting_dropout:
            outfit_feature = torch.zeros_like(cloth_lat)
        else:
            outfit_feature = self.outfitting_unet(cloth_lat, timesteps, text).sample
        fused_noisy = noisy_target + self.outfit_adapter(outfit_feature) + 0.05 * person_lat
        return self.denoising_unet(fused_noisy, timesteps, text).sample


def train(args):
    dist_info = setup_dist()
    model = OOTDiffusionModel(args.model_name, args.outfitting_dropout).to(dist_info.device)
    model.denoising_unet = wrap_ddp(model.denoising_unet, dist_info)
    model.outfitting_unet = wrap_ddp(model.outfitting_unet, dist_info)

    ds = OOTCategoryMaskPoseDataset(
        root_dir=args.curvton_data_path,
        category=args.category,
        gender=args.gender,
        size=0,  # full-resolution
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        ds, num_replicas=dist_info.world_size, rank=dist_info.rank, shuffle=True, drop_last=True
    ) if dist_info.world_size > 1 else None
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=_collate,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    params = list(model.denoising_unet.parameters()) + list(model.outfitting_unet.parameters()) + list(model.outfit_adapter.parameters())
    optimizer = AdamW(params, lr=args.lr)
    scaler = GradScaler(enabled=(dist_info.device.type == "cuda"))
    wb_run = _maybe_init_wandb(args, dist_info.is_main)

    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_to_load = args.resume
    if ckpt_to_load is None and not args.no_resume:
        ckpt_to_load = latest_checkpoint(run_dir)

    step = 0
    if ckpt_to_load:
        ckpt = torch.load(ckpt_to_load, map_location=dist_info.device)
        model.denoising_unet.module.load_state_dict(ckpt["denoising_unet_state_dict"]) if hasattr(model.denoising_unet, "module") else model.denoising_unet.load_state_dict(ckpt["denoising_unet_state_dict"])
        model.outfitting_unet.module.load_state_dict(ckpt["outfitting_unet_state_dict"]) if hasattr(model.outfitting_unet, "module") else model.outfitting_unet.load_state_dict(ckpt["outfitting_unet_state_dict"])
        model.outfit_adapter.load_state_dict(ckpt["outfit_adapter_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step = int(ckpt.get("step", 0))

    @torch.no_grad()
    def _sample_tryon(person_lat, cloth_lat, n_steps):
        latents = torch.randn_like(person_lat)
        model.scheduler.set_timesteps(n_steps, device=latents.device)
        for t in model.scheduler.timesteps:
            t_batch = torch.full((latents.shape[0],), int(t), device=latents.device, dtype=torch.long)
            noise_pred = model(latents, person_lat, cloth_lat, t_batch)
            latents = model.scheduler.step(noise_pred, t, latents).prev_sample
        return model.vae.decode(latents / model.vae.config.scaling_factor).sample

    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for batch in loader:
            person = batch["person"].to(dist_info.device, non_blocking=True)
            cloth = batch["cloth"].to(dist_info.device, non_blocking=True)
            pose = batch["pose"].to(dist_info.device, non_blocking=True)
            mask = batch["mask"].to(dist_info.device, non_blocking=True)
            gt = batch["ground_truth"].to(dist_info.device, non_blocking=True)

            # Use actual mask + pose (no surrogate).
            masked_person = person * (1.0 - mask) + (-1.0) * mask
            person_for_model = torch.clamp(0.5 * masked_person + 0.5 * pose, -1.0, 1.0)

            with torch.no_grad():
                target_lat = model.encode(gt)
                person_lat = model.encode(person_for_model)
                cloth_lat = model.encode(cloth)

            noise = torch.randn_like(target_lat)
            timesteps = torch.randint(0, model.scheduler.config.num_train_timesteps, (target_lat.shape[0],), device=target_lat.device).long()
            noisy = model.scheduler.add_noise(target_lat, noise, timesteps)
            with autocast(enabled=(dist_info.device.type == "cuda")):
                pred = model(noisy, person_lat, cloth_lat, timesteps)
                loss = F.mse_loss(pred.float(), noise.float())

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1

            if dist_info.is_main and step % args.log_interval == 0:
                print(f"[step {step:>6}/{args.max_steps}] loss={loss.item():.6f}", flush=True)
            if dist_info.is_main and wb_run is not None:
                wb_run.log({"train/loss": float(loss.item()), "train/step": step}, step=step)
            if dist_info.is_main and wb_run is not None and step % args.image_log_interval == 0:
                with torch.no_grad():
                    k = min(8, person_lat.shape[0])
                    pred_img = _sample_tryon(person_lat[:k], cloth_lat[:k], args.num_inference_steps)
                payload = {
                    "images/pred_tryon": _to_wandb_image(pred_img[:8].detach().cpu(), f"OOT-mask pred step {step}"),
                    "images/person_masked_pose": _to_wandb_image(person_for_model[:8].detach().cpu(), f"person+pose step {step}"),
                    "images/mask": _to_wandb_image(mask[:8].repeat(1, 3, 1, 1).detach().cpu() * 2 - 1, f"mask step {step}"),
                }
                wb_run.log({k: v for k, v in payload.items() if v is not None}, step=step)

            if dist_info.is_main and step % args.save_interval == 0:
                torch.save(
                    {
                        "step": step,
                        "architecture": "OOTDiffusion + real mask/pose",
                        "denoising_unet_state_dict": model.denoising_unet.module.state_dict() if hasattr(model.denoising_unet, "module") else model.denoising_unet.state_dict(),
                        "outfitting_unet_state_dict": model.outfitting_unet.module.state_dict() if hasattr(model.outfitting_unet, "module") else model.outfitting_unet.state_dict(),
                        "outfit_adapter_state_dict": model.outfit_adapter.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "args": vars(args),
                    },
                    os.path.join(run_dir, f"ckpt_step_{step}.pt"),
                )
            if step >= args.max_steps:
                break

    if dist_info.is_main:
        torch.save(
            {
                "step": step,
                "architecture": "OOTDiffusion + real mask/pose",
                "denoising_unet_state_dict": model.denoising_unet.module.state_dict() if hasattr(model.denoising_unet, "module") else model.denoising_unet.state_dict(),
                "outfitting_unet_state_dict": model.outfitting_unet.module.state_dict() if hasattr(model.outfitting_unet, "module") else model.outfitting_unet.state_dict(),
                "outfit_adapter_state_dict": model.outfit_adapter.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "args": vars(args),
            },
            os.path.join(run_dir, "ckpt_final.pt"),
        )
        if wb_run is not None:
            wb_run.finish()
    cleanup_dist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OOTDiffusion trainer with real mask+pose")
    add_common_args(parser)
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--outfitting_dropout", type=float, default=0.1)
    parser.add_argument("--category", type=str, default="all", choices=["all", "dresses", "upper_body", "lower_body", "uncertain"])
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_resume", action="store_true", default=False)
    args = parser.parse_args()
    args.run_name = args.run_name or "train_ootdiffusion_mask"
    args.image_size = 0
    train(args)
