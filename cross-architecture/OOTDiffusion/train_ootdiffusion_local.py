"""OOTDiffusion architecture-faithful local trainer for CurvTON.

Paper architecture:
- A normal denoising UNet predicts diffusion noise for the person/try-on latent.
- A separate outfitting UNet, architecturally identical to the denoising UNet,
  learns garment detail features from the garment latent.
- Garment information is fused into the denoising process; this local version
  implements fusion through a learned latent feature adapter because the public
  OOTDiffusion repo does not expose training source for its attention hooks.

Dataset adaptation:
- CurvTON provides initial_person_image, cloth_image, and tryon_image.
- This trainer explicitly uses:
  - masked person input := initial person image
  - mask input := black tensor (all zeros)
"""

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
from torchvision.utils import make_grid

from common import (
    add_common_args,
    cleanup_dist,
    latest_checkpoint,
    setup_dist,
    wrap_ddp,
)
from utils import _local_load_image, collate_fn
from torchvision import transforms

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


def raw_module(module):
    return module.module if hasattr(module, "module") else module


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


_FC_MC_RE = re.compile(r"_(?:fc|mc)_")
_CAT_RE = re.compile(r"_(dresses|upper_body|lower_body|uncertain)\.png$", re.IGNORECASE)


class OOTStratifiedTypeDataset(Dataset):
    """Dataset for:
    root/
      female|male/
        cloth_image/
        initial_person_image/
        tryon_image/
    """

    def __init__(self, root_dir: str, gender: str = "all", size: int = 512, category: str = "all"):
        self.root_dir = root_dir
        self.gender = gender
        self.category = category.lower()
        self.size = size
        self.samples = []

        if size and size > 0:
            self.img_tf = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            self.img_tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

        genders = ("female", "male") if gender == "all" else (gender,)
        for g in genders:
            self._collect_gender(g)

        if not self.samples:
            raise RuntimeError(
                f"No valid samples found under {root_dir} for gender={gender}, category={self.category}"
            )
        print(f"[OOTDataset] Loaded {len(self.samples)} samples (gender={gender}, category={self.category})")

    def _collect_gender(self, gender: str):
        leaf = os.path.join(self.root_dir, gender)
        cloth_dir = os.path.join(leaf, "cloth_image")
        person_dir = os.path.join(leaf, "initial_person_image")
        tryon_dir = os.path.join(leaf, "tryon_image")
        for d in (cloth_dir, person_dir, tryon_dir):
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Missing directory: {d}")

        cloth_files = sorted([f for f in os.listdir(cloth_dir) if f.lower().endswith(".png")])
        person_stems = {os.path.splitext(f)[0] for f in os.listdir(person_dir) if f.lower().endswith(".png")}
        tryon_set = set([f for f in os.listdir(tryon_dir) if f.lower().endswith(".png")])

        for fname in cloth_files:
            m_cat = _CAT_RE.search(fname)
            if m_cat is None:
                continue
            cat = m_cat.group(1).lower()
            if self.category != "all" and cat != self.category:
                continue
            if fname not in tryon_set:
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
                os.path.join(tryon_dir, fname),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        person_path, cloth_path, tryon_path = self.samples[idx]
        person_img = _local_load_image(person_path)
        cloth_img = _local_load_image(cloth_path)
        tryon_img = _local_load_image(tryon_path)
        person = self.img_tf(person_img)
        cloth = self.img_tf(cloth_img)
        gt = self.img_tf(tryon_img)
        return {
            "ground_truth": gt,
            "cloth": cloth,
            "person": person,
            "mask": torch.zeros(1, person.shape[1], person.shape[2]),
        }


def _build_loader(args, dist_info):
    # Enforce full-resolution training for OOTDiffusion.
    enforced_size = 0
    if getattr(args, "image_size", None) not in (None, 0):
        if dist_info.is_main:
            print(f"[OOT] Overriding image_size={args.image_size} -> 0 (full resolution)")
    ds = OOTStratifiedTypeDataset(
        root_dir=args.curvton_data_path,
        gender=args.gender,
        size=enforced_size,
        category=args.category,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        ds,
        num_replicas=dist_info.world_size,
        rank=dist_info.rank,
        shuffle=True,
        drop_last=True,
    ) if dist_info.world_size > 1 else None
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    return loader, sampler


def _batch_images(batch, device, dtype=torch.float32):
    gt = batch["ground_truth"].to(device, dtype=dtype, non_blocking=True)
    cloth = batch["cloth"].to(device, dtype=dtype, non_blocking=True)
    person = batch["person"].to(device, dtype=dtype, non_blocking=True)
    return person, cloth, gt


def train(args):
    dist_info = setup_dist()
    model = OOTDiffusionModel(args.model_name, args.outfitting_dropout).to(dist_info.device)
    model.denoising_unet = wrap_ddp(model.denoising_unet, dist_info)
    model.outfitting_unet = wrap_ddp(model.outfitting_unet, dist_info)
    loader, sampler = _build_loader(args, dist_info)
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
        raw_module(model.denoising_unet).load_state_dict(ckpt["denoising_unet_state_dict"])
        raw_module(model.outfitting_unet).load_state_dict(ckpt["outfitting_unet_state_dict"])
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
            person, cloth, gt = _batch_images(batch, dist_info.device)
            # Enforce requested behavior:
            # 1) masked person image -> use initial person image
            # 2) mask -> black tensor
            black_mask = torch.zeros(
                person.shape[0], 1, person.shape[2], person.shape[3],
                device=person.device, dtype=person.dtype
            )
            masked_person = person
            person_for_model = masked_person * (1.0 - black_mask)
            with torch.no_grad():
                target_lat = model.encode(gt)
                person_lat = model.encode(person_for_model)
                cloth_lat = model.encode(cloth)

            noise = torch.randn_like(target_lat)
            timesteps = torch.randint(
                0,
                model.scheduler.config.num_train_timesteps,
                (target_lat.shape[0],),
                device=target_lat.device,
            ).long()
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
                    pred_img = _sample_tryon(
                        person_lat[: min(8, person_lat.shape[0])],
                        cloth_lat[: min(8, cloth_lat.shape[0])],
                        args.num_inference_steps,
                    )
                payload = {
                    "train/step": step,
                    "images/pred_tryon": _to_wandb_image(pred_img[:8].detach().cpu(), f"OOT pred step {step}"),
                    "images/gt_tryon": _to_wandb_image(gt[:8].detach().cpu(), f"OOT gt step {step}"),
                    "images/person": _to_wandb_image(person_for_model[:8].detach().cpu(), f"OOT person step {step}"),
                    "images/cloth": _to_wandb_image(cloth[:8].detach().cpu(), f"OOT cloth step {step}"),
                }
                wb_run.log({k: v for k, v in payload.items() if v is not None}, step=step)
            if dist_info.is_main and step % args.save_interval == 0:
                os.makedirs(os.path.join(args.output_dir, args.run_name), exist_ok=True)
                torch.save({
                    "step": step,
                    "architecture": "OOTDiffusion denoising UNet + outfitting UNet + outfitting dropout",
                    "denoising_unet_state_dict": raw_module(model.denoising_unet).state_dict(),
                    "outfitting_unet_state_dict": raw_module(model.outfitting_unet).state_dict(),
                    "outfit_adapter_state_dict": model.outfit_adapter.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "args": vars(args),
                }, os.path.join(args.output_dir, args.run_name, f"ckpt_step_{step}.pt"))
            if step >= args.max_steps:
                break

    if dist_info.is_main:
        os.makedirs(os.path.join(args.output_dir, args.run_name), exist_ok=True)
        torch.save({
            "step": step,
            "architecture": "OOTDiffusion denoising UNet + outfitting UNet + outfitting dropout",
            "denoising_unet_state_dict": raw_module(model.denoising_unet).state_dict(),
            "outfitting_unet_state_dict": raw_module(model.outfitting_unet).state_dict(),
            "outfit_adapter_state_dict": model.outfit_adapter.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "args": vars(args),
        }, os.path.join(args.output_dir, args.run_name, "ckpt_final.pt"))
        if wb_run is not None:
            wb_run.finish()
    cleanup_dist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local OOTDiffusion architecture trainer")
    add_common_args(parser)
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--outfitting_dropout", type=float, default=0.1)
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        choices=["all", "dresses", "upper_body", "lower_body", "uncertain"],
        help="Garment type filter inferred from filename suffix.",
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_resume", action="store_true", default=False)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    args = parser.parse_args()
    args.run_name = args.run_name or "train_ootdiffusion"
    train(args)
