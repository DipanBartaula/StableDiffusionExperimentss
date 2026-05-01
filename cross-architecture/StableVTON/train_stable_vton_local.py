"""Local StableVITON-style trainer.

Architecture alignment (adapted to triplet dataset):
- Latent diffusion virtual try-on with 13-channel UNet input.
- StableVITON-style preprocessing keys are constructed locally:
  agnostic, agnostic-mask, densepose, cloth_mask, gt_cloth_warped_mask.
- Per user request:
  - agnostic branch uses initial person image
  - 1-channel mask is pure black
"""

import argparse
import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torchvision.utils import make_grid

from common import (
    add_common_args,
    batch_images,
    build_curvton_loader,
    cleanup_dist,
    latest_checkpoint,
    save_checkpoint,
    setup_dist,
    tv_loss,
    wrap_ddp,
)

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


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


class StableVTONModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
        self.scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.vae.requires_grad_(False)

        # Trainable SD encoder copy (paper-style branch approximation)
        self.sd_encoder_copy = copy.deepcopy(self.unet)
        self.sd_encoder_copy.requires_grad_(True)

        old_conv = self.unet.conv_in
        new_conv = nn.Conv2d(
            13,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight[:, :4].copy_(old_conv.weight[:, :4])
            nn.init.xavier_uniform_(new_conv.weight[:, 4:])
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        self.unet.conv_in = new_conv
        self.unet.config["in_channels"] = 13
        self.sd_encoder_copy.config["in_channels"] = 4

        # Zero cross-attention conditioning adapters (zero-init)
        self.garment_token_proj = nn.Conv2d(4, self.cross_attention_dim, kernel_size=1, bias=True)
        self.zero_cross_linear = nn.Linear(self.cross_attention_dim, self.cross_attention_dim, bias=True)
        nn.init.zeros_(self.garment_token_proj.weight)
        nn.init.zeros_(self.garment_token_proj.bias)
        nn.init.zeros_(self.zero_cross_linear.weight)
        nn.init.zeros_(self.zero_cross_linear.bias)

        # Freeze base UNet so training focuses on SD encoder copy + zero-cross blocks.
        self.unet.requires_grad_(False)

    @property
    def cross_attention_dim(self):
        dim = self.unet.config.cross_attention_dim
        return int(dim[0] if isinstance(dim, (list, tuple)) else dim)

    def encode(self, image):
        latents = self.vae.encode(image).latent_dist.sample()
        return latents * self.vae.config.scaling_factor

    def forward(self, noisy, mask_lat, agnostic_lat, pose_lat, timesteps):
        hidden_base = torch.zeros(
            noisy.shape[0],
            77,
            self.cross_attention_dim,
            device=noisy.device,
            dtype=noisy.dtype,
        )
        # Trainable SD encoder-copy processes garment latent and produces garment tokens.
        garment_feat = self.sd_encoder_copy(agnostic_lat, timesteps, hidden_base).sample
        garment_tokens = self.garment_token_proj(garment_feat)
        garment_tokens = F.adaptive_avg_pool2d(garment_tokens, (4, 4)).flatten(2).transpose(1, 2)
        garment_tokens = self.zero_cross_linear(garment_tokens)
        hidden = torch.cat([hidden_base, garment_tokens], dim=1)

        # StableVITON/PBE-style 13ch path: noisy(4) + mask(1) + agnostic(4) + pose(4)
        x = torch.cat([noisy, mask_lat, agnostic_lat, pose_lat], dim=1)
        return self.unet(x, timesteps, hidden).sample


def stableviton_preprocess(person, cloth, gt):
    """Build StableVITON-style preprocessing tensors from triplet data.

    Dataset-specific adaptations requested by user:
    - agnostic = initial person image
    - agnostic-mask = black image
    """
    del gt  # explicit: no gt-dependent masking for this adaptation
    agnostic = person
    agnostic_mask = torch.zeros(
        person.shape[0], 1, person.shape[2], person.shape[3],
        device=person.device, dtype=person.dtype
    )
    # Triplet dataset has no densepose file; use initial person image as pose surrogate.
    pose_img = person
    # Approximate cloth masks from cloth intensity.
    cloth_mask = (cloth.mean(dim=1, keepdim=True) > -0.95).to(cloth.dtype)
    # No explicit warped cloth mask available; identity proxy keeps ATV term well-posed.
    gt_cloth_warped_mask = cloth_mask
    return {
        "agnostic": agnostic,
        "agnostic_mask": agnostic_mask,
        "pose_img": pose_img,
        "cloth_mask": cloth_mask,
        "gt_cloth_warped_mask": gt_cloth_warped_mask,
    }


def train(args):
    dist_info = setup_dist()
    model = StableVTONModel(args.model_name).to(dist_info.device)
    model.unet = wrap_ddp(model.unet, dist_info)
    loader, sampler = build_curvton_loader(args, dist_info)
    trainable_params = (
        list(model.sd_encoder_copy.parameters())
        + list(model.garment_token_proj.parameters())
        + list(model.zero_cross_linear.parameters())
    )
    optimizer = AdamW(trainable_params, lr=args.lr)
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
        model.unet.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "sd_encoder_copy_state_dict" in ckpt:
            model.sd_encoder_copy.load_state_dict(ckpt["sd_encoder_copy_state_dict"], strict=False)
        if "garment_token_proj_state_dict" in ckpt:
            model.garment_token_proj.load_state_dict(ckpt["garment_token_proj_state_dict"], strict=False)
        if "zero_cross_linear_state_dict" in ckpt:
            model.zero_cross_linear.load_state_dict(ckpt["zero_cross_linear_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step = int(ckpt.get("step", 0))

    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for batch in loader:
            person, cloth, gt = batch_images(batch, dist_info.device)
            prep = stableviton_preprocess(person, cloth, gt)

            with torch.no_grad():
                target_lat = model.encode(gt)
                agnostic_lat = model.encode(prep["agnostic"])
                pose_lat = model.encode(prep["pose_img"])
                mask_lat = F.interpolate(
                    prep["agnostic_mask"], size=target_lat.shape[-2:], mode="bilinear", align_corners=False
                )
                atv_mask_lat = F.interpolate(
                    prep["gt_cloth_warped_mask"], size=target_lat.shape[-2:], mode="nearest"
                )

            noise = torch.randn_like(target_lat)
            timesteps = torch.randint(
                0,
                model.scheduler.config.num_train_timesteps,
                (target_lat.shape[0],),
                device=target_lat.device,
            ).long()
            noisy = model.scheduler.add_noise(target_lat, noise, timesteps)
            with autocast(enabled=(dist_info.device.type == "cuda")):
                pred = model(noisy, mask_lat, agnostic_lat, pose_lat, timesteps)
                denoise_loss = F.mse_loss(pred.float(), noise.float())
                if args.use_atv_loss:
                    loss = denoise_loss + args.lambda_atv * tv_loss(pred.float() * atv_mask_lat.float())
                else:
                    loss = denoise_loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            step += 1
            if dist_info.is_main and step % args.log_interval == 0:
                print(
                    f"[step {step:>6}/{args.max_steps}] "
                    f"loss={loss.item():.6f} denoise={denoise_loss.item():.6f}",
                    flush=True,
                )
            if dist_info.is_main and wb_run is not None:
                payload = {
                    "train/loss": float(loss.item()),
                    "train/denoise_loss": float(denoise_loss.item()),
                    "train/step": step,
                }
                if args.use_atv_loss:
                    payload["train/atv_term"] = float((loss - denoise_loss).item())
                wb_run.log(payload, step=step)
            if dist_info.is_main and wb_run is not None and step % args.image_log_interval == 0:
                with torch.no_grad():
                    pred_x0 = model.scheduler.step(pred, timesteps, noisy).pred_original_sample
                    pred_img = model.vae.decode(pred_x0 / model.vae.config.scaling_factor).sample
                payload = {
                    "train/step": step,
                    "images/pred_tryon": _to_wandb_image(pred_img[:8].detach().cpu(), f"StableVTON pred step {step}"),
                    "images/gt_tryon": _to_wandb_image(gt[:8].detach().cpu(), f"StableVTON gt step {step}"),
                    "images/person": _to_wandb_image(person[:8].detach().cpu(), f"StableVTON person step {step}"),
                    "images/cloth": _to_wandb_image(cloth[:8].detach().cpu(), f"StableVTON cloth step {step}"),
                }
                wb_run.log({k: v for k, v in payload.items() if v is not None}, step=step)
            if dist_info.is_main and step % args.save_interval == 0:
                torch.save(
                    {
                        "step": step,
                        "architecture": "StableVITON local: frozen base UNet + trainable SD encoder copy + zero cross-attn adapters",
                        "model_state_dict": model.unet.state_dict(),
                        "sd_encoder_copy_state_dict": model.sd_encoder_copy.state_dict(),
                        "garment_token_proj_state_dict": model.garment_token_proj.state_dict(),
                        "zero_cross_linear_state_dict": model.zero_cross_linear.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "args": vars(args),
                    },
                    os.path.join(args.output_dir, args.run_name, f"ckpt_step_{step}.pt"),
                )
            if step >= args.max_steps:
                break

    if dist_info.is_main:
        torch.save(
            {
                "step": step,
                "architecture": "StableVITON local: frozen base UNet + trainable SD encoder copy + zero cross-attn adapters",
                "model_state_dict": model.unet.state_dict(),
                "sd_encoder_copy_state_dict": model.sd_encoder_copy.state_dict(),
                "garment_token_proj_state_dict": model.garment_token_proj.state_dict(),
                "zero_cross_linear_state_dict": model.zero_cross_linear.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "args": vars(args),
            },
            os.path.join(args.output_dir, args.run_name, "ckpt_final.pt"),
        )
        if wb_run is not None:
            wb_run.finish()
    cleanup_dist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local StableVITON aligned training")
    add_common_args(parser)
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--use_atv_loss", action="store_true", default=False)
    parser.add_argument("--lambda_atv", type=float, default=0.01)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_resume", action="store_true", default=False)
    args = parser.parse_args()
    args.run_name = args.run_name or "train_stable_vton"
    train(args)
