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
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from torch.optim import AdamW

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


class StableVTONModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
        self.scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.vae.requires_grad_(False)

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

    @property
    def cross_attention_dim(self):
        dim = self.unet.config.cross_attention_dim
        return int(dim[0] if isinstance(dim, (list, tuple)) else dim)

    def encode(self, image):
        latents = self.vae.encode(image).latent_dist.sample()
        return latents * self.vae.config.scaling_factor

    def forward(self, noisy, mask_lat, agnostic_lat, pose_lat, timesteps):
        hidden = torch.zeros(
            noisy.shape[0],
            77,
            self.cross_attention_dim,
            device=noisy.device,
            dtype=noisy.dtype,
        )
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
    optimizer = AdamW(model.unet.parameters(), lr=args.lr)

    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_to_load = args.resume
    if ckpt_to_load is None and not args.no_resume:
        ckpt_to_load = latest_checkpoint(run_dir)

    step = 0
    if ckpt_to_load:
        ckpt = torch.load(ckpt_to_load, map_location=dist_info.device)
        model.unet.load_state_dict(ckpt["model_state_dict"])
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
            pred = model(noisy, mask_lat, agnostic_lat, pose_lat, timesteps)
            denoise_loss = F.mse_loss(pred.float(), noise.float())
            if args.use_atv_loss:
                loss = denoise_loss + args.lambda_atv * tv_loss(pred.float() * atv_mask_lat.float())
            else:
                loss = denoise_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1
            if dist_info.is_main and step % args.save_interval == 0:
                save_checkpoint(
                    os.path.join(args.output_dir, args.run_name, f"ckpt_{step}.pt"),
                    model.unet,
                    optimizer,
                    step,
                    {"architecture": "StableVITON local 13-channel latent diffusion"},
                )
            if step >= args.max_steps:
                break

    if dist_info.is_main:
        save_checkpoint(
            os.path.join(args.output_dir, args.run_name, "ckpt_final.pt"),
            model.unet,
            optimizer,
            step,
        )
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
