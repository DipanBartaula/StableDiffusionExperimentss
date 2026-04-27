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
- No agnostic mask is used.
"""

import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW

from common import (
    add_common_args,
    batch_images,
    build_curvton_loader,
    cleanup_dist,
    latest_checkpoint,
    setup_dist,
    wrap_ddp,
)


def raw_module(module):
    return module.module if hasattr(module, "module") else module


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
    loader, sampler = build_curvton_loader(args, dist_info)
    params = list(model.denoising_unet.parameters()) + list(model.outfitting_unet.parameters()) + list(model.outfit_adapter.parameters())
    optimizer = AdamW(params, lr=args.lr)
    scaler = GradScaler(enabled=(dist_info.device.type == "cuda"))

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

    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for batch in loader:
            person, cloth, gt = batch_images(batch, dist_info.device)
            with torch.no_grad():
                target_lat = model.encode(gt)
                person_lat = model.encode(person)
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
            if dist_info.is_main and step % args.save_interval == 0:
                os.makedirs(os.path.join(args.output_dir, args.run_name), exist_ok=True)
                torch.save({
                    "step": step,
                    "architecture": "OOTDiffusion denoising UNet + outfitting UNet + outfitting dropout",
                    "denoising_unet_state_dict": raw_module(model.denoising_unet).state_dict(),
                    "outfitting_unet_state_dict": raw_module(model.outfitting_unet).state_dict(),
                    "outfit_adapter_state_dict": model.outfit_adapter.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, os.path.join(args.output_dir, args.run_name, f"ckpt_{step}.pt"))
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
        }, os.path.join(args.output_dir, args.run_name, "ckpt_final.pt"))
    cleanup_dist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local OOTDiffusion architecture trainer")
    add_common_args(parser)
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--outfitting_dropout", type=float, default=0.1)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_resume", action="store_true", default=False)
    args = parser.parse_args()
    args.run_name = args.run_name or "train_ootdiffusion"
    train(args)
