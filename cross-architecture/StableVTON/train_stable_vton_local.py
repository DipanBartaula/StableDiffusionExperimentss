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
from transformers import CLIPVisionModelWithProjection

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


class ZeroCrossAttentionBlock(nn.Module):
    """Per-decoder zero-initialized cross-attention block."""

    def __init__(self, dim, cross_dim, heads=8):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(cross_dim)
        num_heads = max(1, min(heads, dim // 32 if dim >= 32 else 1))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            kdim=cross_dim,
            vdim=cross_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.zero_linear = nn.Linear(dim, dim)
        nn.init.zeros_(self.zero_linear.weight)
        nn.init.zeros_(self.zero_linear.bias)

    def forward(self, x_tokens, garment_tokens):
        q = self.norm_q(x_tokens)
        kv = self.norm_kv(garment_tokens)
        x_attn, _ = self.cross_attn(q, kv, kv, need_weights=False)
        x = x_tokens + x_attn
        x = x + self.ffn(self.norm_ffn(x))
        x = x + self.zero_linear(x)
        return x


class StableVTONModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
        self.scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            model_name, subfolder="image_encoder"
        )
        self.vae.requires_grad_(False)
        self.clip_image_encoder.requires_grad_(False)

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

        # CLIP garment pathway + projection into UNet cross-attn space.
        clip_proj_dim = int(self.clip_image_encoder.config.projection_dim)
        self.clip_to_cross = nn.Linear(clip_proj_dim, self.cross_attention_dim)
        self.garment_token_proj = nn.Linear(self.cross_attention_dim, self.cross_attention_dim, bias=True)
        nn.init.zeros_(self.garment_token_proj.weight)
        nn.init.zeros_(self.garment_token_proj.bias)

        # True per-decoder zero-cross blocks attached to each decoder/up block.
        self.decoder_zero_blocks = nn.ModuleList()
        self._decoder_hook_handles = []
        self._hook_context_tokens = None
        for up_block in self.unet.up_blocks:
            ch = int(up_block.resnets[-1].out_channels)
            self.decoder_zero_blocks.append(ZeroCrossAttentionBlock(dim=ch, cross_dim=self.cross_attention_dim))
        self._register_decoder_hooks()

        # Freeze base UNet so training focuses on SD encoder copy + zero-cross blocks.
        self.unet.requires_grad_(False)
        self.clip_to_cross.requires_grad_(True)
        self.garment_token_proj.requires_grad_(True)
        self.decoder_zero_blocks.requires_grad_(True)

    def _register_decoder_hooks(self):
        for h in self._decoder_hook_handles:
            h.remove()
        self._decoder_hook_handles = []
        for idx, up_block in enumerate(self.unet.up_blocks):
            def _make_hook(i):
                def _hook(_module, _inputs, output):
                    if self._hook_context_tokens is None or not torch.is_tensor(output):
                        return output
                    b, c, h, w = output.shape
                    x_tokens = output.flatten(2).transpose(1, 2)
                    x_tokens = self.decoder_zero_blocks[i](x_tokens, self._hook_context_tokens)
                    return x_tokens.transpose(1, 2).reshape(b, c, h, w)
                return _hook
            self._decoder_hook_handles.append(up_block.register_forward_hook(_make_hook(idx)))

    @property
    def cross_attention_dim(self):
        dim = self.unet.config.cross_attention_dim
        return int(dim[0] if isinstance(dim, (list, tuple)) else dim)

    def encode(self, image):
        latents = self.vae.encode(image).latent_dist.sample()
        return latents * self.vae.config.scaling_factor

    def _encode_garment_clip_tokens(self, cloth_img, dtype):
        pix = ((cloth_img.clamp(-1, 1) + 1.0) * 0.5).float()
        pix = F.interpolate(pix, size=(224, 224), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=pix.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=pix.device).view(1, 3, 1, 1)
        pix = (pix - mean) / std
        embeds = self.clip_image_encoder(pixel_values=pix).image_embeds
        return self.garment_token_proj(self.clip_to_cross(embeds)).unsqueeze(1).to(dtype=dtype)

    def forward(self, noisy, mask_lat, agnostic_lat, pose_lat, cloth_img, timesteps):
        hidden_base = torch.zeros(
            noisy.shape[0],
            77,
            self.cross_attention_dim,
            device=noisy.device,
            dtype=noisy.dtype,
        )
        # Trainable SD encoder-copy + explicit CLIP garment-image pathway.
        cloth_lat = self.encode(cloth_img)
        garment_feat = self.sd_encoder_copy(cloth_lat, timesteps, hidden_base).sample
        garment_lat_tokens = F.adaptive_avg_pool2d(garment_feat, (4, 4)).flatten(2).transpose(1, 2)
        if garment_lat_tokens.shape[-1] < self.cross_attention_dim:
            pad = self.cross_attention_dim - garment_lat_tokens.shape[-1]
            garment_lat_tokens = F.pad(garment_lat_tokens, (0, pad), value=0.0)
        garment_lat_tokens = garment_lat_tokens[:, :, : self.cross_attention_dim]
        clip_tokens = self._encode_garment_clip_tokens(cloth_img, noisy.dtype)
        garment_tokens = torch.cat([garment_lat_tokens, clip_tokens], dim=1)
        hidden = torch.cat([hidden_base, garment_tokens], dim=1)

        # StableVITON/PBE-style 13ch path: noisy(4) + mask(1) + agnostic(4) + pose(4)
        x = torch.cat([noisy, mask_lat, agnostic_lat, pose_lat], dim=1)
        self._hook_context_tokens = garment_tokens
        try:
            return self.unet(x, timesteps, hidden).sample
        finally:
            self._hook_context_tokens = None


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
        + list(model.clip_to_cross.parameters())
        + list(model.garment_token_proj.parameters())
        + list(model.decoder_zero_blocks.parameters())
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
        if "clip_to_cross_state_dict" in ckpt:
            model.clip_to_cross.load_state_dict(ckpt["clip_to_cross_state_dict"], strict=False)
        if "decoder_zero_blocks_state_dict" in ckpt:
            model.decoder_zero_blocks.load_state_dict(ckpt["decoder_zero_blocks_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step = int(ckpt.get("step", 0))

    @torch.no_grad()
    def _sample_tryon(agnostic_lat, pose_lat, mask_lat, cloth_img, n_steps):
        latents = torch.randn_like(agnostic_lat)
        model.scheduler.set_timesteps(n_steps, device=latents.device)
        for t in model.scheduler.timesteps:
            t_batch = torch.full((latents.shape[0],), int(t), device=latents.device, dtype=torch.long)
            noise_pred = model(latents, mask_lat, agnostic_lat, pose_lat, cloth_img, t_batch)
            latents = model.scheduler.step(noise_pred, t, latents).prev_sample
        return model.vae.decode(latents / model.vae.config.scaling_factor).sample

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
                pred = model(noisy, mask_lat, agnostic_lat, pose_lat, cloth, timesteps)
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
                    k = min(8, agnostic_lat.shape[0])
                    pred_img = _sample_tryon(
                        agnostic_lat[:k],
                        pose_lat[:k],
                        mask_lat[:k],
                        cloth[:k],
                        args.num_inference_steps,
                    )
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
                        "clip_to_cross_state_dict": model.clip_to_cross.state_dict(),
                        "garment_token_proj_state_dict": model.garment_token_proj.state_dict(),
                        "decoder_zero_blocks_state_dict": model.decoder_zero_blocks.state_dict(),
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
                "clip_to_cross_state_dict": model.clip_to_cross.state_dict(),
                "garment_token_proj_state_dict": model.garment_token_proj.state_dict(),
                "decoder_zero_blocks_state_dict": model.decoder_zero_blocks.state_dict(),
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
    parser.add_argument("--use_atv_loss", action="store_true", default=True)
    parser.add_argument("--no_atv_loss", action="store_false", dest="use_atv_loss")
    parser.add_argument("--lambda_atv", type=float, default=0.01)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_resume", action="store_true", default=False)
    args = parser.parse_args()
    args.run_name = args.run_name or "train_stable_vton"
    train(args)
