"""
inference_catvton.py — Self-contained CatVTON inference for the SD v1.5 model.

Architecture:
    cond_input  = cat([person, cloth], dim=3)   [B, 3, 512, 1024]
    cond_latent = VAE(cond_input)               [B, 4, 64, 128]
    UNet input  = cat([noisy(4), cond(4)], dim=1)  [B, 8, 64, 128]
    pred_wide   = VAE_decode(pred_latents)      [B, 3, 512, 1024]
    tryon       = pred_wide[:, :, :, :512]      [B, 3, 512, 512]  ← left half

OOTD mode (--ootd):
    cond_input  = cloth only                    [B, 3, 512, 512]
    output      = full decoded image (no slice)

Usage:
    python inference_catvton.py \
        --checkpoint /path/to/ckpt_final.pt \
        --person     /path/to/person.jpg \
        --cloth      /path/to/cloth.jpg \
        --output     result.png \
        --steps      50
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel

# ── Constants ──────────────────────────────────────────────────────
_BASE_MODEL   = "runwayml/stable-diffusion-v1-5"
_IMAGE_SIZE   = 512


# ── Helpers ────────────────────────────────────────────────────────
def _load_image(path: str, size: int) -> torch.Tensor:
    """Load an RGB image, resize to (size, size), normalise to [-1, 1]."""
    img = Image.open(path).convert("RGB")
    tf  = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return tf(img).unsqueeze(0)   # [1, 3, H, W]


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert a [1, 3, H, W] tensor in [0, 1] to a PIL image."""
    arr = (t[0].permute(1, 2, 0).cpu().float().numpy() * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


# ── Model loading ──────────────────────────────────────────────────
def build_model(base_model: str, dtype: torch.dtype):
    """Load VAE + 8-channel UNet from the base SD checkpoint."""
    print(f"Loading VAE from {base_model} …")
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=dtype)
    vae.requires_grad_(False)

    print(f"Loading UNet from {base_model} …")
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet", torch_dtype=dtype)

    # Expand conv_in from 4→8 channels (same as training)
    old_conv = unet.conv_in   # Conv2d(4, 320, 3, padding=1)
    new_conv = nn.Conv2d(
        8, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    ).to(dtype=dtype)
    with torch.no_grad():
        new_conv.weight[:, :4] = old_conv.weight.clone()
        nn.init.xavier_uniform_(new_conv.weight[:, 4:])
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    unet.conv_in = new_conv
    unet.config["in_channels"] = 8
    unet.requires_grad_(False)

    print("✓ 8-channel UNet built (4 noise + 4 conditioning channels)")
    return vae, unet


def load_checkpoint(unet: UNet2DConditionModel, ckpt_path: str):
    """Load fine-tuned UNet weights from a training checkpoint .pt file."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Checkpoints may wrap state dict under different keys
    if "unet_state_dict" in ckpt:
        state_dict = ckpt["unet_state_dict"]
        step = ckpt.get("step", "?")
        print(f"  ↳ unet_state_dict found  (saved at step {step})")
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        # Assume the checkpoint IS the state dict
        state_dict = ckpt

    # Strip 'module.' prefix added by DDP
    clean = {k.replace("module.", ""): v for k, v in state_dict.items()}

    missing, unexpected = unet.load_state_dict(clean, strict=False)
    if missing:
        print(f"  ⚠ Missing keys  ({len(missing)}): {missing[:5]}{'…' if len(missing)>5 else ''}")
    if unexpected:
        print(f"  ⚠ Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'…' if len(unexpected)>5 else ''}")
    print("✓ Checkpoint loaded")


# ── Inference ──────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(
    vae:      AutoencoderKL,
    unet:     UNet2DConditionModel,
    person:   torch.Tensor,          # [1, 3, H, W]  in [-1, 1]
    cloth:    torch.Tensor,          # [1, 3, H, W]  in [-1, 1]
    device:   torch.device,
    dtype:    torch.dtype,
    steps:    int = 50,
    ootd:     bool = False,
) -> torch.Tensor:                   # returns [1, 3, H, W]  in [0, 1]
    """Full DDIM denoising loop.  Returns try-on image in [0, 1]."""
    person = person.to(device, dtype)
    cloth  = cloth.to(device, dtype)

    # ── Build conditioning input ────────────────────────────────
    if ootd:
        cond_px = cloth                                     # [1, 3, H, W]
    else:
        cond_px = torch.cat([person, cloth], dim=3)         # [1, 3, H, 2W]

    # ── VAE encode ──────────────────────────────────────────────
    cond_latent = vae.encode(cond_px).latent_dist.sample() * 0.18215
    # cond_latent: [1, 4, H/8, W/8]  e.g. [1, 4, 64, 128]

    # ── DDIM noise schedule ─────────────────────────────────────
    scheduler = DDIMScheduler.from_pretrained(_BASE_MODEL, subfolder="scheduler")
    scheduler.set_timesteps(steps, device=device)

    # Start from pure Gaussian noise (same spatial shape as cond_latent)
    latents = torch.randn_like(cond_latent)
    latents = latents * scheduler.init_noise_sigma

    # Zero text embedding — unconditional (no text encoder needed)
    B = latents.shape[0]
    text_emb = torch.zeros(B, 77, 768, device=device, dtype=dtype)  # [B, 77, 768]

    # ── Denoising loop ──────────────────────────────────────────
    print(f"  Running {steps}-step DDIM denoising …")
    for i, t in enumerate(scheduler.timesteps):
        # Channel-concat: [noisy(4) ‖ cond(4)] → [B, 8, H/8, W/8]
        unet_input = torch.cat([latents, cond_latent], dim=1)

        # Scale input for current timestep
        unet_input_scaled = scheduler.scale_model_input(unet_input, t)

        # Predict noise
        noise_pred = unet(unet_input_scaled, t, encoder_hidden_states=text_emb).sample

        # Scheduler step (update noisy latents)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        if (i + 1) % 10 == 0 or i == steps - 1:
            print(f"    step {i+1}/{steps}")

    # ── VAE decode ──────────────────────────────────────────────
    decoded = vae.decode(latents / 0.18215).sample          # [1, 3, H/8*8, W/8*8]
    decoded = (decoded / 2 + 0.5).clamp(0, 1)              # [0, 1]

    # ── Slice try-on (left half) ─────────────────────────────────
    if ootd:
        return decoded                                      # [1, 3, H, W]
    else:
        W = person.shape[3]                                 # 512
        return decoded[:, :, :, :W]                        # [1, 3, H, W]  left half = try-on


# ── CLI ────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="CatVTON inference — SD v1.5 fine-tuned for virtual try-on"
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to trained checkpoint (.pt file, e.g. ckpt_final.pt)")
    p.add_argument("--person",     required=True,
                   help="Path to person / model image (jpg/png)")
    p.add_argument("--cloth",      required=True,
                   help="Path to garment / cloth image (jpg/png)")
    p.add_argument("--output",     default="tryon_result.png",
                   help="Output image path (default: tryon_result.png)")
    p.add_argument("--steps",      type=int, default=50,
                   help="Number of DDIM denoising steps (default: 50)")
    p.add_argument("--size",       type=int, default=_IMAGE_SIZE,
                   help=f"Input/output resolution (default: {_IMAGE_SIZE})")
    p.add_argument("--device",     default="cuda",
                   help="Device: cuda / cpu / cuda:1 etc. (default: cuda)")
    p.add_argument("--fp16",       action="store_true",
                   help="Use float16 (faster on GPU, needs CUDA)")
    p.add_argument("--ootd",       action="store_true",
                   help="OOTD mode: cloth-only conditioning (no person concat)")
    p.add_argument("--base_model", default=_BASE_MODEL,
                   help=f"HuggingFace model ID for base weights (default: {_BASE_MODEL})")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Validate inputs ─────────────────────────────────────────
    for label, path in [("--checkpoint", args.checkpoint),
                        ("--person",     args.person),
                        ("--cloth",      args.cloth)]:
        if not os.path.exists(path):
            print(f"[ERROR] {label} path does not exist: {path}", file=sys.stderr)
            sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    dtype  = torch.float16 if (args.fp16 and device.type == "cuda") else torch.float32
    print(f"Device: {device}  |  dtype: {dtype}")

    # ── Build & load model ──────────────────────────────────────
    vae, unet = build_model(args.base_model, dtype)
    load_checkpoint(unet, args.checkpoint)
    vae.to(device);  unet.to(device)
    vae.eval();      unet.eval()

    # ── Load images ─────────────────────────────────────────────
    print(f"Loading person: {args.person}")
    person = _load_image(args.person, args.size)
    print(f"Loading cloth:  {args.cloth}")
    cloth  = _load_image(args.cloth,  args.size)

    # ── Run inference ───────────────────────────────────────────
    print("\nRunning inference …")
    result = run_inference(
        vae=vae, unet=unet,
        person=person, cloth=cloth,
        device=device, dtype=dtype,
        steps=args.steps,
        ootd=args.ootd,
    )

    # ── Save output ──────────────────────────────────────────────
    out_img = _tensor_to_pil(result)
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    out_img.save(args.output)
    print(f"\n✓ Saved try-on result → {args.output}")
    print(f"  Output size: {out_img.size[0]}×{out_img.size[1]}")


if __name__ == "__main__":
    main()
