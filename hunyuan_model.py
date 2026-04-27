"""
hunyuan_model.py — HunyuanDiT v1.1 for CatVTON-style virtual try-on.

Architecture overview
─────────────────────
•  VAE        : Standard SD AutoencoderKL — 4 latent channels, scale=0.18215
•  Denoiser   : HunyuanDiT2DModel (DiT) — UNMODIFIED architecture
•  Scheduler  : DDPMScheduler (training noise)  +
                UniPCMultistepScheduler (inference denoising)

CatVTON-style conditioning (spatial concatenation)
──────────────────────────────────────────────────
NO architecture modifications needed. The conditioning is spatially concatenated
with the noisy target latents along the width axis so the DiT processes a wider
image — exactly as in the Flux CatVTON approach, but without sequence packing
(HunyuanDiT2DModel takes 2D [B, C, H, W] inputs directly).

    cond_lat    = VAE(cat([person, cloth], W))    [B, 4, H, 2W]
    target_lat  = VAE(cat([gt,     cloth], W))    [B, 4, H, 2W]
    noisy       = DDPM.add_noise(target_lat, ε, t) [B, 4, H, 2W]
    full_input  = cat([noisy, cond_lat], W)         [B, 4, H, 4W]
    → DiT → take left half → ε loss

Rotary positional embeddings (RoPE) are recomputed for the wider input H × 4W.

PREREQUISITES:
  • No gated license — public model on HuggingFace.
  • ~10–14 GB VRAM in float16 with gradient checkpointing.
"""

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, DDPMScheduler, UniPCMultistepScheduler
from diffusers.models.transformers import HunyuanDiT2DModel

from config import HUNYUAN_MODEL_NAME


# ============================================================
# ROTARY POSITIONAL EMBEDDING HELPER
# ============================================================

def prepare_rotary_pos_embed(
    H_lat: int, W_lat: int,
    head_dim: int, patch_size: int,
    device, dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute 2-D RoPE (cos, sin) embeddings for a latent of spatial size [H_lat, W_lat].

    The HunyuanDiT patchifies with `patch_size`×`patch_size` patches, so the grid is
    [H_lat // patch_size, W_lat // patch_size].

    Returns:
        cos, sin  — each [grid_h * grid_w, head_dim // 2]  on `device` / `dtype`
    """
    try:
        from diffusers.models.embeddings import get_2d_rotary_pos_embed
    except ImportError:
        from diffusers.models.attention import get_2d_rotary_pos_embed   # older diffusers

    grid_h = H_lat // patch_size
    grid_w = W_lat // patch_size

    cos, sin = get_2d_rotary_pos_embed(
        embed_dim=head_dim,
        crops_coords=((0, 0), (grid_h, grid_w)),
        grid_size=(grid_h, grid_w),
        use_real=True,
    )
    cos = torch.tensor(cos, device=device, dtype=dtype)
    sin = torch.tensor(sin, device=device, dtype=dtype)
    return cos, sin


# ============================================================
# HUNYUANDIT MODEL WRAPPER
# ============================================================

class HunyuanDiTModel:
    """
    Wraps HunyuanDiT v1.1 components for CatVTON-style virtual try-on.

    Key differences from Flux:
      • 4-channel VAE (standard SD),  not 16-channel
      • DDPM noise schedule (epsilon prediction),  not flow matching
      • 2D spatial DiT input — NO sequence packing / unpacking
      • RoPE positional embeddings recomputed per spatial size
      • ~1.5 B params in float16 (~6 GB VRAM)
    """

    def __init__(self, dtype=torch.float16, gradient_checkpointing: bool = True):
        self.dtype = dtype
        print(f"Loading HunyuanDiT v1.1 from {HUNYUAN_MODEL_NAME} (dtype={dtype}) …")

        # ── VAE (4-channel, SD-style) ───────────────────────────
        self.vae = AutoencoderKL.from_pretrained(
            HUNYUAN_MODEL_NAME, subfolder="vae", torch_dtype=dtype,
        )
        self.vae.requires_grad_(False)
        self.vae_scale: float = self.vae.config.scaling_factor      # 0.18215

        # ── DiT — unmodified ────────────────────────────────────
        self.transformer = HunyuanDiT2DModel.from_pretrained(
            HUNYUAN_MODEL_NAME, subfolder="transformer", torch_dtype=dtype,
        )
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
            print("   ✓ Gradient checkpointing enabled on Transformer")

        # ── Schedulers ──────────────────────────────────────────
        # DDPMScheduler: used during training to add noise
        self.scheduler = DDPMScheduler.from_pretrained(
            HUNYUAN_MODEL_NAME, subfolder="scheduler",
        )
        # UniPC: faster inference than DDIM at same quality
        self.inference_scheduler = UniPCMultistepScheduler.from_config(
            self.scheduler.config,
        )

        # ── Derived config ──────────────────────────────────────
        cfg = self.transformer.config
        self.patch_size:     int = getattr(cfg, "patch_size",            2)
        self.num_heads:      int = getattr(cfg, "num_attention_heads",  16)
        self.hidden_size:    int = getattr(cfg, "hidden_size",        1408)
        self.head_dim:       int = self.hidden_size // self.num_heads   # 88
        self.cross_attn_dim: int = getattr(cfg, "cross_attention_dim", 1024)

        _n = sum(p.numel() for p in self.transformer.parameters())
        print(
            f"✓ HunyuanDiT v1.1 loaded  "
            f"(VAE: 4 ch, scale={self.vae_scale:.5f};  "
            f"Transformer: {_n:,} params, "
            f"head_dim={self.head_dim}, patch={self.patch_size}×{self.patch_size})"
        )

    # ── device helpers ──────────────────────────────────────────
    def to(self, device):
        self.vae.to(device)
        self.transformer.to(device)
        return self

    # ── latent encode / decode ──────────────────────────────────
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Pixel-space image ([-1,1]) → SD latent (scaled)."""
        latent = self.vae.encode(image).latent_dist.sample()
        return latent * self.vae_scale

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """SD latent → pixel-space image clamped to [0,1]."""
        image = self.vae.decode(latent / self.vae_scale).sample
        return (image / 2 + 0.5).clamp(0, 1)

    # ── RoPE helper ─────────────────────────────────────────────
    def get_rope_embed(
        self, H_lat: int, W_lat: int, device, dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE (cos, sin) for a latent of size [H_lat, W_lat]."""
        return prepare_rotary_pos_embed(
            H_lat, W_lat, self.head_dim, self.patch_size, device, dtype,
        )


# ============================================================
# PARAMETER UTILITIES
# ============================================================

def count_parameters_hunyuan(transformer, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    return sum(p.numel() for p in transformer.parameters())


def freeze_non_attention_hunyuan(transformer):
    """Freeze everything except attention projections in the HunyuanDiT."""
    for param in transformer.parameters():
        param.requires_grad = False

    attn_modules = []
    for name, module in transformer.named_modules():
        if any(k in name for k in ("attn", "attention")):
            for param in module.parameters():
                param.requires_grad = True
            attn_modules.append(name)

    print(f"✓ Unfroze {len(attn_modules)} attention modules in HunyuanDiT")
    return transformer


def print_trainable_params_hunyuan(model: HunyuanDiTModel, mode: str) -> int:
    """Print detailed trainable parameter stats for the HunyuanDiT."""
    print("\n" + "=" * 60)
    print(f"TRAINABLE PARAMETERS — HunyuanDiT v1.1 ({mode})")
    print("=" * 60)

    total     = count_parameters_hunyuan(model.transformer, trainable_only=False)
    trainable = count_parameters_hunyuan(model.transformer, trainable_only=True)
    frozen    = total - trainable

    print(f"\n  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Frozen parameters:    {frozen:,}")
    print(f"  Trainable ratio:      {100 * trainable / total:.2f}%")

    print(f"\n  Top trainable layer groups:")
    layer_counts: dict[str, int] = {}
    for name, param in model.transformer.named_parameters():
        if param.requires_grad:
            parts = name.split(".")
            key = ".".join(parts[:3]) if len(parts) > 3 else name
            layer_counts[key] = layer_counts.get(key, 0) + param.numel()

    for layer, count in sorted(layer_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"      {layer}: {count:,}")
    if len(layer_counts) > 20:
        print(f"      … and {len(layer_counts) - 20} more groups")

    return trainable
