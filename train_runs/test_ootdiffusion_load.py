#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch


def _setup_paths():
    root = Path(__file__).resolve().parents[1]
    cross_arch = root / "cross-architecture"
    for p in (root, cross_arch):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def main():
    parser = argparse.ArgumentParser(description="Smoke-test OOTDiffusion model loading")
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    _setup_paths()
    from OOTDiffusion.train_ootdiffusion_local import OOTDiffusionModel  # noqa: E402

    print(f"[INFO] device={args.device}")
    print(f"[INFO] loading OOTDiffusionModel with model_name={args.model_name}")
    model = OOTDiffusionModel(args.model_name).to(args.device)
    model.train()

    b = 1
    h = 64
    w = 64
    noisy = torch.randn(b, 4, h, w, device=args.device)
    person_lat = torch.randn(b, 4, h, w, device=args.device)
    cloth_lat = torch.randn(b, 4, h, w, device=args.device)
    timesteps = torch.randint(
        0, model.scheduler.config.num_train_timesteps, (b,), device=args.device
    ).long()

    pred = model(noisy, person_lat, cloth_lat, timesteps)
    target = torch.randn_like(pred)
    loss = (pred - target).pow(2).mean()
    loss.backward()

    print("[OK] OOTDiffusionModel loaded successfully.")
    print("[OK] Components: vae, scheduler, denoising_unet, outfitting_unet, outfit_adapter")
    print(f"[OK] forward+backward passed. loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
