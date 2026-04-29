#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def _setup_paths():
    root = Path(__file__).resolve().parents[1]
    cross_arch = root / "cross-architecture"
    for p in (root, cross_arch):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def main():
    parser = argparse.ArgumentParser(description="Smoke-test IDM-VTON model loading")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    parser.add_argument("--pretrained_garmentnet_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--image_encoder_path", type=str, default="ckpt/image_encoder")
    parser.add_argument("--num_tokens", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    _setup_paths()
    from IDMVTON.train_idm_vton_local import IDMVTONModel  # noqa: E402

    ns = SimpleNamespace(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        pretrained_garmentnet_path=args.pretrained_garmentnet_path,
        image_encoder_path=args.image_encoder_path,
        num_tokens=args.num_tokens,
    )

    print(f"[INFO] device={args.device}")
    print("[INFO] loading IDMVTONModel with:")
    print(f"       pretrained_model_name_or_path={ns.pretrained_model_name_or_path}")
    print(f"       pretrained_garmentnet_path={ns.pretrained_garmentnet_path}")
    print(f"       image_encoder_path={ns.image_encoder_path}")
    model = IDMVTONModel(ns).to(args.device)
    model.train()

    b = 1
    h = 64
    w = 64
    noisy = torch.randn(b, 4, h, w, device=args.device)
    person_mask = torch.zeros(b, 1, h, w, device=args.device)
    person_lat = torch.randn(b, 4, h, w, device=args.device)
    pose_lat = torch.randn(b, 4, h, w, device=args.device)
    cloth_lat = torch.randn(b, 4, h, w, device=args.device)
    cloth = torch.randn(b, 3, 512, 512, device=args.device)
    timesteps = torch.randint(
        0, model.scheduler.config.num_train_timesteps, (b,), device=args.device
    ).long()
    captions = ["model is wearing a garment"] * b
    cloth_captions = ["a photo of a garment"] * b

    pred = model(
        noisy,
        person_mask,
        person_lat,
        pose_lat,
        cloth,
        cloth_lat,
        timesteps,
        captions,
        cloth_captions,
    )
    target = torch.randn_like(pred)
    loss = (pred - target).pow(2).mean()
    loss.backward()

    print("[OK] IDMVTONModel loaded successfully.")
    print("[OK] Components: vae, scheduler, tokenizers/text encoders, image_encoder, unet_encoder, unet, image_proj_model")
    print(f"[OK] forward+backward passed. loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
