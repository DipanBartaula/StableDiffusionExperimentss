import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

THIS_DIR = os.path.dirname(__file__)
STABLE_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
CROSS_ARCH_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
for p in (STABLE_DIR, CROSS_ARCH_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import CURVTON_TEST_PATH, STREET_TRYON_PATH, TRIPLET_TEST_PATH  # noqa: E402
from eval_common import build_eval_loaders, evaluate_all_splits  # noqa: E402
from common import latest_checkpoint  # noqa: E402
from train_idm_vton_local import IDMVTONModel  # noqa: E402


def build_predict_fn(model, num_inference_steps: int):
    @torch.no_grad()
    def _predict(batch, device):
        person = batch.get("person", batch.get("masked_person")).to(device)
        cloth = batch["cloth"].to(device)
        person_lat = model.encode(person)
        pose_lat = model.encode(person)
        cloth_lat = model.encode(cloth)
        person_mask = F.interpolate(person.mean(dim=1, keepdim=True), size=person_lat.shape[-2:], mode="bilinear", align_corners=False)
        latents = torch.randn_like(person_lat)
        captions = ["model is wearing a garment"] * latents.shape[0]
        cloth_captions = ["a photo of a garment"] * latents.shape[0]
        model.scheduler.set_timesteps(num_inference_steps, device=device)
        for t in model.scheduler.timesteps:
            t_batch = torch.full((latents.shape[0],), int(t), device=device, dtype=torch.long)
            noise_pred = model(latents, person_mask, person_lat, pose_lat, cloth, cloth_lat, t_batch, captions, cloth_captions)
            latents = model.scheduler.step(noise_pred, t, latents).prev_sample
        return model.vae.decode(latents / model.vae.config.scaling_factor).sample

    return _predict


def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = IDMVTONModel(args).to(device).eval()

    if args.use_init_weights:
        print("Using initial IDM-VTON weights (no checkpoint load).")
    else:
        run_dir = os.path.join(args.output_dir, args.run_name)
        ckpt_path = args.checkpoint or latest_checkpoint(run_dir)
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoint found in {run_dir}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.unet.load_state_dict(ckpt["unet_state_dict"])
        if "image_proj_state_dict" in ckpt:
            model.image_proj_model.load_state_dict(ckpt["image_proj_state_dict"])
        print(f"Loaded checkpoint: {ckpt_path}")
    loaders = build_eval_loaders(
        curvton_test_data_path=args.curvton_test_data_path,
        triplet_test_data_path=args.triplet_test_data_path,
        street_tryon_data_path=args.street_tryon_data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gender=args.gender,
        street_split=args.street_split,
    )
    results = evaluate_all_splits(
        loaders=loaders,
        predict_fn=build_predict_fn(model, args.num_inference_steps),
        device=device,
        max_batches=args.max_batches,
        eval_frac_curvton=args.eval_frac_curvton,
        eval_frac_triplet=args.eval_frac_triplet,
        eval_frac_street=args.eval_frac_street,
    )
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved metrics JSON to: {args.output_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser("Evaluate IDM-VTON on CurvTON/Triplet/StreetTryOn")
    p.add_argument("--pretrained_model_name_or_path", type=str, default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    p.add_argument("--pretrained_garmentnet_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--image_encoder_path", type=str, default="ckpt/image_encoder")
    p.add_argument("--num_tokens", type=int, default=16)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--output_dir", type=str, default="runs/cross_architecture")
    p.add_argument("--run_name", type=str, default="train_idm_vton")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--use_init_weights", action="store_true", default=False)
    p.add_argument("--curvton_test_data_path", type=str, default=CURVTON_TEST_PATH)
    p.add_argument("--triplet_test_data_path", type=str, default=TRIPLET_TEST_PATH)
    p.add_argument("--street_tryon_data_path", type=str, default=STREET_TRYON_PATH)
    p.add_argument("--street_split", type=str, default="validation", choices=["train", "validation"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--gender", type=str, default="all", choices=["female", "male", "all"])
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--max_batches", type=int, default=0, help="0 = full dataset")
    p.add_argument("--eval_frac_curvton", type=float, default=0.10)
    p.add_argument("--eval_frac_triplet", type=float, default=0.30)
    p.add_argument("--eval_frac_street", type=float, default=0.30)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output_json", type=str, default=None)
    main(p.parse_args())
