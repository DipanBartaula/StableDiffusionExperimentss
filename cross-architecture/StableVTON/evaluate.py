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
from train_stable_vton_local import StableVTONModel, stableviton_preprocess  # noqa: E402


def build_predict_fn(model, num_inference_steps: int):
    @torch.no_grad()
    def _predict(batch, device):
        person = batch.get("person", batch.get("masked_person")).to(device)
        cloth = batch["cloth"].to(device)
        gt = batch["ground_truth"].to(device)
        prep = stableviton_preprocess(person, cloth, gt)
        agnostic_lat = model.encode(prep["agnostic"])
        pose_lat = model.encode(prep["pose_img"])
        latents = torch.randn_like(agnostic_lat)
        mask_lat = F.interpolate(prep["agnostic_mask"], size=latents.shape[-2:], mode="bilinear", align_corners=False)
        model.scheduler.set_timesteps(num_inference_steps, device=device)
        for t in model.scheduler.timesteps:
            t_batch = torch.full((latents.shape[0],), int(t), device=device, dtype=torch.long)
            noise_pred = model(latents, mask_lat, agnostic_lat, pose_lat, t_batch)
            latents = model.scheduler.step(noise_pred, t, latents).prev_sample
        return model.vae.decode(latents / model.vae.config.scaling_factor).sample

    return _predict


def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = StableVTONModel(args.model_name).to(device).eval()

    if args.use_init_weights:
        print("Using initial StableVTON weights (no checkpoint load).")
    elif args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.unet.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("Using initial StableVTON weights (no checkpoint load).")
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
    print("\nEvaluation metrics:\n" + json.dumps(results, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved metrics JSON to: {args.output_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser("Evaluate StableVTON on CurvTON/Triplet/StreetTryOn")
    p.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--output_dir", type=str, default="runs/cross_architecture")
    p.add_argument("--run_name", type=str, default="train_stable_vton")
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
