"""
evaluate.py — Standalone checkpoint evaluation script.

Loads a trained checkpoint and runs evaluation on specified test datasets
(CurvTon and/or Triplet). Results are printed to stdout and optionally
logged to Weights & Biases.

Usage examples:

  # Evaluate on CurvTon test set only
  python evaluate.py \
      --checkpoint /path/to/ckpt_step_5000.pt \
      --curvton_test_data_path /path/to/dataset_ultimate_test

  # Evaluate on Triplet test set only
  python evaluate.py \
      --checkpoint /path/to/ckpt_step_5000.pt \
      --triplet_test_data_path /path/to/triplet_dataset

  # Evaluate on both datasets
  python evaluate.py \
      --checkpoint /path/to/ckpt_step_5000.pt \
      --curvton_test_data_path /path/to/dataset_ultimate_test \
      --triplet_test_data_path /path/to/triplet_dataset

  # Multi-GPU evaluation via torchrun
  torchrun --nproc_per_node=4 evaluate.py \
      --checkpoint /path/to/ckpt_step_5000.pt \
      --curvton_test_data_path /path/to/dataset_ultimate_test
"""

import os
import sys
import json
import argparse

import torch
import torch.distributed as dist

from config import IMAGE_SIZE, MODEL_NAME, WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY
from model import SDModel
from utils import (
    get_curvton_test_dataloaders,
    get_triplet_test_dataloaders,
    evaluate_on_test,
)


def main(args):
    # ── Distributed setup ──────────────────────────────────────
    rank       = int(os.environ.get("RANK",       0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main    = (rank == 0)

    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main:
        print(f"Device: {device}  |  rank={rank}/{world_size}")
        print(f"Checkpoint: {args.checkpoint}")

    # ── Load model ─────────────────────────────────────────────
    model = SDModel().to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    unet_sd = ckpt["unet_state_dict"]
    # Strip DDP 'module.' prefix if present
    cleaned = {}
    for k, v in unet_sd.items():
        cleaned[k.removeprefix("module.")] = v

    missing, unexpected = model.unet.load_state_dict(cleaned, strict=False)
    if is_main:
        print(f"Loaded UNet weights from step {ckpt.get('step', '?')}")
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

    model.unet.eval()

    # ── Build test loaders ─────────────────────────────────────
    test_loaders: dict = {}

    if args.curvton_test_data_path:
        genders = ("female", "male") if args.gender == "all" else (args.gender,)
        if is_main:
            print(f"\nBuilding CurvTon test loaders from {args.curvton_test_data_path} ...")
        curvton_test = get_curvton_test_dataloaders(
            root_dir=args.curvton_test_data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            size=IMAGE_SIZE,
            genders=genders,
        )
        for k, v in curvton_test.items():
            if k == "all":
                continue
            test_loaders[f"curvton_{k}"] = v
        if is_main:
            print(f"CurvTon test loaders: {list(k for k in test_loaders if k.startswith('curvton_'))}")

    if args.triplet_test_data_path:
        if is_main:
            print(f"\nBuilding Triplet test loaders from {args.triplet_test_data_path} ...")
        triplet_test = get_triplet_test_dataloaders(
            root_dir=args.triplet_test_data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            size=IMAGE_SIZE,
        )
        test_loaders.update(triplet_test)
        if is_main:
            print(f"Triplet test loaders: {list(triplet_test.keys())}")

    if not test_loaders:
        print("ERROR: No test data paths provided. "
              "Use --curvton_test_data_path and/or --triplet_test_data_path.")
        sys.exit(1)

    # ── Run evaluation ─────────────────────────────────────────
    if is_main:
        print(f"\nRunning evaluation on {list(test_loaders.keys())} ...")
        print(f"  eval_frac={args.eval_frac}, n_samples={args.n_samples}, "
              f"num_inference_steps={args.num_inference_steps}")

    metrics = evaluate_on_test(
        model, test_loaders, device,
        num_inference_steps=args.num_inference_steps,
        eval_frac=args.eval_frac,
        ootd=args.ootd,
        rank=rank,
        world_size=world_size,
        num_eval_steps=args.num_inference_steps,
        n_samples=args.n_samples,
    )

    # ── Print & save results ───────────────────────────────────
    if is_main:
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        for key in sorted(metrics.keys()):
            print(f"  {key}: {metrics[key]:.6f}")
        print("=" * 60)

        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"\nResults saved to {args.output_json}")

        if args.wandb:
            import wandb
            wandb.login(key=WANDB_API_KEY)
            run_name = args.wandb_run_name or f"eval_{os.path.basename(args.checkpoint)}"
            wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY,
                       name=run_name, job_type="eval")
            wandb.log(metrics)
            wandb.finish()
            print("Results logged to W&B")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone evaluation of a trained virtual try-on checkpoint")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a .pt checkpoint file")
    # Dataset paths
    parser.add_argument("--curvton_test_data_path", type=str, default=None,
                        help="Path to CurvTon test dataset root (easy/medium/hard splits)")
    parser.add_argument("--triplet_test_data_path", type=str, default=None,
                        help="Path to triplet_dataset root (dresscode/viton-hd test splits)")
    # Evaluation parameters
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for evaluation (default: 8)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader workers (default: 8)")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps for inference (default: 50)")
    parser.add_argument("--eval_frac", type=float, default=0.01,
                        help="Fraction of test set per evaluation sample (default: 0.01)")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Number of bootstrap eval samples for mean/std (default: 10)")
    parser.add_argument("--gender", type=str, default="all",
                        choices=["female", "male", "all"],
                        help="CurvTon gender subset (default: all)")
    parser.add_argument("--ootd", action="store_true", default=False,
                        help="OOTDiffusion mode: cloth-only conditioning")
    # Output
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to save results as JSON (optional)")
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Log results to Weights & Biases")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name for eval logging")

    args = parser.parse_args()
    main(args)
