import argparse
import json
import os
import sys

import torch

THIS_DIR = os.path.dirname(__file__)
STABLE_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
CROSS_ARCH_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
for p in (STABLE_DIR, CROSS_ARCH_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import CURVTON_TEST_PATH, STREET_TRYON_PATH, TRIPLET_TEST_PATH  # noqa: E402
from eval_common import build_eval_loaders, evaluate_all_splits  # noqa: E402
from train_cpvton_local import GMM, TOM  # noqa: E402


def build_predict_fn(gmm, tom, stage: str):
    @torch.no_grad()
    def _predict(batch, device):
        person = batch.get("person", batch.get("masked_person")).to(device)
        cloth = batch["cloth"].to(device)
        gt = batch["ground_truth"].to(device)
        warped, _, _ = gmm(person, cloth)
        if stage == "GMM" or tom is None:
            return warped
        mask = (person - gt).abs().mean(dim=1, keepdim=True).clamp(0, 1)
        person_agnostic = person * (1 - mask)
        final, _, _ = tom(person_agnostic, warped)
        return final

    return _predict


def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    gmm = GMM().to(device).eval()
    tom = TOM().to(device).eval() if args.stage == "TOM" else None
    if args.use_init_weights:
        print("Using initial CPVTON weights (no checkpoint load).")
    elif args.gmm_checkpoint or args.tom_checkpoint:
        if args.gmm_checkpoint:
            gmm_state = torch.load(args.gmm_checkpoint, map_location=device)
            gmm.load_state_dict(gmm_state["model_state_dict"], strict=False)
            print(f"Loaded GMM checkpoint: {args.gmm_checkpoint}")
        if args.stage == "TOM" and args.tom_checkpoint:
            tom_state = torch.load(args.tom_checkpoint, map_location=device)
            tom.load_state_dict(tom_state["model_state_dict"], strict=False)
            print(f"Loaded TOM checkpoint: {args.tom_checkpoint}")
    else:
        print("Using initial CPVTON weights (no checkpoint load).")

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
        predict_fn=build_predict_fn(gmm, tom, args.stage),
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
    p = argparse.ArgumentParser("Evaluate CPVTON on CurvTON/Triplet/StreetTryOn")
    p.add_argument("--output_dir", type=str, default="runs/cross_architecture")
    p.add_argument("--run_name", type=str, default="train_cpvton_tom")
    p.add_argument("--stage", type=str, default="TOM", choices=["GMM", "TOM"])
    p.add_argument("--gmm_checkpoint", type=str, default=None)
    p.add_argument("--tom_checkpoint", type=str, default=None)
    p.add_argument("--use_init_weights", action="store_true", default=False)
    p.add_argument("--curvton_test_data_path", type=str, default=CURVTON_TEST_PATH)
    p.add_argument("--triplet_test_data_path", type=str, default=TRIPLET_TEST_PATH)
    p.add_argument("--street_tryon_data_path", type=str, default=STREET_TRYON_PATH)
    p.add_argument("--street_split", type=str, default="validation", choices=["train", "validation"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--gender", type=str, default="all", choices=["female", "male", "all"])
    p.add_argument("--max_batches", type=int, default=0, help="0 = full dataset")
    p.add_argument("--eval_frac_curvton", type=float, default=0.10)
    p.add_argument("--eval_frac_triplet", type=float, default=0.30)
    p.add_argument("--eval_frac_street", type=float, default=0.30)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output_json", type=str, default=None)
    main(p.parse_args())
