import argparse
import os
import sys

import torch
from PIL import Image


THIS_DIR = os.path.dirname(__file__)
CROSS_ARCH_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if CROSS_ARCH_DIR not in sys.path:
    sys.path.insert(0, CROSS_ARCH_DIR)

from common import DistInfo, batch_images, build_curvton_loader, latest_stage_checkpoint  # noqa: E402
from train_cpvton_local import GMM, TOM  # noqa: E402


def save_tensor_image(tensor, path):
    x = ((tensor.detach().cpu().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
    x = x.permute(1, 2, 0).numpy()
    Image.fromarray(x).save(path)


def infer(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.save_dir, exist_ok=True)

    gmm_ckpt = args.gmm_checkpoint or latest_stage_checkpoint(run_dir, "gmm")
    if gmm_ckpt is None:
        raise FileNotFoundError(f"No GMM checkpoint found in {run_dir}")
    gmm = GMM().to(device).eval()
    gmm_state = torch.load(gmm_ckpt, map_location=device)
    gmm.load_state_dict(gmm_state["model_state_dict"], strict=False)

    tom = None
    if args.stage == "TOM":
        tom_ckpt = args.tom_checkpoint or latest_stage_checkpoint(run_dir, "tom")
        if tom_ckpt is None:
            raise FileNotFoundError(f"No TOM checkpoint found in {run_dir}")
        tom = TOM().to(device).eval()
        tom_state = torch.load(tom_ckpt, map_location=device)
        tom.load_state_dict(tom_state["model_state_dict"], strict=False)

    loader_args = argparse.Namespace(
        curvton_data_path=args.curvton_data_path,
        difficulty=args.difficulty,
        gender=args.gender,
        batch_size=1,
        num_workers=args.num_workers,
    )
    dist_info = DistInfo(rank=0, local_rank=0, world_size=1, device=device, is_main=True)
    loader, _ = build_curvton_loader(loader_args, dist_info)

    written = 0
    with torch.no_grad():
        for batch in loader:
            person, cloth, gt = batch_images(batch, device)
            warped, _, _ = gmm(person, cloth)
            if args.stage == "GMM":
                out = warped
            else:
                mask = (person - gt).abs().mean(dim=1, keepdim=True).clamp(0, 1)
                person_agnostic = person * (1 - mask)
                out, _, _ = tom(person_agnostic, warped)
            save_tensor_image(out[0], os.path.join(args.save_dir, f"cpvton_{written:05d}.png"))
            written += 1
            if written >= args.num_samples:
                break

    print(f"Inference complete. Saved {written} images to {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPVTON local inference")
    parser.add_argument("--curvton_data_path", type=str, required=True)
    parser.add_argument("--difficulty", type=str, default="all", choices=["easy", "medium", "hard", "all", "easy_hard", "medium_hard"])
    parser.add_argument("--gender", type=str, default="all", choices=["female", "male", "all"])
    parser.add_argument("--output_dir", type=str, default="runs/cross_architecture")
    parser.add_argument("--run_name", type=str, default="train_cpvton_tom")
    parser.add_argument("--stage", type=str, default="TOM", choices=["GMM", "TOM"])
    parser.add_argument("--gmm_checkpoint", type=str, default=None)
    parser.add_argument("--tom_checkpoint", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="runs/cross_architecture/cpvton_infer")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    infer(parser.parse_args())
