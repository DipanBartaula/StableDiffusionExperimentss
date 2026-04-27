import argparse
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image


THIS_DIR = os.path.dirname(__file__)
CROSS_ARCH_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if CROSS_ARCH_DIR not in sys.path:
    sys.path.insert(0, CROSS_ARCH_DIR)

from common import DistInfo, batch_images, build_curvton_loader, latest_checkpoint  # noqa: E402
from train_stable_vton_local import StableVTONModel, stableviton_preprocess  # noqa: E402


def save_tensor_image(tensor, path):
    x = ((tensor.detach().cpu().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
    x = x.permute(1, 2, 0).numpy()
    Image.fromarray(x).save(path)


def infer(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = StableVTONModel(args.model_name).to(device)
    model.eval()

    run_dir = os.path.join(args.output_dir, args.run_name)
    ckpt_path = args.checkpoint or latest_checkpoint(run_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.unet.load_state_dict(ckpt["model_state_dict"], strict=False)

    os.makedirs(args.save_dir, exist_ok=True)
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
            prep = stableviton_preprocess(person, cloth, gt)
            agnostic_lat = model.encode(prep["agnostic"])
            pose_lat = model.encode(prep["pose_img"])
            latents = torch.randn_like(agnostic_lat)
            mask_lat = F.interpolate(prep["agnostic_mask"], size=latents.shape[-2:], mode="bilinear", align_corners=False)

            model.scheduler.set_timesteps(args.num_inference_steps, device=device)
            for t in model.scheduler.timesteps:
                t_batch = torch.full((latents.shape[0],), int(t), device=device, dtype=torch.long)
                noise_pred = model(latents, mask_lat, agnostic_lat, pose_lat, t_batch)
                latents = model.scheduler.step(noise_pred, t, latents).prev_sample

            out = model.vae.decode(latents / model.vae.config.scaling_factor).sample
            save_tensor_image(out[0], os.path.join(args.save_dir, f"stable_{written:05d}.png"))
            written += 1
            if written >= args.num_samples:
                break

    print(f"Inference complete. Saved {written} images to {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StableVTON local inference")
    parser.add_argument("--curvton_data_path", type=str, required=True)
    parser.add_argument("--difficulty", type=str, default="all", choices=["easy", "medium", "hard", "all", "easy_hard", "medium_hard"])
    parser.add_argument("--gender", type=str, default="all", choices=["female", "male", "all"])
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--output_dir", type=str, default="runs/cross_architecture")
    parser.add_argument("--run_name", type=str, default="train_stable_vton")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="runs/cross_architecture/stable_infer")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    infer(parser.parse_args())
