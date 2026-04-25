import argparse
import glob
import os
import random
from typing import Dict, Iterator

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from tqdm import tqdm

from dataloader import build_curvton_difficulty_files, make_loader, subset_files, _list_images
from model import DiT250M, DiTConfig, count_parameters
from utils import curriculum_weights, ensure_dir, make_beta_schedule, q_sample, save_batch_preview


def _maybe_init_ddp() -> tuple[int, int, int, bool, torch.device]:
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = rank == 0
    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return rank, local_rank, world_size, is_main, device


def _next_from(iters: Dict[str, Iterator[torch.Tensor]], loaders: Dict[str, torch.utils.data.DataLoader], key: str) -> torch.Tensor:
    try:
        return next(iters[key])
    except StopIteration:
        iters[key] = iter(loaders[key])
        return next(iters[key])


def train(args: argparse.Namespace) -> None:
    rank, _, world_size, is_main, device = _maybe_init_ddp()
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    cfg = DiTConfig(
        image_size=args.image_size,
        in_channels=3,
        patch_size=args.patch_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
    )
    model = DiT250M(cfg).to(device)
    raw_model = model
    if world_size > 1:
        model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)

    if is_main:
        params_m = count_parameters(raw_model) / 1e6
        print(f"DiT params: {params_m:.2f}M")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    betas = make_beta_schedule(args.diffusion_steps).to(device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    sqrt_ab = torch.sqrt(alpha_bar)
    sqrt_1mab = torch.sqrt(1 - alpha_bar)

    diff_files = build_curvton_difficulty_files(args.data_path, gender=args.gender)
    for k in diff_files:
        diff_files[k] = subset_files(diff_files[k], args.data_fraction, seed=args.seed)

    diff_loaders: Dict[str, torch.utils.data.DataLoader] = {}
    for diff in ("easy", "medium", "hard"):
        loader, _ = make_loader(
            diff_files[diff],
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            world_size=world_size,
            rank=rank,
            shuffle=True,
        )
        diff_loaders[diff] = loader

    all_files = diff_files["easy"] + diff_files["medium"] + diff_files["hard"]
    all_loader, _ = make_loader(
        all_files,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        world_size=world_size,
        rank=rank,
        shuffle=True,
    )

    phase2_loader = None
    phase2_iter = None
    if args.phase2_data_path:
        phase2_files = subset_files(_list_images(args.phase2_data_path), args.data_fraction, seed=args.seed + 7)
        phase2_loader, _ = make_loader(
            phase2_files,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            world_size=world_size,
            rank=rank,
            shuffle=True,
        )

    iters = {k: iter(v) for k, v in diff_loaders.items()}
    all_iter = iter(all_loader)

    run_root = os.path.join(args.output_dir, args.run_name)
    ckpt_dir = os.path.join(run_root, "checkpoints")
    sample_dir = os.path.join(run_root, "samples")
    ensure_dir(ckpt_dir)
    ensure_dir(sample_dir)

    ckpt_to_load = args.resume
    if ckpt_to_load is None and not args.no_resume:
        candidates = glob.glob(os.path.join(ckpt_dir, "ckpt_step_*.pt")) + glob.glob(
            os.path.join(ckpt_dir, "ckpt_final.pt")
        )
        if candidates:
            def _step_num(p):
                base = os.path.basename(p)
                if base == "ckpt_final.pt":
                    return float("inf")
                try:
                    return int(base.split("ckpt_step_")[1].split(".pt")[0])
                except Exception:
                    return -1
            ckpt_to_load = max(candidates, key=_step_num)

    global_step = 0
    if ckpt_to_load:
        ckpt = torch.load(ckpt_to_load, map_location=device)
        raw_model.load_state_dict(ckpt["model"], strict=False)
        global_step = int(ckpt.get("step", 0))

    pbar = tqdm(total=args.max_steps, disable=not is_main, desc="training")
    pbar.update(global_step)
    while global_step < args.max_steps:
        in_phase2 = phase2_loader is not None and global_step >= args.phase2_start_step
        if in_phase2:
            if phase2_iter is None:
                phase2_iter = iter(phase2_loader)
            try:
                x0 = next(phase2_iter)
            except StopIteration:
                phase2_iter = iter(phase2_loader)
                x0 = next(phase2_iter)
        elif args.curriculum == "none":
            try:
                x0 = next(all_iter)
            except StopIteration:
                all_iter = iter(all_loader)
                x0 = next(all_iter)
        else:
            we, wm, wh = curriculum_weights(global_step, args.curriculum, args.stage_steps)
            diff = random.choices(["easy", "medium", "hard"], weights=[we, wm, wh])[0]
            x0 = _next_from(iters, diff_loaders, diff)

        x0 = x0.to(device, non_blocking=True)
        t = torch.randint(0, args.diffusion_steps, (x0.shape[0],), device=device)
        x_t, _ = q_sample(x0, t, sqrt_ab, sqrt_1mab)

        x0_pred = model(x_t, t)
        loss = F.mse_loss(x0_pred, x0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if is_main and global_step % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        if is_main and global_step % args.image_log_interval == 0:
            save_batch_preview(x0_pred[:8].detach().cpu(), os.path.join(sample_dir, f"pred_step_{global_step}.png"))
        if is_main and global_step > 0 and global_step % args.save_interval == 0:
            to_save = raw_model.state_dict() if world_size == 1 else model.module.state_dict()
            torch.save({"step": global_step, "model": to_save, "cfg": cfg.__dict__}, os.path.join(ckpt_dir, f"ckpt_step_{global_step}.pt"))

        global_step += 1
        pbar.update(1)

    if is_main:
        to_save = raw_model.state_dict() if world_size == 1 else model.module.state_dict()
        torch.save({"step": global_step, "model": to_save, "cfg": cfg.__dict__}, os.path.join(ckpt_dir, "ckpt_final.pt"))
    if world_size > 1:
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Custom 250M DiT pretraining (x0-prediction)")
    p.add_argument("--run_name", type=str, default="custom_dit_pretrain")
    p.add_argument("--data_path", type=str, required=True, help="CurvTon root containing easy/medium/hard")
    p.add_argument("--phase2_data_path", type=str, default=None, help="Optional final-stage dataset path")
    p.add_argument("--phase2_start_step", type=int, default=28801)
    p.add_argument("--output_dir", type=str, default="/iopsstor/scratch/cscs/dbartaula/custom_dit_assets")
    p.add_argument("--curriculum", type=str, default="soft", choices=["none", "soft", "reverse", "hard"])
    p.add_argument("--stage_steps", type=int, default=4000)
    p.add_argument("--max_steps", type=int, default=12000)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--image_log_interval", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--patch_size", type=int, default=2)
    p.add_argument("--hidden_size", type=int, default=1536)
    p.add_argument("--depth", type=int, default=9)
    p.add_argument("--num_heads", type=int, default=24)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--diffusion_steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--data_fraction", type=float, default=1.0)
    p.add_argument("--gender", type=str, default="all", choices=["female", "male", "all"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--no_resume", action="store_true", default=False)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())


