"""Local CP-VTON-style trainer.

Architecture alignment:
- GMM stage predicts a geometric transform and warps the cloth.
- TOM stage synthesizes rendered person + composition mask, then blends with
  warped cloth: output = mask * warped_cloth + (1 - mask) * rendered.
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam

from common import (
    UNetGenerator,
    add_common_args,
    batch_images,
    build_curvton_loader,
    cleanup_dist,
    latest_stage_checkpoint,
    save_checkpoint,
    setup_dist,
    tv_loss,
    wrap_ddp,
)


class TPSGridGen(nn.Module):
    def __init__(self, grid_size=5):
        super().__init__()
        axis = torch.linspace(-1.0, 1.0, grid_size)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        ctrl = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
        self.register_buffer("ctrl", ctrl)

    @staticmethod
    def _u(r2):
        return r2 * torch.log(r2 + 1e-6)

    def forward(self, target_ctrl, out_hw):
        bsz, k, _ = target_ctrl.shape
        h, w = out_hw
        source_ctrl = self.ctrl.unsqueeze(0).expand(bsz, -1, -1)

        pairwise = torch.cdist(target_ctrl, target_ctrl).pow(2)
        kernel = self._u(pairwise)
        ones = torch.ones(bsz, k, 1, device=target_ctrl.device, dtype=target_ctrl.dtype)
        zeros_33 = torch.zeros(bsz, 3, 3, device=target_ctrl.device, dtype=target_ctrl.dtype)
        top = torch.cat([kernel, target_ctrl, ones], dim=2)
        bottom_left = torch.cat([target_ctrl.transpose(1, 2), ones.transpose(1, 2)], dim=1)
        system = torch.cat([top, torch.cat([bottom_left, zeros_33], dim=2)], dim=1)
        rhs = torch.cat([source_ctrl, torch.zeros(bsz, 3, 2, device=target_ctrl.device, dtype=target_ctrl.dtype)], dim=1)
        params = torch.linalg.solve(system, rhs)

        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=target_ctrl.device, dtype=target_ctrl.dtype),
            torch.linspace(-1.0, 1.0, w, device=target_ctrl.device, dtype=target_ctrl.dtype),
            indexing="ij",
        )
        points = torch.stack([x.reshape(-1), y.reshape(-1)], dim=1)
        points_b = points.unsqueeze(0).expand(bsz, -1, -1)
        dist = torch.cdist(points_b, target_ctrl).pow(2)
        basis = torch.cat(
            [
                self._u(dist),
                points_b,
                torch.ones(bsz, points.shape[0], 1, device=target_ctrl.device, dtype=target_ctrl.dtype),
            ],
            dim=2,
        )
        grid = torch.bmm(basis, params)
        return grid.view(bsz, h, w, 2).clamp(-1, 1)


class GMM(nn.Module):
    """CP-VTON geometric matching module with local TPS grid generation."""

    def __init__(self, in_channels=6, grid_size=5):
        super().__init__()
        self.grid_gen = TPSGridGen(grid_size=grid_size)
        num_ctrl = grid_size * grid_size
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.regressor = nn.Linear(256, num_ctrl * 2)
        nn.init.zeros_(self.regressor.weight)
        nn.init.zeros_(self.regressor.bias)

    def forward(self, person, cloth):
        delta = torch.tanh(self.regressor(self.features(torch.cat([person, cloth], dim=1)))).view(cloth.shape[0], -1, 2)
        target_ctrl = self.grid_gen.ctrl.unsqueeze(0).to(delta.dtype) + 0.1 * delta
        grid = self.grid_gen(target_ctrl, cloth.shape[-2:])
        warped = F.grid_sample(cloth, grid, padding_mode="border", align_corners=False)
        return warped, grid, target_ctrl


class TOM(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNetGenerator(in_channels=6, out_channels=4, base=64)

    def forward(self, person_agnostic, warped_cloth):
        out = self.unet(torch.cat([person_agnostic, warped_cloth], dim=1))
        rendered = torch.tanh(out[:, :3])
        mask = torch.sigmoid(out[:, 3:4])
        final = mask * warped_cloth + (1 - mask) * rendered
        return final, rendered, mask


def train_gmm(args, dist_info):
    model = wrap_ddp(GMM(), dist_info)
    loader, sampler = build_curvton_loader(args, dist_info)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=(dist_info.device.type == "cuda"))
    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_to_load = args.resume
    if ckpt_to_load is None and not args.no_resume:
        ckpt_to_load = latest_stage_checkpoint(run_dir, "gmm")
    step = 0
    if ckpt_to_load:
        ckpt = torch.load(ckpt_to_load, map_location=dist_info.device)
        target_model = model.module if hasattr(model, "module") else model
        target_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step = int(ckpt.get("step", 0))

    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for batch in loader:
            person, cloth, gt = batch_images(batch, dist_info.device)
            with autocast(enabled=(dist_info.device.type == "cuda")):
                warped, _, theta = model(person, cloth)
                loss_warp = F.l1_loss(warped, gt)
                identity = model.module.grid_gen.ctrl if hasattr(model, "module") else model.grid_gen.ctrl
                identity = identity.to(theta.device, theta.dtype).unsqueeze(0).expand_as(theta)
                loss_reg = F.mse_loss(theta, identity)
                loss = loss_warp + args.lambda_reg * loss_reg

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            step += 1
            if dist_info.is_main and step % args.save_interval == 0:
                save_checkpoint(os.path.join(args.output_dir, args.run_name, f"gmm_{step}.pt"), model, optimizer, step)
            if step >= args.max_steps:
                break
    if dist_info.is_main:
        save_checkpoint(os.path.join(args.output_dir, args.run_name, "gmm_final.pt"), model, optimizer, step)


def train_tom(args, dist_info):
    gmm = GMM().to(dist_info.device)
    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    gmm_ckpt = args.gmm_checkpoint or latest_stage_checkpoint(run_dir, "gmm")
    if gmm_ckpt:
        state = torch.load(gmm_ckpt, map_location=dist_info.device)
        gmm.load_state_dict(state["model_state_dict"], strict=False)
    gmm.eval()

    model = wrap_ddp(TOM(), dist_info)
    loader, sampler = build_curvton_loader(args, dist_info)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=(dist_info.device.type == "cuda"))
    ckpt_to_load = args.resume
    if ckpt_to_load is None and not args.no_resume:
        ckpt_to_load = latest_stage_checkpoint(run_dir, "tom")
    step = 0
    if ckpt_to_load:
        ckpt = torch.load(ckpt_to_load, map_location=dist_info.device)
        target_model = model.module if hasattr(model, "module") else model
        target_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step = int(ckpt.get("step", 0))

    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for batch in loader:
            person, cloth, gt = batch_images(batch, dist_info.device)
            mask = (person - gt).abs().mean(dim=1, keepdim=True).clamp(0, 1)
            person_agnostic = person * (1 - mask)
            with torch.no_grad():
                warped, _, _ = gmm(person, cloth)
            with autocast(enabled=(dist_info.device.type == "cuda")):
                final, rendered, comp_mask = model(person_agnostic, warped)
                loss = (
                    args.lambda_l1 * F.l1_loss(final, gt)
                    + args.lambda_render * F.l1_loss(rendered, gt)
                    + args.lambda_mask_tv * tv_loss(comp_mask)
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            step += 1
            if dist_info.is_main and step % args.save_interval == 0:
                save_checkpoint(os.path.join(args.output_dir, args.run_name, f"tom_{step}.pt"), model, optimizer, step)
            if step >= args.max_steps:
                break
    if dist_info.is_main:
        save_checkpoint(os.path.join(args.output_dir, args.run_name, "tom_final.pt"), model, optimizer, step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local CP-VTON aligned GMM/TOM training")
    add_common_args(parser)
    parser.add_argument("--stage", type=str, default="GMM", choices=["GMM", "TOM"])
    parser.add_argument("--gmm_checkpoint", type=str, default=None)
    parser.add_argument("--lambda_reg", type=float, default=0.01)
    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_render", type=float, default=1.0)
    parser.add_argument("--lambda_mask_tv", type=float, default=0.01)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_resume", action="store_true", default=False)
    args = parser.parse_args()
    args.run_name = args.run_name or f"train_cpvton_{args.stage.lower()}"

    dist_info = setup_dist()
    if args.stage == "GMM":
        train_gmm(args, dist_info)
    else:
        train_tom(args, dist_info)
    cleanup_dist()
