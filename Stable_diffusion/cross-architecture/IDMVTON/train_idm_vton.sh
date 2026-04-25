#!/bin/bash
#SBATCH --job-name=idm_vton
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --account=a168
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# IDM-VTON local implementation aligned to the official architecture/loss.
# Repo: https://github.com/yisol/IDM-VTON
# Architecture: SDXL inpainting-style UNet with garment conditioning.
# Loss: diffusion denoising objective on predicted noise (epsilon), with scheduler weighting.

WORK_DIR="${WORK_DIR:-/capstor/store/cscs/swissai/a168/dbartaula/Stable_Diffusion}"
DATA_DIR="${DATA_DIR:-/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate}"
OUT_DIR="${OUT_DIR:-${WORK_DIR}/runs/idm_vton}"

cd "${WORK_DIR}"
export PYTHONPATH="${WORK_DIR}:${WORK_DIR}/cross-architecture:${PYTHONPATH:-}"

export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export FI_CXI_ATS=0
export GLOO_SOCKET_IFNAME=hsn

srun torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)-hsn0:29500" \
  --rdzv_id="$SLURM_JOB_ID" \
  cross-architecture/IDMVTON/train_idm_vton_local.py \
  --curvton_data_path "${DATA_DIR}" \
  --batch_size 8 \
  --num_workers 8 \
  --max_steps 14400 \
  --save_interval 1000 \
  --output_dir "${OUT_DIR}" \
  --run_name train_idm_vton

