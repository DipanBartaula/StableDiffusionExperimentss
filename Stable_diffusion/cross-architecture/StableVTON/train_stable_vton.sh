#!/bin/bash
#SBATCH --job-name=stable_vton
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --account=a168
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# StableVITON local implementation aligned to the official architecture/loss.
# Repo: https://github.com/rlawjdghek/StableVITON
# Architecture: latent diffusion VTON with modified UNet input channels.
# Loss: denoising objective + optional ATV loss in finetune stage.

WORK_DIR="${WORK_DIR:-/capstor/store/cscs/swissai/a168/dbartaula/Stable_Diffusion}"
DATA_DIR="${DATA_DIR:-/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate}"
OUT_DIR="${OUT_DIR:-${WORK_DIR}/runs/stable_vton}"

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
  cross-architecture/StableVTON/train_stable_vton_local.py \
  --curvton_data_path "${DATA_DIR}" \
  --batch_size 32 \
  --num_workers 8 \
  --max_steps 12000 \
  --save_interval 1000 \
  --output_dir "${OUT_DIR}" \
  --run_name train_stable_vton
