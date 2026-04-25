#!/bin/bash
#SBATCH --job-name=ootd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --account=a168
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# OOTDiffusion local implementation aligned to the paper-level architecture/loss.
# Official repo: https://github.com/levihsu/OOTDiffusion
# Architecture: latent diffusion with outfitting-fusion person/garment latents.
# Loss: diffusion denoising objective on predicted noise.

WORK_DIR="${WORK_DIR:-/capstor/store/cscs/swissai/a168/dbartaula/Stable_Diffusion}"
DATA_DIR="${DATA_DIR:-/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate}"
OUT_DIR="${OUT_DIR:-${WORK_DIR}/runs/ootdiffusion}"

cd "${WORK_DIR}"
export PYTHONPATH="${WORK_DIR}:${WORK_DIR}/cross-architecture:${PYTHONPATH:-}"

export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export FI_CXI_ATS=0
export GLOO_SOCKET_IFNAME=hsn

export MASTER_ADDR
MASTER_ADDR="127.0.0.1"
export MASTER_PORT=29500
export TORCHELASTIC_ERROR_FILE=/tmp/torch_elastic_error_${SLURM_JOB_ID}.json
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

srun torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  --rdzv_id=$SLURM_JOB_ID \
  cross-architecture/OOTDiffusion/train_ootdiffusion_local.py \
  --curvton_data_path "${DATA_DIR}" \
  --batch_size 8 \
  --num_workers 8 \
  --max_steps 14400 \
  --save_interval 1000 \
  --output_dir "${OUT_DIR}" \
  --run_name train_ootdiffusion

