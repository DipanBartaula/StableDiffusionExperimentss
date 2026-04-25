#!/bin/bash
#SBATCH --job-name=cpvton
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --account=a168
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# CP-VTON local implementation aligned to the official two-stage pipeline.
# Reference repo used in this workspace: https://github.com/sergeywong/cp-vton
# Stage 1 (GMM): geometric matching with L1 warp supervision.
# Stage 2 (TOM): try-on module with reconstruction losses on synthesized output.

WORK_DIR="${WORK_DIR:-/capstor/store/cscs/swissai/a168/dbartaula/Stable_Diffusion}"
DATA_DIR="${DATA_DIR:-/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3/dataset_ultimate}"
OUT_DIR="${OUT_DIR:-${WORK_DIR}/runs/cpvton}"
CPVTON_STAGE="${CPVTON_STAGE:-GMM}"   # GMM or TOM
CPVTON_RUN_NAME="${CPVTON_RUN_NAME:-cpvton_${CPVTON_STAGE,,}_bs32}"

if [[ "${CPVTON_STAGE}" != "GMM" && "${CPVTON_STAGE}" != "TOM" ]]; then
  echo "ERROR: CPVTON_STAGE must be GMM or TOM (got: ${CPVTON_STAGE})"
  exit 1
fi

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
  cross-architecture/CPVTON/train_cpvton_local.py \
  --stage "${CPVTON_STAGE}" \
  --curvton_data_path "${DATA_DIR}" \
  --batch_size 32 \
  --num_workers 8 \
  --max_steps 12000 \
  --save_interval 1000 \
  --output_dir "${OUT_DIR}" \
  --run_name "${CPVTON_RUN_NAME}"
