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
DATA_DIR="${DATA_DIR:-/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate}"
OUT_DIR="${OUT_DIR:-${WORK_DIR}/runs/cpvton}"
CPVTON_STAGE="${CPVTON_STAGE:-GMM}"   # GMM or TOM
CPVTON_RUN_NAME="${CPVTON_RUN_NAME:-cpvton_${CPVTON_STAGE,,}_bs32}"

if [[ "${CPVTON_STAGE}" != "GMM" && "${CPVTON_STAGE}" != "TOM" ]]; then
  echo "ERROR: CPVTON_STAGE must be GMM or TOM (got: ${CPVTON_STAGE})"
  exit 1
fi

cd "${WORK_DIR}"

# Clean inherited Python env vars first; bad PYTHONHOME/PYTHONPATH can break stdlib encodings.
unset PYTHONHOME || true
unset PYTHONPATH || true

# Activate the intended conda env used on the cluster.
CONDA_ROOT="/iopsstor/scratch/cscs/dbartaula/miniforge3"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-torch26_env_new}"
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
else
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV_NAME"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${WORK_DIR}:${WORK_DIR}/cross-architecture:${PYTHONPATH:-}"

export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export FI_CXI_ATS=0
export GLOO_SOCKET_IFNAME=hsn

# Select a concrete network interface name for Gloo/NCCL ("hsn" alone is not a valid device).
for _if in hsn0 ib0 eth0 enp0s3 lo; do
  if ip -o link show "$_if" >/dev/null 2>&1; then
    export NCCL_SOCKET_IFNAME="$_if"
    export GLOO_SOCKET_IFNAME="$_if"
    break
  fi
done

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export TORCHELASTIC_ERROR_FILE=/tmp/torch_elastic_error_${SLURM_JOB_ID}.json
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "[DEBUG] Host=$(hostname) JobID=${SLURM_JOB_ID:-unknown}"
echo "[DEBUG] Conda env=$CONDA_ENV_NAME CONDA_PREFIX=${CONDA_PREFIX:-unset}"
echo "[DEBUG] NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"
echo "[DEBUG] Python=$(which python || true) Torchrun=$(which torchrun || true)"
python -V || true

srun torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --standalone \
  --master_port=$MASTER_PORT \
  cross-architecture/CPVTON/train_cpvton_local.py \
  --stage "${CPVTON_STAGE}" \
  --curvton_data_path "${DATA_DIR}" \
  --batch_size 32 \
  --num_workers 8 \
  --max_steps 12000 \
  --save_interval 1000 \
  --output_dir "${OUT_DIR}" \
  --run_name "${CPVTON_RUN_NAME}"
