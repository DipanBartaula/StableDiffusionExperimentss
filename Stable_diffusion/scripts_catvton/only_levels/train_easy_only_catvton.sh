#!/bin/bash
#SBATCH --job-name=easy_only_catvton
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --account=a168
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

WORK_DIR="/capstor/store/cscs/swissai/a168/dbartaula/Stable_Diffusion"
DATA_DIR="/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate"

cd "$WORK_DIR"

# Clean inherited Python env vars first; bad PYTHONHOME/PYTHONPATH can break stdlib encodings.
unset PYTHONHOME || true
unset PYTHONPATH || true

# Activate the intended conda env used on the cluster.
CONDA_ROOT="/iopsstor/scratch/cscs/dbartaula/miniforge3"
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
else
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
fi
conda activate tryon_env

export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORK_DIR:${PYTHONPATH:-}"

export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export FI_CXI_ATS=0
export GLOO_SOCKET_IFNAME=hsn

# For single-node training, use localhost to avoid hostname resolution issues
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export TORCHELASTIC_ERROR_FILE=/tmp/torch_elastic_error_${SLURM_JOB_ID}.json
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export DISABLE_WANDB=1
export WANDB_MODE=disabled
export WANDB_SILENT=true

trap 'echo "[DEBUG] TORCHELASTIC_ERROR_FILE=$TORCHELASTIC_ERROR_FILE"; if [ -f "$TORCHELASTIC_ERROR_FILE" ]; then echo "[DEBUG] Dumping torchelastic error JSON"; cat "$TORCHELASTIC_ERROR_FILE"; fi' EXIT

echo "[DEBUG] Host=$(hostname) JobID=${SLURM_JOB_ID:-unknown}"
echo "[DEBUG] Python=$(which python || true) Torchrun=$(which torchrun || true)"
python -V || true
python -c "import sys, encodings; print('[DEBUG] exe', sys.executable); print('[DEBUG] prefix', sys.prefix); print('[DEBUG] encodings', encodings.__file__)" || true
python -c "import torch; print('[DEBUG] torch', torch.__version__)" || true
nvidia-smi -L || true

srun torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --standalone \
  --master_port=$MASTER_PORT \
  train.py --dataset curvton --difficulty easy --max_steps 12000 --curvton_data_path ${DATA_DIR} --batch_size 16 --num_workers 8 --gender all --save_interval 1000 --image_log_interval 250 --skip_eval --run_name train_easy_only_catvton
