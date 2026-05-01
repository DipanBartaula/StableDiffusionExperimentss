#!/bin/bash
#SBATCH --job-name=medium_only_catvton_fullres
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --account=a168
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

WORK_DIR="/capstor/store/cscs/swissai/a168/dbartaula/Stable_Diffusion"
DATA_DIR="/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate"

cd "$WORK_DIR"
unset PYTHONHOME || true
unset PYTHONPATH || true

CONDA_ROOT="/iopsstor/scratch/cscs/dbartaula/miniforge3"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-Dipan}"
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then source "$CONDA_ROOT/etc/profile.d/conda.sh"; else source "$HOME/miniforge3/etc/profile.d/conda.sh"; fi
conda activate "$CONDA_ENV_NAME"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORK_DIR:${PYTHONPATH:-}"
export WANDB_PROJECT=Stable_diffusion
export WANDB_ENTITY=078bct-anandi-tribhuvan-university-institute-of-engineering

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

srun torchrun --nnodes=1 --nproc_per_node=4 --standalone --master_port=$MASTER_PORT \
  train.py --dataset curvton --difficulty medium --max_steps 28000 --curvton_data_path ${DATA_DIR} --batch_size 8 --num_workers 16 --gender all --image_size 0 --save_interval 1000 --image_log_interval 1000 --use_dream --dream_lambda 10.0 --skip_eval --no_resume --run_name Stable_diffusion_train_medium_only_catvton_fullres

