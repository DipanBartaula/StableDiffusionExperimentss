#!/bin/bash
#SBATCH --job-name=hard_only_catvton_fullres
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --account=a168
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

WORK_DIR="/iopsstor/scratch/cscs/dbartaula/StableDiffusionExperimentss"
DATA_DIR="/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate"

cd "$WORK_DIR"
unset PYTHONHOME || true
unset PYTHONPATH || true

CONDA_ROOT="/iopsstor/scratch/cscs/dbartaula/miniforge3"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-torch27_env_new}"
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then source "$CONDA_ROOT/etc/profile.d/conda.sh"; else source "$HOME/miniforge3/etc/profile.d/conda.sh"; fi
conda activate "$CONDA_ENV_NAME"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORK_DIR:${PYTHONPATH:-}"
export WANDB_PROJECT=Stable_diffusion
export WANDB_ENTITY=078bct-anandi-tribhuvan-university-institute-of-engineering

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

srun torchrun --nnodes=1 --nproc_per_node=4 --standalone --master_port=$MASTER_PORT \
  train.py --dataset curvton --difficulty hard --max_steps 28000 --curvton_data_path ${DATA_DIR} --batch_size 6 --num_workers 16 --gender all --image_size 0 --save_interval 1000 --image_log_interval 500 --use_dream --dream_lambda 10.0 --skip_eval --no_resume --run_name Stable_diffusion_train_hard_only_catvton_fullres

