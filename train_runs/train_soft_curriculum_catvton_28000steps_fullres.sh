#!/bin/bash
#SBATCH --job-name=soft_cat_28000_fullres
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
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
else
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV_NAME"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORK_DIR:${PYTHONPATH:-}"
export WANDB_PROJECT=Stable_diffusion
export WANDB_ENTITY=078bct-anandi-tribhuvan-university-institute-of-engineering

export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export FI_CXI_ATS=0
export GLOO_SOCKET_IFNAME=hsn

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

srun torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --standalone \
  --master_port=$MASTER_PORT \
  train.py --dataset curvton --curriculum soft --stage_steps 9600 --max_steps 28000 --curvton_data_path ${DATA_DIR} --batch_size 6 --num_workers 16 --gender all --image_size 0 --save_interval 1000 --image_log_interval 500 --use_dream --dream_lambda 10.0 --skip_eval --no_resume --run_name Stable_diffusion_train_soft_curriculum_catvton_28000steps_fullres


