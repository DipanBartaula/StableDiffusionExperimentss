#!/bin/bash
#SBATCH --job-name=cdit_meanflow_12000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --account=a168
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

WORK_DIR="/capstor/store/cscs/swissai/a168/dbartaula/Stable_Diffusion"
DATA_DIR="/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate"
PHASE2_DIR="/iopsstor/scratch/cscs/dbartaula/human_gen/triplet_dataset_backup_1_1"

cd "$WORK_DIR"

unset PYTHONHOME || true
unset PYTHONPATH || true

CONDA_ROOT="/iopsstor/scratch/cscs/dbartaula/miniforge3"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-torch26_env_new}"
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
  custom_model_pretraining/train_meanflow.py \
    --data_path ${DATA_DIR} \
    --phase2_data_path ${PHASE2_DIR} \
    --phase2_start_step 10000 \
    --curriculum soft \
    --stage_steps 4000 \
    --max_steps 12000 \
    --batch_size 16 \
    --num_workers 8 \
    --save_interval 1000 \
    --image_log_interval 1000 \
    --gender all \
    --run_name Stable_diffusion_train_custom_dit_meanflow_12000steps \
    --wandb_project Stable_diffusion

