#!/bin/bash
#SBATCH --job-name=medium_hard_soft_catvton
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --account=a168
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

WORK_DIR="/capstor/store/cscs/swissai/a168/dbartaula/Stable_Diffusion"
DATA_DIR="/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate"

cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"

export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export FI_CXI_ATS=0
export GLOO_SOCKET_IFNAME=hsn

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

srun torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  --rdzv_id=$SLURM_JOB_ID \
  train.py --dataset curvton --difficulty medium_hard --curriculum soft --stage_steps 6000 --max_steps 12000 --curvton_data_path ${DATA_DIR} --batch_size 16 --num_workers 8 --gender all --save_interval 1000 --image_log_interval 250 --skip_eval --run_name train_medium_hard_soft_curriculum_catvton

