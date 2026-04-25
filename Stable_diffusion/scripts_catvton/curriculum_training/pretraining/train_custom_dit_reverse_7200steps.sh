#!/bin/bash
#SBATCH --job-name=cdit_rev_7200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --account=a168
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

WORK_DIR="/capstor/store/cscs/swissai/a168/dbartaula/Stable_Diffusion"
DATA_DIR="/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate"
PHASE2_DIR="/iopsstor/scratch/cscs/dbartaula/human_gen/triplet_dataset_backup_1_1"

cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"

export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export FI_CXI_ATS=0
export GLOO_SOCKET_IFNAME=hsn

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)-hsn0
export MASTER_PORT=29500

srun torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  --rdzv_id=$SLURM_JOB_ID \
  custom_model_pretraining/train.py --data_path ${DATA_DIR} --curriculum reverse --stage_steps 2400 --max_steps 7200 --batch_size 16 --num_workers 8 --save_interval 1000 --image_log_interval 250 --run_name train_custom_dit_reverse_7200steps

