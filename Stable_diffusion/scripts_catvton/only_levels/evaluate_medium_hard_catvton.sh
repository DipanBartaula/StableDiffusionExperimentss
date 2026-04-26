#!/bin/bash
#SBATCH --job-name=evaluate_medium_hard_catvton
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --account=a168
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

WORK_DIR="/capstor/store/cscs/swissai/a168/dbartaula/Stable_Diffusion"
RUN_NAME="train_medium_hard_soft_curriculum_catvton"
CKPT_DIR="/iopsstor/scratch/cscs/dbartaula/experiments_assets_1/${RUN_NAME}/checkpoints"
CURVTON_TEST_DIR="/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate_test"
TRIPLET_TEST_DIR="/iopsstor/scratch/cscs/dbartaula/human_gen/triplet_dataset_backup_1"
STREET_TRYON_DIR="/iopsstor/scratch/cscs/dbartaula/human_gen/benchmark_datasets/street_tryon"

BATCH_SIZE_PER_GPU=8
NPROC_PER_NODE=4
GLOBAL_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * NPROC_PER_NODE))

cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"

if [ ! -d "$CKPT_DIR" ]; then
  echo "Checkpoint directory not found: $CKPT_DIR"
  exit 1
fi

if [ -f "$CKPT_DIR/ckpt_final.pt" ]; then
  CKPT_PATH="$CKPT_DIR/ckpt_final.pt"
else
  CKPT_PATH=$(ls -1 "$CKPT_DIR"/ckpt_step_*.pt 2>/dev/null | sort -V | tail -n 1)
fi

if [ -z "$CKPT_PATH" ] || [ ! -f "$CKPT_PATH" ]; then
  echo "No checkpoint found in $CKPT_DIR"
  exit 1
fi

echo "Evaluating run: $RUN_NAME"
echo "Using checkpoint: $CKPT_PATH"
echo "Batch size per GPU: $BATCH_SIZE_PER_GPU (target global: $GLOBAL_BATCH_SIZE)"

python evaluate.py \
  --checkpoint "$CKPT_PATH" \
  --curvton_test_data_path "$CURVTON_TEST_DIR" \
  --triplet_test_data_path "$TRIPLET_TEST_DIR" \
  --street_tryon_data_path "$STREET_TRYON_DIR" \
  --street_split validation \
  --batch_size "$BATCH_SIZE_PER_GPU" \
  --num_workers 8 \
  --eval_frac_curvton 0.10 \
  --eval_frac_triplet 0.30 \
  --eval_frac_street 0.30
