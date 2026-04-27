#!/bin/bash
#SBATCH --job-name=eval_cdit_dp250_12000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --account=a168
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

WORK_DIR="/capstor/store/cscs/swissai/a168/dbartaula/Stable_Diffusion"
RUN_NAME="Stable_diffusion_train_custom_dit_datapred_250m_12000steps"
CKPT_DIR="/iopsstor/scratch/cscs/dbartaula/experiments_assets/${RUN_NAME}/checkpoints"
CURVTON_TEST_DIR="/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate_test"
TRIPLET_TEST_DIR="/iopsstor/scratch/cscs/dbartaula/human_gen/triplet_dataset_backup_1"
STREET_TRYON_DIR="/iopsstor/scratch/cscs/dbartaula/human_gen/benchmark_datasets/street_tryon"

BATCH_SIZE_PER_GPU=16
NPROC_PER_NODE=4
GLOBAL_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * NPROC_PER_NODE))

cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"

if [ -f "$CKPT_DIR/ckpt_final.pt" ]; then
  CKPT_PATH="$CKPT_DIR/ckpt_final.pt"
else
  CKPT_PATH=$(ls -1 "$CKPT_DIR"/ckpt_step_*.pt 2>/dev/null | sort -V | tail -n 1)
fi

echo "Using checkpoint: $CKPT_PATH"
echo "Batch size per GPU: $BATCH_SIZE_PER_GPU (target global: $GLOBAL_BATCH_SIZE)"
echo "Metric datasets:"
echo "  CURVTON_TEST_DIR=$CURVTON_TEST_DIR"
echo "  TRIPLET_TEST_DIR=$TRIPLET_TEST_DIR"
echo "  STREET_TRYON_DIR=$STREET_TRYON_DIR"

python custom_model_pretraining/evaluate_fid_kid.py \
  --approach datapred \
  --model_size 250m \
  --checkpoint "$CKPT_PATH" \
  --curvton_test_data_path "$CURVTON_TEST_DIR" \
  --triplet_test_data_path "$TRIPLET_TEST_DIR" \
  --street_tryon_data_path "$STREET_TRYON_DIR" \
  --street_split validation \
  --image_size 64 \
  --batch_size "$BATCH_SIZE_PER_GPU" \
  --num_workers 8 \
  --diffusion_steps 50 \
  --eval_frac_curvton 0.10 \
  --eval_frac_triplet 0.30 \
  --eval_frac_street 0.30 \
  --output_json "/iopsstor/scratch/cscs/dbartaula/experiments_assets/${RUN_NAME}/eval_fid_kid.json"

