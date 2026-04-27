#!/bin/bash
#SBATCH --job-name=infer_gan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --account=a168
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

WORK_DIR="/capstor/store/cscs/swissai/a168/dbartaula/Stable_Diffusion"

cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"

# ── Edit these paths ──────────────────────────────────────────
CHECKPOINT="/path/to/ckpt_final.pt"
PERSON_IMG="/path/to/person.jpg"
CLOTH_IMG="/path/to/cloth.jpg"
OUTPUT="results/tryon_gan.png"
# ──────────────────────────────────────────────────────────────

python inference_gan.py \
  --checkpoint "$CHECKPOINT" \
  --person "$PERSON_IMG" \
  --cloth "$CLOTH_IMG" \
  --output "$OUTPUT" \
  --fp16

