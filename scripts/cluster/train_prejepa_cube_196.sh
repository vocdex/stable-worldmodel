#!/bin/bash
#SBATCH -J swm_prejepa_cube_196
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-12:00          # ~10 min extract + ~7h train at bs=256
#SBATCH --mem=200G
#SBATCH --output=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_prejepa_cube_196_%j.out
#SBATCH --error=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_prejepa_cube_196_%j.err

# -----------------------------------------------------------------------------
# Train PreJEPA on cube at image_size=196 (matching LeWM paper) to test
# whether the 224->196 token-count change closes the 71% -> 86% baseline gap.
#
# Step 1: Extract DINOv2 features at 196x196 (196 tokens) — only if h5 missing.
# Step 2: Train with the new features + image_size=196 override.
#
# Submit:
#   sbatch scripts/cluster/train_prejepa_cube_196.sh
# -----------------------------------------------------------------------------

echo "=================================================="
echo "SLURM Job: $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Node: $SLURM_NODELIST  Partition: $SLURM_JOB_PARTITION"
echo "Start: $(date)"
echo "=================================================="

set +u
source ~/.bashrc
CONDA_ENV_PATH="/mnt/lustre/work/martius/mot956/.conda/swm"
conda activate "$CONDA_ENV_PATH"
set -u

echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Caches
export WANDB_DIR=/mnt/lustre/work/martius/mot956/wandb
export HF_HOME=/mnt/lustre/work/martius/mot956/hf
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=/mnt/lustre/work/martius/mot956/torch_hub
export PYTHONUNBUFFERED=1
mkdir -p "$WANDB_DIR" "$HF_HOME" "$TORCH_HOME"

WORK_BIND_DIR=/mnt/lustre/work/martius/mot956/stable-worldmodel
export STABLEWM_HOME=$WORK_BIND_DIR
if [ ! -e "$STABLEWM_HOME/datasets" ] && [ -d "$STABLEWM_HOME/dataset" ]; then
    ln -s "$STABLEWM_HOME/dataset" "$STABLEWM_HOME/datasets"
fi

cd "$WORK_BIND_DIR"

echo "uv sync (with train + env extras) ..."
uv sync --extra train --extra env

# --- Step 1: extract 196x196 features if missing ---
FEATURES_H5="$STABLEWM_HOME/datasets/cube_single_expert_features_196.h5"
if [ ! -f "$FEATURES_H5" ]; then
    echo "=== Extracting DINOv2 features at 196x196 ==="
    srun --kill-on-bad-exit=1 --unbuffered uv run python scripts/extract/extract_dino_features_cube.py \
        --image_size 196 \
        --dst "$FEATURES_H5"
    EXTRACT_EXIT=$?
    if [ "$EXTRACT_EXIT" -ne 0 ]; then
        echo "FATAL: feature extraction failed (exit $EXTRACT_EXIT)"
        exit "$EXTRACT_EXIT"
    fi
else
    echo "Features already at $FEATURES_H5 — skipping extraction"
fi

# --- Step 2: train ---
echo "=== Training PreJEPA at image_size=196 ==="
srun --kill-on-bad-exit=1 --unbuffered uv run python scripts/train/prejepa.py \
    --config-name prejepa_cube_features \
    dataset_name=cube_single_expert_features_196 \
    image_size=196 \
    output_model_name=cube_dinov2_small_actiononly_cachedfeats_196_psmall \
    cache_dir=$STABLEWM_HOME \
    trainer.max_epochs=20 \
    +trainer.limit_val_batches=0 \
    batch_size=256 \
    num_workers=16 \
    wandb.enable=true \
    wandb.entity=vocdex \
    wandb.project=swm-cube

EXIT_CODE=$?

# Reap GPU stragglers
sleep 2
LEFTOVER=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
           | grep -v '^$' || true)
if [ -n "$LEFTOVER" ]; then
    echo "Killing GPU stragglers: $LEFTOVER"
    echo "$LEFTOVER" | xargs -r kill -KILL 2>/dev/null || true
fi

echo ""
echo "=================================================="
echo "Exit: $EXIT_CODE  End: $(date)"
echo "=================================================="
exit $EXIT_CODE
