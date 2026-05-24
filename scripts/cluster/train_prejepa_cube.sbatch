#!/bin/bash
#SBATCH --job-name=swm_prejepa_cube
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00            # D-HH:MM — 2 days; tune to expected epoch count
#SBATCH --mem=200G
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --output=./logs/swm_prejepa_cube_%j.out
#SBATCH --error=./logs/swm_prejepa_cube_%j.err
set -eo pipefail

# -----------------------------------------------------------------------------
# Train PreJEPA / DINO-WM on OGbench-Cube (action-only, no proprio).
#
# Assumes /mnt/lustre/work/martius/mot956/stable-worldmodel IS the SWM repo
# clone, with the h5 already under dataset/.
#
# One-time setup:
#   cd /mnt/lustre/work/martius/mot956/stable-worldmodel
#   git fetch && git checkout pusht-multi && git pull
#   uv sync
#
# Submit:
#   mkdir -p logs
#   sbatch scripts/cluster/train_prejepa_cube.sbatch
#
# The script auto-creates a `datasets/` symlink to `dataset/` because SWM
# hard-codes the subfolder name as plural (HDF5Dataset uses
# sub_folder='datasets').
# -----------------------------------------------------------------------------

# --- Paths ---
WORK_DIR=/mnt/lustre/work/martius/mot956/stable-worldmodel
export STABLEWM_HOME=$WORK_DIR
if [ ! -e "$STABLEWM_HOME/datasets" ] && [ -d "$STABLEWM_HOME/dataset" ]; then
    ln -s "$STABLEWM_HOME/dataset" "$STABLEWM_HOME/datasets"
fi

# --- HF / WANDB caches (avoid filling $HOME quota) ---
SHARED_CACHE=/mnt/lustre/work/martius/mot956
export HF_HOME=$SHARED_CACHE/hf
export TRANSFORMERS_CACHE=$SHARED_CACHE/hf
export TORCH_HOME=$SHARED_CACHE/torch_hub
export WANDB_DIR=$SHARED_CACHE/wandb
export WANDB_CACHE_DIR=$WANDB_DIR
export WANDB_CONFIG_DIR=$WANDB_DIR
export WANDB_DATA_DIR=$WANDB_DIR
export WANDB_ARTIFACT_DIR=$WANDB_DIR
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME" "$TORCH_HOME" "$WANDB_DIR"

# --- GPU sanity ---
nvidia-smi
echo "node=$(hostname)  cuda=${CUDA_VISIBLE_DEVICES:-unset}  user=$(whoami)  date=$(date)"
echo "STABLEWM_HOME=$STABLEWM_HOME"
test -f "$STABLEWM_HOME/datasets/cube_single_expert.h5" || {
    echo "ERROR: cube_single_expert.h5 not found at $STABLEWM_HOME/datasets/"
    echo "Place the h5 there (or symlink dataset/ -> datasets/) and resubmit."
    exit 1
}

# --- Run ---
cd "$WORK_DIR"
srun uv run python scripts/train/prejepa.py \
    --config-name prejepa_cube \
    cache_dir=$STABLEWM_HOME \
    trainer.max_epochs=100 \
    batch_size=64 \
    num_workers=8 \
    wandb.enable=true \
    wandb.entity=stable-wm \
    wandb.project=swm \
    2>&1
