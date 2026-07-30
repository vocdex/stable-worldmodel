#!/bin/bash
#SBATCH -J swm_prejepa_cube_dinov3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-12:00          # extraction ~1h + cached-feats train ~5-8h on A100 40GB at bs=256
#SBATCH --mem=200G
#SBATCH --output=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_prejepa_cube_dinov3_%j.out
#SBATCH --error=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_prejepa_cube_dinov3_%j.err

# -----------------------------------------------------------------------------
# DINOv3 backbone ablation: train PreJEPA / DINO-WM on OGbench-Cube (single,
# action-only) with a frozen DINOv3-S/16 encoder instead of DINOv2-S/14.
# Mirrors the cube_dinov2_small_actiononly_cachedfeats_psmall run exactly
# (same data, hyperparameters, 20 epochs) — only the backbone differs.
#
# Steps: (1) extract DINOv3 features from cube_single_expert.h5 if not already
# cached, (2) train the predictor on the cached features.
#
# Branch: dinov3-cube-ablation
#   cd /mnt/lustre/work/martius/mot956/stable-worldmodel
#   git fetch && git checkout dinov3-cube-ablation && git pull
#
# GATED WEIGHTS: facebook/dinov3-vits16-pretrain-lvd1689m requires an accepted
# license on HuggingFace. Either seed the cache once from a machine that has
# the snapshot (run locally):
#   rsync -av ~/.cache/huggingface/hub/models--facebook--dinov3-vits16-pretrain-lvd1689m \
#       mot956@134.2.168.114:/mnt/lustre/work/martius/mot956/hf/hub/
# or export HF_TOKEN before sbatch. The job fails fast if neither is present.
#
# Submit:
#   sbatch scripts/cluster/train/cube_dinov3.sh
# -----------------------------------------------------------------------------

echo "=================================================="
echo "SLURM Job: $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Node: $SLURM_NODELIST  Partition: $SLURM_JOB_PARTITION"
echo "Start: $(date)"
echo "=================================================="

source ~/.bashrc

CONDA_ENV_PATH="/mnt/lustre/work/martius/mot956/.conda/swm"
conda activate "$CONDA_ENV_PATH"

echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "conda env: ${CONDA_DEFAULT_ENV:-?}  python=$(which python)"
echo "uv: $(uv --version 2>/dev/null || echo not found)"
echo ""

# Caches (avoid $HOME quota)
export WANDB_DIR=/mnt/lustre/work/martius/mot956/wandb
export WANDB_ARTIFACT_DIR=$WANDB_DIR
export WANDB_CACHE_DIR=$WANDB_DIR
export WANDB_CONFIG_DIR=$WANDB_DIR
export WANDB_DATA_DIR=$WANDB_DIR
export HF_HOME=/mnt/lustre/work/martius/mot956/hf
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=/mnt/lustre/work/martius/mot956/torch_hub
export PYTHONUNBUFFERED=1
mkdir -p "$WANDB_DIR" "$HF_HOME" "$TORCH_HOME"

# --- Pre-flight: gated DINOv3 weights must be reachable ---
DINOV3_SNAPSHOT="$HF_HOME/hub/models--facebook--dinov3-vits16-pretrain-lvd1689m"
if [ ! -d "$DINOV3_SNAPSHOT" ] && [ -z "${HF_TOKEN:-}" ]; then
    echo "FATAL: DINOv3 weights not in $HF_HOME and HF_TOKEN unset."
    echo "Seed the cache from a machine with the snapshot:"
    echo "  rsync -av ~/.cache/huggingface/hub/models--facebook--dinov3-vits16-pretrain-lvd1689m \\"
    echo "      mot956@134.2.168.114:$HF_HOME/hub/"
    exit 3
fi

# Dataset cache root (SWM hard-codes the subfolder name `datasets` plural;
# symlink dataset/ -> datasets/ if needed)
WORK_BIND_DIR=/mnt/lustre/work/martius/mot956/stable-worldmodel
export STABLEWM_HOME=$WORK_BIND_DIR
if [ ! -e "$STABLEWM_HOME/datasets" ] && [ -d "$STABLEWM_HOME/dataset" ]; then
    ln -s "$STABLEWM_HOME/dataset" "$STABLEWM_HOME/datasets"
fi

cd "$WORK_BIND_DIR"

# hydra-core / transformers / wandb / stable-pretraining are in the `train`
# optional-dep group; env deps in `env`. Plain `uv sync` misses them.
echo "uv sync (with train + env extras) ..."
uv sync --extra train --extra env

echo "torch check: $(uv run python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"
echo ""

# --- Step 1: extract DINOv3 features (idempotent; script refuses overwrite) ---
SRC_H5=$STABLEWM_HOME/datasets/cube_single_expert.h5
DST_H5=$STABLEWM_HOME/datasets/cube_single_expert_dinov3_features.h5
if [ -f "$DST_H5" ]; then
    echo "Features already extracted: $DST_H5"
else
    if [ ! -f "$SRC_H5" ]; then
        echo "FATAL: source dataset $SRC_H5 not found"
        exit 4
    fi
    echo "Extracting DINOv3 features -> $DST_H5"
    srun --kill-on-bad-exit=1 --unbuffered uv run python \
        scripts/extract/extract_dino_features_cube.py \
        --src "$SRC_H5" \
        --dst "$DST_H5" \
        --backbone facebook/dinov3-vits16-pretrain-lvd1689m \
        --image_size 224 \
        --frameskip 5 \
        --batch_size 128 \
        --num_workers 4
    EXTRACT_CODE=$?
    if [ $EXTRACT_CODE -ne 0 ]; then
        echo "FATAL: feature extraction failed (exit $EXTRACT_CODE)"
        # Remove partial output so a re-run doesn't trip the overwrite guard
        rm -f "$DST_H5"
        exit $EXTRACT_CODE
    fi
fi
echo ""

# --- Step 2: train ---
srun --kill-on-bad-exit=1 --unbuffered uv run python scripts/train/prejepa.py \
    --config-name prejepa_cube_dinov3_features \
    cache_dir=$STABLEWM_HOME \
    trainer.max_epochs=20 \
    +trainer.limit_val_batches=0 \
    batch_size=256 \
    num_workers=16 \
    wandb.enable=true \
    wandb.entity=vocdex \
    wandb.project=swm-cube

EXIT_CODE=$?

# Reap GPU stragglers (DataLoader workers + wandb threads sometimes hold the GPU)
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
