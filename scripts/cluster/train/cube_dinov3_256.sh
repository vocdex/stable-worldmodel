#!/bin/bash
#SBATCH -J swm_prejepa_cube_dinov3_256
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=galvani-cn227  # every cube-matrix timeout happened on cn227
#SBATCH --time=0-12:00          # extraction ~1h + cached-feats train ~5-8h on A100 40GB at bs=256
#SBATCH --mem=200G
#SBATCH --output=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_prejepa_cube_dinov3_256_%j.out
#SBATCH --error=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_prejepa_cube_dinov3_256_%j.err

# -----------------------------------------------------------------------------
# DINOv3 256px variant: same 224px cube frames UPSCALED to 256 before
# encoding, so DINOv3-S/16 runs at its native pretraining resolution and
# yields a 16x16 = 256-patch grid — the same token count as DINOv2-S/14 at
# 224. Tests whether the coarser 14x14 grid explains the in-dist gap
# (74 vs ~65 at 224px; epoch sweep ruled out epoch selection).
#
# Steps: (1) extract 256px DINOv3 features from cube_single_expert.h5 if not
# already cached (~80 GB), (2) train the predictor on the cached features.
# Output ckpt: cube_dinov3_small_256_actiononly_cachedfeats_psmall.
#
# Branch: dinov3-cube-ablation
#   cd /mnt/lustre/work/martius/mot956/stable-worldmodel
#   git fetch && git checkout dinov3-cube-ablation && git pull
#
# GATED WEIGHTS: facebook/dinov3-vits16-pretrain-lvd1689m requires an accepted
# license on HuggingFace. The job downloads the weights automatically (compute
# nodes have outbound internet) once a token from an account with the accepted
# license is available. One-time setup on the cluster (token string = contents
# of ~/.cache/huggingface/token on your local machine):
#   mkdir -p /mnt/lustre/work/martius/mot956/hf
#   echo 'hf_...your-token...' > /mnt/lustre/work/martius/mot956/hf/token
#   chmod 600 /mnt/lustre/work/martius/mot956/hf/token
# (huggingface_hub reads $HF_HOME/token; exporting HF_TOKEN also works.)
# The job fails fast if neither a cached snapshot nor a token is present.
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

# --- Pre-flight: gated DINOv3 weights need a COMPLETE cached snapshot or HF auth ---
# A bare directory is not enough (a partial rsync/download leaves one behind);
# require a resolvable config.json inside snapshots/. -L follows the blob
# symlinks so broken links from an aborted transfer don't count.
DINOV3_SNAPSHOT="$HF_HOME/hub/models--facebook--dinov3-vits16-pretrain-lvd1689m"
CACHED_CFG=$(find -L "$DINOV3_SNAPSHOT/snapshots" -maxdepth 2 -name config.json 2>/dev/null | head -1)
if [ -z "$CACHED_CFG" ] && [ -z "${HF_TOKEN:-}" ] && [ ! -f "$HF_HOME/token" ]; then
    echo "FATAL: no complete DINOv3 snapshot in $HF_HOME and no HF token found."
    echo "The weights are gated; the job downloads them automatically once a"
    echo "token from an account with the accepted license exists. Run once:"
    echo "  echo 'hf_...your-token...' > $HF_HOME/token && chmod 600 $HF_HOME/token"
    echo "(token string = contents of ~/.cache/huggingface/token on your laptop)"
    if [ -d "$DINOV3_SNAPSHOT" ]; then
        echo "NOTE: $DINOV3_SNAPSHOT exists but is incomplete (aborted transfer?);"
        echo "consider removing it: rm -rf $DINOV3_SNAPSHOT"
    fi
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
DST_H5=$STABLEWM_HOME/datasets/cube_single_expert_dinov3_256_features.h5
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
        --image_size 256 \
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
    --config-name prejepa_cube_dinov3_256_features \
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
