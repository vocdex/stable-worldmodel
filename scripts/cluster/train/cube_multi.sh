#!/bin/bash
#SBATCH -J swm_prejepa_cube
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-12:00
#SBATCH --mem=200G
#SBATCH --output=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_prejepa_cube_%x_%j.out
#SBATCH --error=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_prejepa_cube_%x_%j.err

# Extract DINO features then train PreJEPA for a given cube env_type.
#
# Submit:
#   sbatch --job-name=cube_single    --export=ENV_TYPE=single    scripts/cluster/train/cube_multi.sh
#   sbatch --job-name=cube_double    --export=ENV_TYPE=double    scripts/cluster/train/cube_multi.sh
#   sbatch --job-name=cube_triple    --export=ENV_TYPE=triple    scripts/cluster/train/cube_multi.sh
#   sbatch --job-name=cube_quadruple --export=ENV_TYPE=quadruple scripts/cluster/train/cube_multi.sh

ENV_TYPE=${ENV_TYPE:-single}

echo "=================================================="
echo "SLURM Job: $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Node: $SLURM_NODELIST  Partition: $SLURM_JOB_PARTITION"
echo "ENV_TYPE: $ENV_TYPE"
echo "Start: $(date)"
echo "=================================================="

source ~/.bashrc

CONDA_ENV_PATH="/mnt/lustre/work/martius/mot956/.conda/swm"
conda activate "$CONDA_ENV_PATH"

echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

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

WORK_DIR=/mnt/lustre/work/martius/mot956/stable-worldmodel
cd "$WORK_DIR"

uv sync --extra train --extra env

echo "torch check: $(uv run python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"
echo ""

# Raw datasets: $WORK_DIR/datasets/ogbench/cube_*_expert.h5
# Features out: $WORK_DIR/datasets/cube_*_expert_features.h5  (same level as cube_single)
SRC="$WORK_DIR/datasets/cube_${ENV_TYPE}_expert.h5"
DST="$WORK_DIR/datasets/cube_${ENV_TYPE}_expert_features.h5"

if [ ! -f "$SRC" ]; then
    echo "ERROR: source dataset not found: $SRC"
    exit 1
fi

# --- Feature extraction ---
if [ -f "$DST" ]; then
    echo "Features already exist: $DST — skipping extraction."
else
    echo "Extracting DINO features: $SRC -> $DST"
    uv run python scripts/extract/extract_dino_features_cube.py \
        --src "$SRC" \
        --dst "$DST" \
        --batch_size 256 \
        --num_workers 8 || exit 1
    echo "Extraction done: $(date)"
fi

# --- PreJEPA training ---
echo ""
echo "Training PreJEPA on cube_${ENV_TYPE}_expert_features ..."
uv run python scripts/train/prejepa.py \
    --config-name prejepa_cube_features \
    dataset_name=cube_${ENV_TYPE}_expert_features \
    cache_dir=$WORK_DIR \
    trainer.max_epochs=20 \
    +trainer.limit_val_batches=0 \
    batch_size=256 \
    num_workers=8 \
    wandb.enable=true \
    wandb.entity=vocdex \
    wandb.project=swm-cube \
    "output_model_name=dino_wm_${ENV_TYPE}"

EXIT_CODE=$?

sleep 2
LEFTOVER=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
           | grep -v '^$' || true)
if [ -n "$LEFTOVER" ]; then
    echo "Killing GPU stragglers: $LEFTOVER"
    echo "$LEFTOVER" | xargs -r kill -KILL 2>/dev/null || true
fi

echo "=================================================="
echo "Exit: $EXIT_CODE  End: $(date)"
echo "=================================================="
exit $EXIT_CODE
