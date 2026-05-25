#!/bin/bash
#SBATCH -J swm_prejepa_cube
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-15:00          # ~10 h expected for 20 epochs
#SBATCH --mem=200G
#SBATCH --output=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_prejepa_cube_%j.out
#SBATCH --error=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_prejepa_cube_%j.err

# -----------------------------------------------------------------------------
# Train PreJEPA / DINO-WM on OGbench-Cube (action-only, no proprio).
#
# One-time cluster setup (in an interactive node, NOT login):
#   cat > ~/.condarc <<EOF
#   pkgs_dirs:
#     - $WORK/conda/pkgs
#   channels:
#     - conda-forge
#   EOF
#   conda create -y -p $WORK/.conda/swm python=3.11
#   conda activate $WORK/.conda/swm
#   conda install -y -c conda-forge uv
#   cd /mnt/lustre/work/martius/mot956/stable-worldmodel
#   git fetch && git checkout pusht-multi && git pull
#   uv sync
#
# Submit:
#   sbatch scripts/cluster/train_prejepa_cube.sh
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

# Dataset cache root (SWM hard-codes the subfolder name `datasets` plural;
# symlink dataset/ -> datasets/ if needed)
WORK_BIND_DIR=/mnt/lustre/work/martius/mot956/stable-worldmodel
export STABLEWM_HOME=$WORK_BIND_DIR
if [ ! -e "$STABLEWM_HOME/datasets" ] && [ -d "$STABLEWM_HOME/dataset" ]; then
    ln -s "$STABLEWM_HOME/dataset" "$STABLEWM_HOME/datasets"
fi

cd "$WORK_BIND_DIR"

# Sync project deps into .venv/ via uv. Idempotent — a no-op once synced,
# fast otherwise. Without this, `uv run python` lazy-imports a few packages
# then crashes when a transitive import (hydra, torch, etc.) isn't there.
echo "uv sync (with train + env extras) ..."
# hydra-core / transformers / wandb / stable-pretraining are in the `train`
# optional-dep group (pyproject.toml [project.optional-dependencies]); env
# deps (gymnasium[all], opencv, etc.) are in the `env` group. Plain `uv sync`
# only installs base deps and crashes on `import hydra`.
uv sync --extra train --extra env

echo "torch check: $(uv run python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"
echo ""

srun uv run python scripts/train/prejepa.py \
    --config-name prejepa_cube \
    dataset_name=cube_single_expert \
    cache_dir=$STABLEWM_HOME \
    trainer.max_epochs=20 \
    batch_size=64 \
    num_workers=8 \
    wandb.enable=true \
    wandb.entity=vocdex \
    wandb.project=swm-cube

EXIT_CODE=$?
echo ""
echo "=================================================="
echo "Exit: $EXIT_CODE  End: $(date)"
echo "=================================================="
exit $EXIT_CODE
