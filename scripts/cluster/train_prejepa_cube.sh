#!/bin/bash
#SBATCH -J swm_prejepa_cube
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-15:00  # D-HH:MM — ~10 h expected for 20 epochs on A100
#SBATCH --mem=200G
#SBATCH --output=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_prejepa_cube_%j.out
#SBATCH --error=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_prejepa_cube_%j.err

set -eo pipefail

# -----------------------------------------------------------------------------
# Train PreJEPA / DINO-WM on OGbench-Cube (action-only, no proprio).
#
# One-time cluster setup (do this in an INTERACTIVE node, not the login node;
# follow the cluster's "Do not initialize Conda on login" warning):
#
#   # 1. Ensure ~/.condarc points pkgs to $WORK so $HOME isn't filled:
#   cat > ~/.condarc <<EOF
#   pkgs_dirs:
#     - $WORK/conda/pkgs
#   channels:
#     - conda-forge
#   EOF
#
#   # 2. Create the env + install uv from conda-forge:
#   conda create -y -p $WORK/.conda/swm python=3.11
#   conda activate $WORK/.conda/swm
#   conda install -y -c conda-forge uv
#
#   # 3. Clone + sync the repo's Python deps:
#   cd /mnt/lustre/work/martius/mot956/stable-worldmodel
#   git fetch && git checkout pusht-multi && git pull
#   uv sync
#
# Submit (from anywhere):
#   sbatch /mnt/lustre/work/martius/mot956/stable-worldmodel/scripts/cluster/train_prejepa_cube.sh
# -----------------------------------------------------------------------------

# Diagnostic header — so silent failures aren't silent.
echo "=== ENVIRONMENT @ $(date) ==="
echo "node=$(hostname)  user=$(whoami)  pwd=$(pwd)"
echo "HOME=$HOME"
echo "WORK=${WORK:-<UNSET>}"
echo "============================="

# Hardcode the lustre root in case $WORK isn't exported into this
# non-interactive sbatch context.
WORK=${WORK:-/mnt/lustre/work/martius/mot956}
echo "Using WORK=$WORK"

source ~/.bashrc
which conda || { echo "ERROR: conda not on PATH after 'source ~/.bashrc'" >&2; exit 1; }
conda activate "$WORK/.conda/swm"
which uv || { echo "ERROR: uv not on PATH after 'conda activate'" >&2; exit 1; }
which python
echo "conda env: ${CONDA_DEFAULT_ENV:-?}"

# --- Caches (avoid filling $HOME quota) ---
SHARED_CACHE=$WORK
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

# --- Repo + dataset paths ---
WORK_BIND_DIR=/mnt/lustre/work/martius/mot956/stable-worldmodel
export STABLEWM_HOME=$WORK_BIND_DIR

# SWM hard-codes the dataset subfolder name as `datasets` (plural). Auto-create
# a symlink `datasets -> dataset` if needed so HDF5Dataset can resolve the h5.
if [ ! -e "$STABLEWM_HOME/datasets" ] && [ -d "$STABLEWM_HOME/dataset" ]; then
    ln -s "$STABLEWM_HOME/dataset" "$STABLEWM_HOME/datasets"
fi

# --- Run training using uv ---
cd "$WORK_BIND_DIR"
srun uv run python scripts/train/prejepa.py \
    --config-name prejepa_cube \
    cache_dir=$STABLEWM_HOME \
    trainer.max_epochs=20 \
    batch_size=64 \
    num_workers=8 \
    wandb.enable=true \
    wandb.entity=vocdex \
    wandb.project=swm-cube \
    2>&1

conda deactivate
