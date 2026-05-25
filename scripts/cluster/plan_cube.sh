#!/bin/bash
#SBATCH -J swm_plan_cube
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-01:00          # 1h budget — full N=300 CEM on 50 episodes lands in ~25-45 min on A100
#SBATCH --mem=64G
#SBATCH --output=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_plan_cube_%j.out
#SBATCH --error=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_plan_cube_%j.err

# -----------------------------------------------------------------------------
# Plan with a trained PreJEPA / DINO-WM cube ckpt.
#
# Usage:
#   sbatch scripts/cluster/plan_cube.sh                          # uses epoch_10
#   EPOCH=20 sbatch scripts/cluster/plan_cube.sh                 # override epoch
#   sbatch scripts/cluster/plan_cube.sh eval.num_eval=20         # extra hydra overrides
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
export MUJOCO_GL=egl
mkdir -p "$WANDB_DIR" "$HF_HOME" "$TORCH_HOME"

# Repo + dataset paths
WORK_BIND_DIR=/mnt/lustre/work/martius/mot956/stable-worldmodel
export STABLEWM_HOME=$WORK_BIND_DIR
if [ ! -e "$STABLEWM_HOME/datasets" ] && [ -d "$STABLEWM_HOME/dataset" ]; then
    ln -s "$STABLEWM_HOME/dataset" "$STABLEWM_HOME/datasets"
fi

cd "$WORK_BIND_DIR"

# Sync project deps (no-op once installed)
echo "uv sync (with train + env extras) ..."
uv sync --extra train --extra env

echo "torch check: $(uv run python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"
echo ""

# Knobs (override via env or hydra args)
EPOCH=${EPOCH:-10}
NUM_EVAL=${NUM_EVAL:-50}
BATCH_SIZE=${BATCH_SIZE:-10}
NUM_SAMPLES=${NUM_SAMPLES:-300}
N_STEPS=${N_STEPS:-30}
TOPK=${TOPK:-30}
POLICY="cube_dinov2_small_actiononly_cachedfeats_psmall/weights_epoch_${EPOCH}.pt"

srun uv run python scripts/plan/eval_wm.py --config-name cube \
    policy="$POLICY" \
    eval.dataset_name=cube_single_expert \
    cache_dir=$STABLEWM_HOME \
    eval.num_eval=$NUM_EVAL \
    solver.batch_size=$BATCH_SIZE \
    solver.num_samples=$NUM_SAMPLES \
    solver.n_steps=$N_STEPS \
    solver.topk=$TOPK \
    "$@"

EXIT_CODE=$?
echo ""
echo "=================================================="
echo "Exit: $EXIT_CODE  End: $(date)"
echo "=================================================="
exit $EXIT_CODE
