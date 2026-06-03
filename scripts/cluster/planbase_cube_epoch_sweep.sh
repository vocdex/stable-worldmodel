#!/bin/bash
#SBATCH -J swm_cube_epsweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-08:00
#SBATCH --mem=64G
#SBATCH --array=0-1             # 2 epochs: 10, 15
#SBATCH --output=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_cube_epsweep_%A_%a.out
#SBATCH --error=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_cube_epsweep_%A_%a.err

# -----------------------------------------------------------------------------
# Baseline-only (no OOD) success rate sweep over epoch 10 and 15.
# Tests whether the 71% number at epoch 20 is post-peak (paper claims 86%
# at epoch 10 with same recipe).
#
# Submit:
#   sbatch scripts/cluster/planbase_cube_epoch_sweep.sh
# -----------------------------------------------------------------------------

set -u

echo "=================================================="
echo "SLURM job=$SLURM_JOB_ID array_task=$SLURM_ARRAY_TASK_ID"
echo "Node=$SLURM_NODELIST  Partition=$SLURM_JOB_PARTITION"
echo "Start: $(date)"
echo "=================================================="

EPOCHS=(10 15)
EPOCH=${EPOCHS[$SLURM_ARRAY_TASK_ID]}

NUM_EVAL=${NUM_EVAL:-50}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_SAMPLES=${NUM_SAMPLES:-300}
N_STEPS=${N_STEPS:-30}
TOPK=${TOPK:-30}
SEED=${SEED:-42}

WORK_BIND_DIR=/mnt/lustre/work/martius/mot956/stable-worldmodel
export STABLEWM_HOME=$WORK_BIND_DIR

POLICY="cube_dinov2_small_actiononly_cachedfeats_psmall/weights_epoch_${EPOCH}.pt"
RESULTS_DIR=$WORK_BIND_DIR/checkpoints/cube_dinov2_small_actiononly_cachedfeats_psmall/baseline_epoch_sweep
CELL_DIR="$RESULTS_DIR/e${EPOCH}_s${SEED}"
mkdir -p "$CELL_DIR"

if [ -f "$CELL_DIR/done.flag" ]; then
    echo "Cell already done. Skipping."
    exit 0
fi
rm -f "$CELL_DIR/failed.flag"

set +u
source ~/.bashrc
CONDA_ENV_PATH="/mnt/lustre/work/martius/mot956/.conda/swm"
conda activate "$CONDA_ENV_PATH" || { echo "FATAL: conda activate failed"; exit 3; }
set -u

echo "Epoch: $EPOCH   Policy: $POLICY"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

export WANDB_DIR=$WORK_BIND_DIR/wandb
export HF_HOME=$WORK_BIND_DIR/hf
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=$WORK_BIND_DIR/torch_hub
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -e "$STABLEWM_HOME/datasets" ] && [ -d "$STABLEWM_HOME/dataset" ]; then
    ln -s "$STABLEWM_HOME/dataset" "$STABLEWM_HOME/datasets"
fi

cd "$WORK_BIND_DIR"

echo "uv sync ..."
uv sync --extra train --extra env || { echo "FATAL: uv sync failed"; touch "$CELL_DIR/failed.flag"; exit 4; }

CELL_LOG="$CELL_DIR/eval.log"
START_TIME=$(date +%s)

echo "Launching baseline planning at epoch=$EPOCH ..."
srun --kill-on-bad-exit=1 --unbuffered uv run python scripts/plan/eval_wm.py \
    --config-name cube \
    policy="$POLICY" \
    eval.dataset_name=cube_single_expert \
    cache_dir="$STABLEWM_HOME" \
    eval.num_eval="$NUM_EVAL" \
    solver.batch_size="$BATCH_SIZE" \
    solver.num_samples="$NUM_SAMPLES" \
    solver.n_steps="$N_STEPS" \
    solver.topk="$TOPK" \
    seed="$SEED" 2>&1 | tee "$CELL_LOG"
EXIT_CODE=${PIPESTATUS[0]}

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# Reap GPU stragglers
sleep 2
LEFTOVER=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v '^$' || true)
if [ -n "$LEFTOVER" ]; then
    echo "Killing GPU stragglers: $LEFTOVER"
    echo "$LEFTOVER" | xargs -r kill -KILL 2>/dev/null || true
fi

if [ "$EXIT_CODE" -ne 0 ]; then
    touch "$CELL_DIR/failed.flag"
    SR="NA"
else
    SR=$(grep -oP "'success_rate':\s*\K[0-9.]+" "$CELL_LOG" | tail -1)
    SR=${SR:-NA}
    if [ "$SR" = "NA" ]; then
        touch "$CELL_DIR/failed.flag"
    else
        touch "$CELL_DIR/done.flag"
    fi
fi

echo "Epoch $EPOCH baseline SR=$SR elapsed=${ELAPSED}s"
echo "=================================================="
echo "End: $(date)  exit=$EXIT_CODE"
echo "=================================================="
exit "$EXIT_CODE"
