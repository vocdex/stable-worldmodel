#!/bin/bash
#SBATCH -J swm_cube_countgen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-06:00
#SBATCH --mem=64G
#SBATCH --array=0-15
#SBATCH --output=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_cube_countgen_%A_%a.out
#SBATCH --error=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_cube_countgen_%A_%a.err

# 4x4 count-generalization matrix for DINO-WM.
# Rows = train count (which model), cols = test count (which env).
# Cell id = train_idx * 4 + test_idx.
#
# Submit after all 4 training jobs finish:
#   sbatch scripts/cluster/plan_ood/cube_count_generalization.sh
#
# Override knobs:
#   EPOCH=20 NUM_EVAL=50 NUM_SAMPLES=300 SEED=42 sbatch ...

COUNTS=(single double triple quadruple)
CONFIGS=(cube cube_double cube_triple cube_quadruple)

TASK_ID=$SLURM_ARRAY_TASK_ID
TRAIN_IDX=$(( TASK_ID / 4 ))
TEST_IDX=$(( TASK_ID % 4 ))
TRAIN_COUNT="${COUNTS[$TRAIN_IDX]}"
TEST_COUNT="${COUNTS[$TEST_IDX]}"
CONFIG="${CONFIGS[$TEST_IDX]}"

EPOCH=${EPOCH:-20}
NUM_EVAL=${NUM_EVAL:-50}
NUM_SAMPLES=${NUM_SAMPLES:-300}
BATCH_SIZE=${BATCH_SIZE:-4}
N_STEPS=${N_STEPS:-30}
TOPK=${TOPK:-30}
SEED=${SEED:-42}

echo "=================================================="
echo "SLURM job=$SLURM_JOB_ID array_task=$TASK_ID"
echo "Node=$SLURM_NODELIST  Partition=$SLURM_JOB_PARTITION"
echo "train=$TRAIN_COUNT  test=$TEST_COUNT  config=$CONFIG"
echo "Start: $(date)"
echo "=================================================="

set +u
source ~/.bashrc
CONDA_ENV_PATH="/mnt/lustre/work/martius/mot956/.conda/swm"
conda activate "$CONDA_ENV_PATH" || { echo "FATAL: conda activate failed"; exit 1; }
set -u

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

export WANDB_DIR=/mnt/lustre/work/martius/mot956/wandb
export HF_HOME=/mnt/lustre/work/martius/mot956/hf
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=/mnt/lustre/work/martius/mot956/torch_hub
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SWM_EVAL_TRUST_CHECKS=1

WORK_DIR=/mnt/lustre/work/martius/mot956/stable-worldmodel
export STABLEWM_HOME=$WORK_DIR
cd "$WORK_DIR"

uv sync --extra train --extra env || { echo "FATAL: uv sync failed"; exit 1; }

POLICY="dino_wm_${TRAIN_COUNT}/weights_epoch_${EPOCH}.pt"
CKPT="$WORK_DIR/checkpoints/$POLICY"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT"
    exit 1
fi

RESULTS_DIR="$WORK_DIR/checkpoints/cube_count_generalization/train_${TRAIN_COUNT}_test_${TEST_COUNT}_e${EPOCH}_n${NUM_SAMPLES}_eps${NUM_EVAL}_s${SEED}"
mkdir -p "$RESULTS_DIR"

if [ -f "$RESULTS_DIR/done.flag" ]; then
    echo "Already done ($RESULTS_DIR/done.flag). Skipping."
    exit 0
fi

echo "Evaluating: policy=dino_wm_${TRAIN_COUNT}  env=cube_${TEST_COUNT}"

CELL_LOG="$RESULTS_DIR/eval.log"
uv run python scripts/plan/eval_wm.py \
    --config-name "$CONFIG" \
    policy="$POLICY" \
    cache_dir="$WORK_DIR" \
    eval.num_eval="$NUM_EVAL" \
    solver.batch_size="$BATCH_SIZE" \
    solver.num_samples="$NUM_SAMPLES" \
    solver.n_steps="$N_STEPS" \
    solver.topk="$TOPK" \
    seed="$SEED" \
    "+output.video_path=$RESULTS_DIR/videos" \
    2>&1 | tee "$CELL_LOG"
EXIT_CODE=${PIPESTATUS[0]}

sleep 2
LEFTOVER=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v '^$' || true)
if [ -n "$LEFTOVER" ]; then
    echo "$LEFTOVER" | xargs -r kill -KILL 2>/dev/null || true
fi

if [ "$EXIT_CODE" -eq 0 ]; then
    SR=$(grep -oP "'success_rate':\s*\K[0-9.]+" "$CELL_LOG" | tail -1)
    SR=${SR:-NA}
    touch "$RESULTS_DIR/done.flag"
    echo "SR=$SR" >> "$RESULTS_DIR/done.flag"
else
    SR="NA"
    echo "FAIL: exit code $EXIT_CODE"
fi

echo "=================================================="
echo "train=$TRAIN_COUNT test=$TEST_COUNT SR=$SR"
echo "End: $(date)  exit=$EXIT_CODE"
echo "=================================================="
exit "$EXIT_CODE"
