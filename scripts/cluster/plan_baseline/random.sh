#!/bin/bash
#SBATCH -J swm_rand_cube
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-00:40          # random = no CEM/model, just env stepping; ~10-15 min/cell
#SBATCH --mem=64G
#SBATCH --array=0-2             # baseline + cube_size_{small,large}
#SBATCH --output=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_rand_cube_%A_%a.out
#SBATCH --error=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_rand_cube_%A_%a.err

# -----------------------------------------------------------------------------
# Random-policy floor for OGBench-Cube under OUR exact eval protocol
# (num_eval=50, goal_offset_steps + success threshold from cube.yaml, seed=42).
# This is the same-protocol reference for the "DINO-WM OOD < random" claim.
#
# Random ignores observations, so its success is purely state-based:
#   - VISUAL variations (color/floor/light/camera) don't change dynamics, so
#     random's floor == baseline floor. We measure `baseline` once and reuse it
#     for all visual cells.
#   - PHYSICS variations (cube.size) DO change dynamics (gripper grasp), so they
#     get their own random floor.
#
# Submit:
#   sbatch scripts/cluster/plan_baseline/random.sh
# -----------------------------------------------------------------------------

set -u

echo "=================================================="
echo "SLURM job=$SLURM_JOB_ID array_task=$SLURM_ARRAY_TASK_ID"
echo "Node=$SLURM_NODELIST  Partition=$SLURM_JOB_PARTITION"
echo "Start: $(date)"
echo "=================================================="

# label | hydra eval.variation_overrides string
CELLS=(
  "baseline|null"
  "cube_size_small|{variation:[cube.size],variation_values:{cube.size:[0.012]}}"
  "cube_size_large|{variation:[cube.size],variation_values:{cube.size:[0.028]}}"
)

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#CELLS[@]}" ]; then
    echo "ERROR: array task $SLURM_ARRAY_TASK_ID out of range (have ${#CELLS[@]} cells)"
    exit 2
fi

ENTRY="${CELLS[$SLURM_ARRAY_TASK_ID]}"
LABEL="${ENTRY%%|*}"
OVERRIDE_VALUE="${ENTRY##*|}"
echo "Cell: $SLURM_ARRAY_TASK_ID  label=$LABEL"
echo "Override: $OVERRIDE_VALUE"

# --- Paths ---
WORK_BIND_DIR=/mnt/lustre/work/martius/mot956/stable-worldmodel
export STABLEWM_HOME=$WORK_BIND_DIR

NUM_EVAL=${NUM_EVAL:-50}
SEED=${SEED:-42}

RESULTS_DIR=$WORK_BIND_DIR/checkpoints/cube_dinov2_small_actiononly_cachedfeats_psmall/random_floor_eps${NUM_EVAL}_s${SEED}
CELL_DIR="$RESULTS_DIR/cells/$LABEL"
CSV="$RESULTS_DIR/random_floor.csv"
mkdir -p "$CELL_DIR" "$RESULTS_DIR"

if [ -f "$CELL_DIR/done.flag" ]; then
    echo "Cell already done. Skipping."
    exit 0
fi
rm -f "$CELL_DIR/failed.flag"

# --- Env activation ---
set +u
source ~/.bashrc
CONDA_ENV_PATH="/mnt/lustre/work/martius/mot956/.conda/swm"
conda activate "$CONDA_ENV_PATH" || { echo "FATAL: conda activate failed"; exit 3; }
set -u

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

# --- Hydra overrides: policy=random (no CEM, no ckpt) ---
HYDRA_OVERRIDES=(
    --config-name cube
    policy=random
    eval.dataset_name=cube_single_expert
    cache_dir="$STABLEWM_HOME"
    eval.num_eval="$NUM_EVAL"
    seed="$SEED"
)
if [ "$OVERRIDE_VALUE" != "null" ]; then
    HYDRA_OVERRIDES+=("+eval.variation_overrides=${OVERRIDE_VALUE}")
fi

CELL_LOG="$CELL_DIR/eval.log"
START_TIME=$(date +%s)

echo "Launching eval_wm.py (random) with overrides: ${HYDRA_OVERRIDES[*]}"
srun --kill-on-bad-exit=1 --unbuffered uv run python scripts/plan/eval_wm.py "${HYDRA_OVERRIDES[@]}" 2>&1 | tee "$CELL_LOG"
EXIT_CODE=${PIPESTATUS[0]}

sleep 2
LEFTOVER=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v '^$' || true)
if [ -n "$LEFTOVER" ]; then
    echo "Killing GPU stragglers: $LEFTOVER"
    echo "$LEFTOVER" | xargs -r kill -KILL 2>/dev/null || true
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

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

(
    flock -x 200
    if [ ! -f "$CSV" ]; then
        echo "label,SR,elapsed_s" > "$CSV"
    fi
    printf '%s,%s,%s\n' "$LABEL" "$SR" "$ELAPSED" >> "$CSV"
) 200>"$CSV.lock"

echo "Done cell '$LABEL' SR=$SR elapsed=${ELAPSED}s"
echo "End: $(date)  exit=$EXIT_CODE"
exit "$EXIT_CODE"
