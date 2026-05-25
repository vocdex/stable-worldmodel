#!/bin/bash
#SBATCH -J swm_ood_cube
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-04:00          # per-task budget; one cell ≈ 3-3.5h at N=300, 50 eps
#SBATCH --mem=64G
#SBATCH --array=0-12            # 13 cells (cjepa cube_ood_cem300 matrix minus light dim/bright)
#SBATCH --output=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_ood_cube_%A_%a.out
#SBATCH --error=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_ood_cube_%A_%a.err

# -----------------------------------------------------------------------------
# OOD cube planning matrix (15 cells, single seed, N=300 CEM, 50 episodes).
#
# Mirrors cjepa/experiments/cube_ood_cem300_train/run.sh cell list. Designed
# for the realistic galvani cluster constraint of ~2 concurrent A100s →
# expect ~26h total wall-clock (15 cells / 2 parallel × ~3.5h).
#
# Idempotency: each cell writes `${RESULTS_DIR}/cells/<label>/done.flag` on
# success. Re-submitting the same array (e.g. after a crash) only runs the
# cells without a flag. CSV writes use flock so parallel array tasks don't
# clobber each other.
#
# Submit:
#   sbatch scripts/cluster/plan_cube_ood_array.sh
#
# Submit a subset (e.g. just cells 5,8,12 after a partial failure):
#   sbatch --array=5,8,12 scripts/cluster/plan_cube_ood_array.sh
#
# Override knobs via env:
#   EPOCH=20 NUM_EVAL=50 NUM_SAMPLES=300 sbatch scripts/cluster/plan_cube_ood_array.sh
# -----------------------------------------------------------------------------

set -u   # NOT set -e — we want to catch failures per-cell, not crash the array task

echo "=================================================="
echo "SLURM job=$SLURM_JOB_ID array_task=$SLURM_ARRAY_TASK_ID"
echo "Node=$SLURM_NODELIST  Partition=$SLURM_JOB_PARTITION"
echo "Start: $(date)"
echo "=================================================="

# --- Cell matrix (label | hydra eval.variation_overrides string) ---
# Hydra dict format: must NOT contain spaces; commas in nested arrays are fine.
CELLS=(
  "baseline|null"
  "cube_color_red|{variation:[cube.color],variation_values:{cube.color:[[1.0,0.0,0.0]]}}"
  "cube_color_blue|{variation:[cube.color],variation_values:{cube.color:[[0.0,0.0,1.0]]}}"
  "cube_color_yellow|{variation:[cube.color],variation_values:{cube.color:[[1.0,1.0,0.0]]}}"
  "cube_size_small|{variation:[cube.size],variation_values:{cube.size:[0.012]}}"
  "cube_size_large|{variation:[cube.size],variation_values:{cube.size:[0.028]}}"
  "agent_color_red|{variation:[agent.color],variation_values:{agent.color:[1.0,0.0,0.0]}}"
  "agent_color_black|{variation:[agent.color],variation_values:{agent.color:[0.0,0.0,0.0]}}"
  "floor_color_brown|{variation:[floor.color],variation_values:{floor.color:[[0.4,0.25,0.1],[0.5,0.32,0.15]]}}"
  "floor_color_magenta|{variation:[floor.color],variation_values:{floor.color:[[1.0,0.0,1.0],[0.5,0.0,0.5]]}}"
  "camera_yaw_p5|{variation:[camera.angle_delta],variation_values:{camera.angle_delta:[[5.0,0.0]]}}"
  "camera_yaw_m5|{variation:[camera.angle_delta],variation_values:{camera.angle_delta:[[-5.0,0.0]]}}"
  "camera_pitch_p5|{variation:[camera.angle_delta],variation_values:{camera.angle_delta:[[0.0,5.0]]}}"
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

EPOCH=${EPOCH:-10}
NUM_EVAL=${NUM_EVAL:-50}
BATCH_SIZE=${BATCH_SIZE:-5}
NUM_SAMPLES=${NUM_SAMPLES:-300}
N_STEPS=${N_STEPS:-30}
TOPK=${TOPK:-30}
POLICY="cube_dinov2_small_actiononly_cachedfeats_psmall/weights_epoch_${EPOCH}.pt"

RESULTS_DIR=$WORK_BIND_DIR/checkpoints/cube_dinov2_small_actiononly_cachedfeats_psmall/ood_matrix_e${EPOCH}_n${NUM_SAMPLES}_eps${NUM_EVAL}
CELL_DIR="$RESULTS_DIR/cells/$LABEL"
CSV="$RESULTS_DIR/ood_matrix.csv"
mkdir -p "$CELL_DIR"
mkdir -p "$RESULTS_DIR"

# --- Idempotency: skip if cell already succeeded ---
if [ -f "$CELL_DIR/done.flag" ]; then
    echo "Cell already done (found $CELL_DIR/done.flag). Skipping."
    exit 0
fi
# Clear any prior failure marker — we're about to retry
rm -f "$CELL_DIR/failed.flag"

# --- Env activation ---
source ~/.bashrc
CONDA_ENV_PATH="/mnt/lustre/work/martius/mot956/.conda/swm"
conda activate "$CONDA_ENV_PATH" || { echo "FATAL: conda activate failed"; exit 3; }

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Python: $(which python)"

# --- Caches ---
export WANDB_DIR=$WORK_BIND_DIR/wandb
export HF_HOME=$WORK_BIND_DIR/hf
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=$WORK_BIND_DIR/torch_hub
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Dataset symlink (idempotent)
if [ ! -e "$STABLEWM_HOME/datasets" ] && [ -d "$STABLEWM_HOME/dataset" ]; then
    ln -s "$STABLEWM_HOME/dataset" "$STABLEWM_HOME/datasets"
fi

cd "$WORK_BIND_DIR"

# --- uv sync (idempotent, fast no-op on warm cache) ---
echo "uv sync ..."
uv sync --extra train --extra env || { echo "FATAL: uv sync failed"; touch "$CELL_DIR/failed.flag"; exit 4; }

# --- Hydra overrides for this cell ---
HYDRA_OVERRIDES=(
    --config-name cube
    policy="$POLICY"
    eval.dataset_name=cube_single_expert
    cache_dir="$STABLEWM_HOME"
    eval.num_eval="$NUM_EVAL"
    solver.batch_size="$BATCH_SIZE"
    solver.num_samples="$NUM_SAMPLES"
    solver.n_steps="$N_STEPS"
    solver.topk="$TOPK"
)
if [ "$OVERRIDE_VALUE" != "null" ]; then
    HYDRA_OVERRIDES+=("eval.variation_overrides=${OVERRIDE_VALUE}")
fi

# --- Run, capturing both per-cell log and SR ---
CELL_LOG="$CELL_DIR/eval.log"
START_TIME=$(date +%s)

echo "Launching eval_wm.py with overrides: ${HYDRA_OVERRIDES[*]}"
srun uv run python scripts/plan/eval_wm.py "${HYDRA_OVERRIDES[@]}" 2>&1 | tee "$CELL_LOG"
EXIT_CODE=${PIPESTATUS[0]}

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ "$EXIT_CODE" -ne 0 ]; then
    echo "FAIL: eval_wm.py exit code $EXIT_CODE"
    touch "$CELL_DIR/failed.flag"
    # Still log the failure to the CSV so we know about it
    SR="NA"
    EP_SUCCESSES="NA"
else
    # Parse final success_rate + episode_successes from the cell log.
    # eval_wm prints: {'success_rate': 80.0, 'episode_successes': array([True, False, ...])}
    SR=$(grep -oP "'success_rate':\s*\K[0-9.]+" "$CELL_LOG" | tail -1)
    EP_SUCCESSES=$(grep -oP "'episode_successes':\s*array\(\K\[[^]]+\]" "$CELL_LOG" | tail -1)
    SR=${SR:-NA}
    EP_SUCCESSES=${EP_SUCCESSES:-NA}

    if [ "$SR" = "NA" ]; then
        echo "WARN: success_rate not parsed from log; treating as failure"
        touch "$CELL_DIR/failed.flag"
    else
        touch "$CELL_DIR/done.flag"
    fi
fi

# --- Move rollout videos into the cell dir (default lands in ckpt parent) ---
ROLLOUT_SRC=$WORK_BIND_DIR/checkpoints/cube_dinov2_small_actiononly_cachedfeats_psmall
shopt -s nullglob
for mp4 in "$ROLLOUT_SRC"/rollout_*.mp4; do
    mv "$mp4" "$CELL_DIR/" 2>/dev/null || true
done
shopt -u nullglob

# --- Append result row to shared CSV (flock-protected) ---
# Header written by whichever task grabs the lock first.
(
    flock -x 200
    if [ ! -f "$CSV" ]; then
        echo "label,factor,override,SR,elapsed_s,episode_successes" > "$CSV"
    fi
    # Quote ep_successes in case of internal commas
    printf '%s,%s,"%s",%s,%s,"%s"\n' \
        "$LABEL" \
        "${LABEL%%_*}" \
        "$OVERRIDE_VALUE" \
        "$SR" \
        "$ELAPSED" \
        "$EP_SUCCESSES" >> "$CSV"
) 200>"$CSV.lock"

echo "Done cell '$LABEL' SR=$SR elapsed=${ELAPSED}s"
echo "=================================================="
echo "End: $(date)  exit=$EXIT_CODE"
echo "=================================================="

conda deactivate
exit "$EXIT_CODE"
