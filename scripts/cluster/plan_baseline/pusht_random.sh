#!/bin/bash
#SBATCH -J swm_rand_pusht
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-00:40          # random = no CEM/model, just env stepping
#SBATCH --mem=64G
#SBATCH --array=0-3             # baseline + 3 dynamic-distractor cells
#SBATCH --output=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_rand_pusht_%A_%a.out
#SBATCH --error=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_rand_pusht_%A_%a.err

# -----------------------------------------------------------------------------
# Random-policy floor for PushT under our exact eval protocol, including the
# new dynamic-distractor cells (docs/dynamic_ood_cells.md). The new cells are
# purely visual, so their floors must MATCH the baseline floor — running them
# anyway is a cheap harness check (a deviation means the cell changed the
# dynamics or the eval, i.e. a bug).
#
# bg_* cells need textures: python scripts/data/fetch_textures.py once, with
# SWM_TEXTURE_DIR pointing at the same dir used below.
#
# Submit:
#   sbatch scripts/cluster/plan_baseline/pusht_random.sh
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
  "distractor_moving|{variation:[distractor.motion],variation_values:{distractor.motion:[40,20]}}"
  "bg_natural_static|{variation:[background.texture_id],variation_values:{background.texture_id:1}}"
  "bg_video_dynamic|{variation:[background.texture_id],variation_values:{background.texture_id:4}}"
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

RESULTS_DIR=$WORK_BIND_DIR/checkpoints/dino_wm_legacy_pusht/random_floor_eps${NUM_EVAL}_s${SEED}
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
# fail loudly if the eval trust chain is broken (goal/start frame checks)
export SWM_EVAL_TRUST_CHECKS=1
export SWM_TEXTURE_DIR=${SWM_TEXTURE_DIR:-$WORK_BIND_DIR/textures}

if [ ! -e "$STABLEWM_HOME/datasets" ] && [ -d "$STABLEWM_HOME/dataset" ]; then
    ln -s "$STABLEWM_HOME/dataset" "$STABLEWM_HOME/datasets"
fi

cd "$WORK_BIND_DIR"

echo "uv sync ..."
uv sync --extra train --extra env || { echo "FATAL: uv sync failed"; touch "$CELL_DIR/failed.flag"; exit 4; }

# --- Hydra overrides: policy=random (no CEM, no ckpt) ---
HYDRA_OVERRIDES=(
    --config-name pusht
    policy=random
    eval.dataset_name=pusht_expert_train
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
