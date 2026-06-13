#!/bin/bash
#SBATCH -J swm_ood_pusht_dwm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-01:30          # n_steps=10 cuts CEM ~3x (~11 min); ~55 min env stepping dominates -> ~66 min/cell, 90 min margin
#SBATCH --mem=64G
#SBATCH --array=0-49            # 25 cells x 2 seeds (cell = task%25, seed = task/25)
#SBATCH --output=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_ood_pusht_dwm_%A_%a.out
#SBATCH --error=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_ood_pusht_dwm_%A_%a.err

# -----------------------------------------------------------------------------
# LEGACY DINO-WM (+proprio ckpt) PushT OOD matrix, NO PROPRIO COST (alpha=0).
#
# Cells = cjepa/experiments/pusht/pusht_robustness_sc256_nms0_noproprio matrix
# (agent/block/goal color+scale+shape, distractor.color) PLUS background.color
# (cjepa/experiments/pusht/pusht_robustness_background). 25 cells x 3 seeds.
#
# alpha=0 zeros the proprio MSE term in the planning cost (visual-only cost).
# NOTE: the proprio token is still fed to the model — alpha=0 only removes it
# from the cost, it does not make the model proprio-blind.
#
# Prereqs on cluster:
#   1. /mnt/lustre/work/martius/mot956/dino_wm/ (models/ + checkpoints/outputs/pusht)
#   2. datasets/pusht_expert_train.h5
#
# Submit:
#   sbatch scripts/cluster/plan_ood/pusht_dinowm.sh
# Override e.g.:
#   ALPHA=1.0 sbatch scripts/cluster/plan_ood/pusht_dinowm.sh   # with proprio cost
# -----------------------------------------------------------------------------

set -u

echo "=================================================="
echo "SLURM job=$SLURM_JOB_ID array_task=$SLURM_ARRAY_TASK_ID"
echo "Node=$SLURM_NODELIST  Partition=$SLURM_JOB_PARTITION"
echo "Start: $(date)"
echo "=================================================="

# --- Cell matrix (label | hydra eval.variation_overrides string) ---
# PushT colors are RGB uint8 [0,255]; shapes are int (1=L,4=square,5=I,7=plus);
# scales are floats.
CELLS=(
  "baseline|null"
  "agent_color_red|{variation:[agent.color],variation_values:{agent.color:[255,0,0]}}"
  "agent_color_yellow|{variation:[agent.color],variation_values:{agent.color:[255,255,0]}}"
  "agent_color_black|{variation:[agent.color],variation_values:{agent.color:[0,0,0]}}"
  "agent_scale_small|{variation:[agent.scale],variation_values:{agent.scale:20}}"
  "agent_scale_large|{variation:[agent.scale],variation_values:{agent.scale:60}}"
  "agent_shape_L|{variation:[agent.shape],variation_values:{agent.shape:1}}"
  "agent_shape_square|{variation:[agent.shape],variation_values:{agent.shape:4}}"
  "agent_shape_plus|{variation:[agent.shape],variation_values:{agent.shape:7}}"
  "block_color_red|{variation:[block.color],variation_values:{block.color:[255,0,0]}}"
  "block_color_blue|{variation:[block.color],variation_values:{block.color:[0,0,255]}}"
  "block_color_black|{variation:[block.color],variation_values:{block.color:[0,0,0]}}"
  "block_scale_small|{variation:[block.scale],variation_values:{block.scale:20}}"
  "block_scale_large|{variation:[block.scale],variation_values:{block.scale:60}}"
  "block_shape_L|{variation:[block.shape],variation_values:{block.shape:1}}"
  "block_shape_square|{variation:[block.shape],variation_values:{block.shape:4}}"
  "block_shape_I|{variation:[block.shape],variation_values:{block.shape:5}}"
  "goal_color_red|{variation:[goal.color],variation_values:{goal.color:[255,0,0]}}"
  "goal_color_blue|{variation:[goal.color],variation_values:{goal.color:[0,0,255]}}"
  "goal_color_black|{variation:[goal.color],variation_values:{goal.color:[0,0,0]}}"
  "distractor_color_gray|{variation:[distractor.color],variation_values:{distractor.color:[128,128,128]}}"
  "distractor_color_magenta|{variation:[distractor.color],variation_values:{distractor.color:[255,0,255]}}"
  "background_color_red|{variation:[background.color],variation_values:{background.color:[255,0,0]}}"
  "background_color_blue|{variation:[background.color],variation_values:{background.color:[0,0,255]}}"
  "background_color_black|{variation:[background.color],variation_values:{background.color:[0,0,0]}}"
)
SEEDS=(0 1)

NUM_CELLS=${#CELLS[@]}
TOTAL=$(( NUM_CELLS * ${#SEEDS[@]} ))
if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL" ]; then
    echo "ERROR: task $SLURM_ARRAY_TASK_ID out of range (have $TOTAL = $NUM_CELLS cells x ${#SEEDS[@]} seeds)"
    exit 2
fi

CELL_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_CELLS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_CELLS ))
SEED="${SEEDS[$SEED_IDX]}"

ENTRY="${CELLS[$CELL_IDX]}"
LABEL="${ENTRY%%|*}"
OVERRIDE_VALUE="${ENTRY##*|}"
echo "Task $SLURM_ARRAY_TASK_ID  cell=$LABEL  seed=$SEED"
echo "Override: $OVERRIDE_VALUE"

# --- Paths ---
WORK_BIND_DIR=/mnt/lustre/work/martius/mot956/stable-worldmodel
DINO_WM_SRC=/mnt/lustre/work/martius/mot956/dino_wm
DINO_WM_CKPT=$DINO_WM_SRC/checkpoints/outputs/pusht
export STABLEWM_HOME=$WORK_BIND_DIR

NUM_EVAL=${NUM_EVAL:-50}
BATCH_SIZE=${BATCH_SIZE:-10}     # external adapter encodes-once, fits bs=10 at N=300 on 40GB
NUM_SAMPLES=${NUM_SAMPLES:-300}
N_STEPS=${N_STEPS:-10}
TOPK=${TOPK:-30}
ALPHA=${ALPHA:-0.0}              # no proprio cost (visual-only planning cost)

RESULTS_DIR=$WORK_BIND_DIR/checkpoints/dino_wm_legacy_pusht/robust_a${ALPHA}_n${NUM_SAMPLES}_eps${NUM_EVAL}_s${SEED}
CELL_DIR="$RESULTS_DIR/cells/$LABEL"
CSV="$RESULTS_DIR/robust_matrix.csv"
mkdir -p "$CELL_DIR" "$RESULTS_DIR"

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

HYDRA_OVERRIDES=(
    --config-name pusht_dinowm
    policy="$DINO_WM_CKPT"
    dino_wm_src="$DINO_WM_SRC"
    dino_wm_alpha="$ALPHA"
    eval.dataset_name=pusht_expert_train
    cache_dir="$STABLEWM_HOME"
    eval.num_eval="$NUM_EVAL"
    solver.batch_size="$BATCH_SIZE"
    solver.num_samples="$NUM_SAMPLES"
    solver.n_steps="$N_STEPS"
    solver.topk="$TOPK"
    seed="$SEED"
    "+output.video_path=$CELL_DIR/videos"
)
if [ "$OVERRIDE_VALUE" != "null" ]; then
    HYDRA_OVERRIDES+=("+eval.variation_overrides=${OVERRIDE_VALUE}")
fi

CELL_LOG="$CELL_DIR/eval.log"
START_TIME=$(date +%s)

echo "Launching eval_wm.py with overrides: ${HYDRA_OVERRIDES[*]}"
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
    SR="NA"; EP_SUCCESSES="NA"
else
    SR=$(grep -oP "'success_rate':\s*\K[0-9.]+" "$CELL_LOG" | tail -1)
    EP_SUCCESSES=$(grep -oP "'episode_successes':\s*array\(\K\[[^]]+\]" "$CELL_LOG" | tail -1)
    SR=${SR:-NA}; EP_SUCCESSES=${EP_SUCCESSES:-NA}
    if [ "$SR" = "NA" ]; then touch "$CELL_DIR/failed.flag"; else touch "$CELL_DIR/done.flag"; fi
fi

(
    flock -x 200
    if [ ! -f "$CSV" ]; then
        echo "label,factor,seed,alpha,override,SR,elapsed_s,episode_successes" > "$CSV"
    fi
    printf '%s,%s,%s,%s,"%s",%s,%s,"%s"\n' \
        "$LABEL" "${LABEL%%_*}" "$SEED" "$ALPHA" "$OVERRIDE_VALUE" \
        "$SR" "$ELAPSED" "$EP_SUCCESSES" >> "$CSV"
) 200>"$CSV.lock"

echo "Done cell '$LABEL' seed=$SEED SR=$SR elapsed=${ELAPSED}s"
echo "End: $(date)  exit=$EXIT_CODE"
exit "$EXIT_CODE"
