#!/bin/bash
#SBATCH -J swm_ood_pusht_dwm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-08:00
#SBATCH --mem=64G
#SBATCH --array=0-21             # 22 cells (cjepa pusht_robustness_lewm matrix)
#SBATCH --output=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_ood_pusht_dwm_%A_%a.out
#SBATCH --error=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_ood_pusht_dwm_%A_%a.err

# -----------------------------------------------------------------------------
# OOD PushT planning matrix using the LEGACY DINO-WM (+proprio) checkpoint.
# Uses the fixed dino_wm_external adapter (commit 04a7e49: 92% baseline SR
# on n=50 locally).
#
# Prereqs on cluster:
#   1. rsync DINO-WM repo to /mnt/lustre/work/martius/mot956/dino_wm/
#      (must contain models/ + checkpoints/outputs/pusht/{hydra.yaml,
#       checkpoints/model_latest.pth})
#   2. pusht_expert_train.h5 must be in $WORK/stable-worldmodel/datasets/
#      (for the env's start-state sampling and stats fitting)
#
# Submit:
#   sbatch scripts/cluster/plan_pusht_dinowm_ood_array.sh
# Override:
#   SEED=0 NUM_EVAL=50 sbatch scripts/cluster/plan_pusht_dinowm_ood_array.sh
# -----------------------------------------------------------------------------

set -u

echo "=================================================="
echo "SLURM job=$SLURM_JOB_ID array_task=$SLURM_ARRAY_TASK_ID"
echo "Node=$SLURM_NODELIST  Partition=$SLURM_JOB_PARTITION"
echo "Start: $(date)"
echo "=================================================="

# --- 22-cell matrix from cjepa/experiments/pusht_robustness_lewm/run.sh ---
# PushT colors are RGB uint8 [0,255]; shapes are int (0=circle, 1=L, 2=tee,
# 3=plus, 4=square, 5=I, 6=Z); scales are floats.
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
DINO_WM_SRC=/mnt/lustre/work/martius/mot956/dino_wm
DINO_WM_CKPT=$DINO_WM_SRC/checkpoints/outputs/pusht
export STABLEWM_HOME=$WORK_BIND_DIR

NUM_EVAL=${NUM_EVAL:-50}
BATCH_SIZE=${BATCH_SIZE:-10}     # external adapter encodes-once, fits bs=10 at N=300 on 40GB
NUM_SAMPLES=${NUM_SAMPLES:-300}
N_STEPS=${N_STEPS:-30}
TOPK=${TOPK:-30}
SEED=${SEED:-42}
ALPHA=${ALPHA:-1.0}

RESULTS_DIR=$WORK_BIND_DIR/checkpoints/dino_wm_legacy_pusht/ood_matrix_n${NUM_SAMPLES}_eps${NUM_EVAL}_s${SEED}
CELL_DIR="$RESULTS_DIR/cells/$LABEL"
CSV="$RESULTS_DIR/ood_matrix.csv"
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
# DINO-WM ckpt pickle references `accelerate` (training-time wrapper);
# install after sync so torch.load can resolve the module path.
uv pip install accelerate || { echo "FATAL: accelerate install failed"; touch "$CELL_DIR/failed.flag"; exit 5; }

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
)
if [ "$OVERRIDE_VALUE" != "null" ]; then
    HYDRA_OVERRIDES+=("+eval.variation_overrides=${OVERRIDE_VALUE}")
fi

CELL_LOG="$CELL_DIR/eval.log"
GPU_LOG="$CELL_DIR/gpu.log"
START_TIME=$(date +%s)

echo "ts,mem_used_mib,mem_total_mib,util_pct" > "$GPU_LOG"
( while true; do
      ts=$(date +%s)
      line=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
             --format=csv,noheader,nounits | head -1 | tr -d ' ')
      echo "${ts},${line}" >> "$GPU_LOG"
      sleep 15
  done ) &
GPU_SAMPLER_PID=$!
trap 'kill $GPU_SAMPLER_PID 2>/dev/null || true' EXIT

echo "Launching eval_wm.py with overrides: ${HYDRA_OVERRIDES[*]}"
srun --kill-on-bad-exit=1 --unbuffered uv run python scripts/plan/eval_wm.py "${HYDRA_OVERRIDES[@]}" 2>&1 | tee "$CELL_LOG"
EXIT_CODE=${PIPESTATUS[0]}

kill $GPU_SAMPLER_PID 2>/dev/null || true
wait $GPU_SAMPLER_PID 2>/dev/null || true

sleep 2
LEFTOVER=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v '^$' || true)
if [ -n "$LEFTOVER" ]; then
    echo "Killing GPU stragglers: $LEFTOVER"
    echo "$LEFTOVER" | xargs -r kill -KILL 2>/dev/null || true
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

PEAK_MEM=$(awk -F, 'NR>1 && $2+0 > max {max=$2+0} END {print max+0}' "$GPU_LOG")
MEAN_UTIL=$(awk -F, 'NR>1 {s+=$4+0; n++} END {if(n>0) printf "%.1f", s/n; else print 0}' "$GPU_LOG")
echo "GPU peak mem: ${PEAK_MEM} MiB  mean util: ${MEAN_UTIL}%"

if [ "$EXIT_CODE" -ne 0 ]; then
    touch "$CELL_DIR/failed.flag"
    SR="NA"
    EP_SUCCESSES="NA"
else
    SR=$(grep -oP "'success_rate':\s*\K[0-9.]+" "$CELL_LOG" | tail -1)
    EP_SUCCESSES=$(grep -oP "'episode_successes':\s*array\(\K\[[^]]+\]" "$CELL_LOG" | tail -1)
    SR=${SR:-NA}
    EP_SUCCESSES=${EP_SUCCESSES:-NA}
    if [ "$SR" = "NA" ]; then
        touch "$CELL_DIR/failed.flag"
    else
        touch "$CELL_DIR/done.flag"
    fi
fi

# Move rollout videos (eval_wm dumps them under ckpt parent dir)
shopt -s nullglob
for mp4 in "$DINO_WM_CKPT"/rollout_*.mp4 "$DINO_WM_CKPT"/../rollout_*.mp4; do
    mv "$mp4" "$CELL_DIR/" 2>/dev/null || true
done
shopt -u nullglob

(
    flock -x 200
    if [ ! -f "$CSV" ]; then
        echo "label,factor,override,SR,elapsed_s,peak_mem_mib,mean_util_pct,episode_successes" > "$CSV"
    fi
    printf '%s,%s,"%s",%s,%s,%s,%s,"%s"\n' \
        "$LABEL" "${LABEL%%_*}" "$OVERRIDE_VALUE" "$SR" \
        "$ELAPSED" "$PEAK_MEM" "$MEAN_UTIL" "$EP_SUCCESSES" >> "$CSV"
) 200>"$CSV.lock"

echo "Done cell '$LABEL' SR=$SR elapsed=${ELAPSED}s"
echo "End: $(date)  exit=$EXIT_CODE"
exit "$EXIT_CODE"
