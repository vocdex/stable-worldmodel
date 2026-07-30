#!/bin/bash
#SBATCH -J swm_ood_cube_dinov3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-03:35          # per-task budget; DINOv2 cells measured ~3h16m; DINOv3 has fewer patches (196 vs 256) so this leaves headroom
#SBATCH --mem=64G
#SBATCH --array=0-44            # 45 tasks = 15 cells x 3 seeds (42, 0, 1)
#SBATCH --output=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_ood_cube_dinov3_%A_%a.out
#SBATCH --error=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_ood_cube_dinov3_%A_%a.err

# -----------------------------------------------------------------------------
# DINOv3 ablation: OOD cube planning matrix for the
# cube_dinov3_small_actiononly_cachedfeats_psmall checkpoint.
#
# Same 15 cells (baseline = in-distribution + 14 OOD variations), same eval
# settings (EPOCH=10, N=300 CEM, 50 episodes) as the DINOv2 matrices at
# checkpoints/cube_dinov2_small_actiononly_cachedfeats_psmall/
# ood_matrix_e10_n300_eps50_s{0,1,42} — results are directly comparable.
#
# Seeds are baked into the array: task = seed_idx * 15 + cell, with seeds
# ordered (42, 0, 1) so tasks 0-14 reproduce the primary seed-42 matrix first.
#
# Branch: dinov3-cube-ablation (train with scripts/cluster/train/cube_dinov3.sh
# first). Needs the gated DINOv3 weights in $HF_HOME (see the train script
# header for the one-time rsync).
#
# Idempotency: each cell writes `${RESULTS_DIR}/cells/<label>/done.flag` on
# success. Re-submitting the same array (e.g. after a crash) only runs the
# cells without a flag. CSV writes use flock so parallel array tasks don't
# clobber each other.
#
# Submit:
#   sbatch scripts/cluster/plan_ood/cube_dinov3.sh
#
# Submit a subset (e.g. seed-42 cells only):
#   sbatch --array=0-14 scripts/cluster/plan_ood/cube_dinov3.sh
#
# Override knobs via env:
#   EPOCH=20 NUM_EVAL=50 NUM_SAMPLES=300 sbatch scripts/cluster/plan_ood/cube_dinov3.sh
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
  "light_intensity_dim|{variation:[light.intensity],variation_values:{light.intensity:[0.3]}}"
  "light_intensity_bright|{variation:[light.intensity],variation_values:{light.intensity:[0.95]}}"
)

SEEDS=(42 0 1)
N_CELLS=${#CELLS[@]}
TOTAL=$((N_CELLS * ${#SEEDS[@]}))
if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL" ]; then
    echo "ERROR: array task $SLURM_ARRAY_TASK_ID out of range (have $TOTAL tasks)"
    exit 2
fi

CELL_IDX=$((SLURM_ARRAY_TASK_ID % N_CELLS))
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / N_CELLS))]}

ENTRY="${CELLS[$CELL_IDX]}"
LABEL="${ENTRY%%|*}"
OVERRIDE_VALUE="${ENTRY##*|}"
echo "Cell: $CELL_IDX  label=$LABEL  seed=$SEED"
echo "Override: $OVERRIDE_VALUE"

# --- Paths ---
WORK_BIND_DIR=/mnt/lustre/work/martius/mot956/stable-worldmodel
export STABLEWM_HOME=$WORK_BIND_DIR

EPOCH=${EPOCH:-10}
NUM_EVAL=${NUM_EVAL:-50}
BATCH_SIZE=${BATCH_SIZE:-4}   # bs=5 OOMs at num_eval=50 (50 EGL renderers ≈ 6 GiB env-side)
NUM_SAMPLES=${NUM_SAMPLES:-300}
N_STEPS=${N_STEPS:-30}
TOPK=${TOPK:-30}
POLICY="cube_dinov3_small_actiononly_cachedfeats_psmall/weights_epoch_${EPOCH}.pt"

RESULTS_DIR=$WORK_BIND_DIR/checkpoints/cube_dinov3_small_actiononly_cachedfeats_psmall/ood_matrix_e${EPOCH}_n${NUM_SAMPLES}_eps${NUM_EVAL}_s${SEED}
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
# /etc/bashrc on galvani references unbound vars (BASHRCSOURCED); temporarily
# drop -u so sourcing it doesn't kill the task.
set +u
source ~/.bashrc
CONDA_ENV_PATH="/mnt/lustre/work/martius/mot956/.conda/swm"
conda activate "$CONDA_ENV_PATH" || { echo "FATAL: conda activate failed"; exit 3; }
set -u

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Python: $(which python)"

# --- Caches ---
# HF_HOME matches scripts/cluster/train/cube_dinov3.sh ($WORK-level, NOT
# $WORK_BIND_DIR/hf like the dinov2 plan script) so the gated DINOv3 snapshot
# only needs to be rsynced to one place. eval_wm rebuilds the encoder via
# AutoModel.from_pretrained, so the weights are needed at plan time too.
export WANDB_DIR=$WORK_BIND_DIR/wandb
export HF_HOME=/mnt/lustre/work/martius/mot956/hf
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=$WORK_BIND_DIR/torch_hub
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Pre-flight: gated DINOv3 weights must be reachable ---
DINOV3_SNAPSHOT="$HF_HOME/hub/models--facebook--dinov3-vits16-pretrain-lvd1689m"
if [ ! -d "$DINOV3_SNAPSHOT" ] && [ -z "${HF_TOKEN:-}" ]; then
    echo "FATAL: DINOv3 weights not in $HF_HOME and HF_TOKEN unset."
    echo "See scripts/cluster/train/cube_dinov3.sh header for the rsync."
    touch "$CELL_DIR/failed.flag"
    exit 3
fi

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
    seed="$SEED"
    "+output.video_path=$CELL_DIR/videos"
)
if [ "$OVERRIDE_VALUE" != "null" ]; then
    HYDRA_OVERRIDES+=("+eval.variation_overrides=${OVERRIDE_VALUE}")
fi

# --- Run, capturing both per-cell log and SR ---
CELL_LOG="$CELL_DIR/eval.log"
GPU_LOG="$CELL_DIR/gpu.log"
START_TIME=$(date +%s)

# Background GPU sampler (every 15s: ts, mem_used_MiB, mem_total_MiB, util%)
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

# --- Reap GPU stragglers ---
# PyTorch DataLoader workers + wandb threads + EGL contexts sometimes
# outlive the parent process. Each task owns its own A100, so anything
# in nvidia-smi here is ours and safe to kill.
sleep 2
LEFTOVER=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
           | grep -v '^$' || true)
if [ -n "$LEFTOVER" ]; then
    echo "Killing GPU stragglers: $LEFTOVER"
    echo "$LEFTOVER" | xargs -r kill -KILL 2>/dev/null || true
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# --- GPU usage summary ---
PEAK_MEM=$(awk -F, 'NR>1 && $2+0 > max {max=$2+0} END {print max+0}' "$GPU_LOG")
MAX_UTIL=$(awk -F, 'NR>1 && $4+0 > max {max=$4+0} END {print max+0}' "$GPU_LOG")
MEAN_UTIL=$(awk -F, 'NR>1 {s+=$4+0; n++} END {if(n>0) printf "%.1f", s/n; else print 0}' "$GPU_LOG")
echo "GPU peak mem: ${PEAK_MEM} MiB   util max=${MAX_UTIL}%  mean=${MEAN_UTIL}%"
# Heuristic: util mean <5% over a long run => process probably crashed
if [ "$ELAPSED" -gt 120 ] && [ "${MEAN_UTIL%.*}" -lt 5 ] 2>/dev/null; then
    echo "WARN: GPU mean util ${MEAN_UTIL}% over ${ELAPSED}s — process may have crashed early"
fi

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

# Rollout videos land directly in $CELL_DIR/videos via +output.video_path.

# --- Append result row to shared CSV (flock-protected) ---
# Header written by whichever task grabs the lock first.
(
    flock -x 200
    if [ ! -f "$CSV" ]; then
        echo "label,factor,override,SR,elapsed_s,peak_mem_mib,mean_util_pct,episode_successes" > "$CSV"
    fi
    # Quote ep_successes in case of internal commas
    printf '%s,%s,"%s",%s,%s,%s,%s,"%s"\n' \
        "$LABEL" \
        "${LABEL%%_*}" \
        "$OVERRIDE_VALUE" \
        "$SR" \
        "$ELAPSED" \
        "$PEAK_MEM" \
        "$MEAN_UTIL" \
        "$EP_SUCCESSES" >> "$CSV"
) 200>"$CSV.lock"

echo "Done cell '$LABEL' seed=$SEED SR=$SR elapsed=${ELAPSED}s"
echo "=================================================="
echo "End: $(date)  exit=$EXIT_CODE"
echo "=================================================="

conda deactivate
exit "$EXIT_CODE"
