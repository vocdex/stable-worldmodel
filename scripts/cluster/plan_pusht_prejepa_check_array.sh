#!/bin/bash
#SBATCH -J swm_pusht_prejepa_chk
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-03:30          # measured: bs=4 50-eps = 2x ~5440s CEM ~= 3h; 3.5h margin
#SBATCH --mem=64G
#SBATCH --array=0-3
#SBATCH --output=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_pusht_prejepa_chk_%A_%a.out
#SBATCH --error=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_pusht_prejepa_chk_%A_%a.err

# -----------------------------------------------------------------------------
# Quick check: baseline + agent_color OOD for BOTH retrained PreJEPA pusht
# ckpts (action-only, no proprio). Tests whether bs=32 or bs=256+scaled-lr
# fixed the 10% baseline, and whether either survives a color variation.
#
# 4 cells: {bs32, bs256} x {baseline, agent_color_red}
# -----------------------------------------------------------------------------

set -u

# cell = ckpt_idx * 2 + variation_idx
CKPTS=(
  "pusht_dinov2_small_actiononly_cachedfeats_psmall__bs32_lr5e-4"
  "pusht_dinov2_small_actiononly_cachedfeats_psmall__bs256_lr4e-3"
)
VARIATIONS=(
  "baseline|null"
  "agent_color_red|{variation:[agent.color],variation_values:{agent.color:[255,0,0]}}"
)

CKPT_IDX=$(( SLURM_ARRAY_TASK_ID / 2 ))
VAR_IDX=$(( SLURM_ARRAY_TASK_ID % 2 ))
CKPT_NAME="${CKPTS[$CKPT_IDX]}"
VENTRY="${VARIATIONS[$VAR_IDX]}"
VLABEL="${VENTRY%%|*}"
OVERRIDE_VALUE="${VENTRY##*|}"

echo "Task ${SLURM_ARRAY_TASK_ID}: ckpt=${CKPT_NAME}  variation=${VLABEL}"
echo "Start: $(date)"

WORK_BIND_DIR=/mnt/lustre/work/martius/mot956/stable-worldmodel
export STABLEWM_HOME=$WORK_BIND_DIR

EPOCH=${EPOCH:-10}
NUM_EVAL=${NUM_EVAL:-50}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_SAMPLES=${NUM_SAMPLES:-300}
N_STEPS=${N_STEPS:-30}
TOPK=${TOPK:-30}
SEED=${SEED:-42}
POLICY="${CKPT_NAME}/weights_epoch_${EPOCH}.pt"

RESULTS_DIR=$WORK_BIND_DIR/checkpoints/${CKPT_NAME}/check_n${NUM_SAMPLES}_eps${NUM_EVAL}_s${SEED}
CELL_DIR="$RESULTS_DIR/${VLABEL}"
mkdir -p "$CELL_DIR"

if [ -f "$CELL_DIR/done.flag" ]; then
    echo "Cell already done. Skipping."; exit 0
fi

set +u
source ~/.bashrc
conda activate /mnt/lustre/work/martius/mot956/.conda/swm
set -u

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
uv sync --extra train --extra env

HYDRA_OVERRIDES=(
    --config-name pusht
    policy="$POLICY"
    eval.dataset_name=pusht_expert_train
    cache_dir="$STABLEWM_HOME"
    eval.num_eval="$NUM_EVAL"
    solver.batch_size="$BATCH_SIZE"
    solver.num_samples="$NUM_SAMPLES"
    solver.n_steps="$N_STEPS"
    solver.topk="$TOPK"
    seed="$SEED"
    "+output.video_path=$CELL_DIR"
)
if [ "$OVERRIDE_VALUE" != "null" ]; then
    HYDRA_OVERRIDES+=("+eval.variation_overrides=${OVERRIDE_VALUE}")
fi

CELL_LOG="$CELL_DIR/eval.log"
echo "Launching: ${HYDRA_OVERRIDES[*]}"
srun --kill-on-bad-exit=1 --unbuffered uv run python scripts/plan/eval_wm.py "${HYDRA_OVERRIDES[@]}" 2>&1 | tee "$CELL_LOG"
EXIT_CODE=${PIPESTATUS[0]}

SR=$(grep -oP "'success_rate':\s*\K[0-9.]+" "$CELL_LOG" | tail -1)
SR=${SR:-NA}
[ "$SR" != "NA" ] && touch "$CELL_DIR/done.flag"
echo "==> ${CKPT_NAME} / ${VLABEL}: SR=${SR}"

# --- Robust GPU/cgroup cleanup so the job step doesn't hang post-eval ---
sleep 3
LEFTOVER=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v '^$' || true)
if [ -n "$LEFTOVER" ]; then
    echo "Killing GPU stragglers: $LEFTOVER"
    echo "$LEFTOVER" | xargs -r kill -KILL 2>/dev/null || true
fi
# Fallback: if anything STILL holds the GPU, force SLURM to tear down the step.
sleep 5
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then
    echo "Stragglers persisted — force-cancelling job step ${SLURM_JOB_ID}"
    scancel --signal=KILL --batch "${SLURM_JOB_ID}" 2>/dev/null || true
fi

echo "End: $(date)  exit=$EXIT_CODE"
exit "$EXIT_CODE"
