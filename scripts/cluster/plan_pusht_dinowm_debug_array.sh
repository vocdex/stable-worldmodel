#!/bin/bash
#SBATCH -J swm_dwm_debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-00:45
#SBATCH --mem=64G
#SBATCH --array=0-5
#SBATCH --output=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_dwm_debug_%A_%a.out
#SBATCH --error=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_dwm_debug_%A_%a.err

# -----------------------------------------------------------------------------
# Debug array: 3 cells x 2 alpha values (proprio weight).
# Saves rollout videos to per-cell dirs so we can VISUALLY confirm:
#   (a) variations are actually rendered
#   (b) the model genuinely succeeds vs lucks into it
#   (c) the role of proprio (alpha=0 disables proprio MSE in cost)
#
# Uses small num_eval=10 so videos finish fast and stay small.
# -----------------------------------------------------------------------------

set -u

CELLS=(
  "baseline|null"
  "agent_color_red|{variation:[agent.color],variation_values:{agent.color:[255,0,0]}}"
  "block_scale_large|{variation:[block.scale],variation_values:{block.scale:60}}"
)
ALPHAS=(1.0 0.0)

# Flatten 3 cells x 2 alphas into 6 array tasks. cell_idx = task % 3; alpha_idx = task / 3.
CELL_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))
ALPHA_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
ENTRY="${CELLS[$CELL_IDX]}"
LABEL="${ENTRY%%|*}"
OVERRIDE_VALUE="${ENTRY##*|}"
ALPHA="${ALPHAS[$ALPHA_IDX]}"

echo "Cell: $LABEL  Alpha: $ALPHA  Task: $SLURM_ARRAY_TASK_ID"

WORK_BIND_DIR=/mnt/lustre/work/martius/mot956/stable-worldmodel
DINO_WM_SRC=/mnt/lustre/work/martius/mot956/dino_wm
DINO_WM_CKPT=$DINO_WM_SRC/checkpoints/outputs/pusht
export STABLEWM_HOME=$WORK_BIND_DIR

NUM_EVAL=${NUM_EVAL:-10}
BATCH_SIZE=${BATCH_SIZE:-10}
NUM_SAMPLES=${NUM_SAMPLES:-300}
N_STEPS=${N_STEPS:-30}
TOPK=${TOPK:-30}
SEED=${SEED:-42}

RESULTS_DIR=$WORK_BIND_DIR/checkpoints/dino_wm_legacy_pusht/debug_videos_eps${NUM_EVAL}_s${SEED}
CELL_DIR="$RESULTS_DIR/${LABEL}__alpha${ALPHA}"
mkdir -p "$CELL_DIR"

set +u
source ~/.bashrc
conda activate /mnt/lustre/work/martius/mot956/.conda/swm
set -u

export HF_HOME=$WORK_BIND_DIR/hf
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
    "+output.video_path=$CELL_DIR"
)
if [ "$OVERRIDE_VALUE" != "null" ]; then
    HYDRA_OVERRIDES+=("+eval.variation_overrides=${OVERRIDE_VALUE}")
fi

CELL_LOG="$CELL_DIR/eval.log"
START_TIME=$(date +%s)

echo "Launching eval_wm.py: ${HYDRA_OVERRIDES[*]}"
srun --kill-on-bad-exit=1 --unbuffered uv run python scripts/plan/eval_wm.py "${HYDRA_OVERRIDES[@]}" 2>&1 | tee "$CELL_LOG"
EXIT_CODE=${PIPESTATUS[0]}

ELAPSED=$(( $(date +%s) - START_TIME ))

SR=$(grep -oP "'success_rate':\s*\K[0-9.]+" "$CELL_LOG" | tail -1)
SR=${SR:-NA}
echo "==> ${LABEL} alpha=${ALPHA}: SR=${SR} (${ELAPSED}s)"
echo "Videos in: $CELL_DIR/rollout_*.mp4"
ls -la "$CELL_DIR"/rollout_*.mp4 2>/dev/null | head -3

exit "$EXIT_CODE"
