#!/bin/bash
#SBATCH -J swm_cube_dv3_256_epsweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=galvani-cn227  # every timeout in the cube matrices happened on cn227
#SBATCH --time=0-04:00
#SBATCH --mem=64G
#SBATCH --array=0-11             # 4 epochs (5,10,15,20) x 3 seeds (0,1,2)
#SBATCH --output=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_cube_dv3_256_epsweep_%A_%a.out
#SBATCH --error=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_cube_dv3_256_epsweep_%A_%a.err

# -----------------------------------------------------------------------------
# 256px-variant baseline epoch sweep: e5/e10/e15/e20 x seeds {0,1,2} for
# cube_dinov3_small_256_actiononly_cachedfeats_psmall (DINOv3-S/16 fed
# 256px-upscaled frames -> 256-patch grid). eval.img_size=256 matches the
# training-side upscale; the env still renders 224 and the transform
# upscales, exactly as during feature extraction.
#
# Run scripts/cluster/train/cube_dinov3_256.sh first.
#
# Task layout: EPOCH = task % 4, SEED = task / 4.
#
# Submit:
#   sbatch scripts/cluster/plan_baseline/cube_dinov3_256_epoch_sweep.sh
# -----------------------------------------------------------------------------

set -u

echo "=================================================="
echo "SLURM job=$SLURM_JOB_ID array_task=$SLURM_ARRAY_TASK_ID"
echo "Node=$SLURM_NODELIST  Partition=$SLURM_JOB_PARTITION"
echo "Start: $(date)"
echo "=================================================="

EPOCHS=(5 10 15 20)
SEEDS=(0 1 2)
EPOCH=${EPOCHS[$((SLURM_ARRAY_TASK_ID % 4))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / 4))]}

NUM_EVAL=${NUM_EVAL:-50}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_SAMPLES=${NUM_SAMPLES:-300}
N_STEPS=${N_STEPS:-30}
TOPK=${TOPK:-30}

WORK_BIND_DIR=/mnt/lustre/work/martius/mot956/stable-worldmodel
export STABLEWM_HOME=$WORK_BIND_DIR

POLICY="cube_dinov3_small_256_actiononly_cachedfeats_psmall/weights_epoch_${EPOCH}.pt"
RESULTS_DIR=$WORK_BIND_DIR/checkpoints/cube_dinov3_small_256_actiononly_cachedfeats_psmall/baseline_epoch_sweep
CELL_DIR="$RESULTS_DIR/e${EPOCH}_s${SEED}"
CSV="$RESULTS_DIR/sweep.csv"
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

echo "Epoch: $EPOCH  Seed: $SEED  Policy: $POLICY"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# HF_HOME matches the other dinov3 scripts ($WORK-level) — eval_wm rebuilds
# the gated DINOv3 encoder via from_pretrained at load time.
export WANDB_DIR=$WORK_BIND_DIR/wandb
export HF_HOME=/mnt/lustre/work/martius/mot956/hf
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=$WORK_BIND_DIR/torch_hub
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DINOV3_SNAPSHOT="$HF_HOME/hub/models--facebook--dinov3-vits16-pretrain-lvd1689m"
CACHED_CFG=$(find -L "$DINOV3_SNAPSHOT/snapshots" -maxdepth 2 -name config.json 2>/dev/null | head -1)
if [ -z "$CACHED_CFG" ] && [ -z "${HF_TOKEN:-}" ] && [ ! -f "$HF_HOME/token" ]; then
    echo "FATAL: no complete DINOv3 snapshot in $HF_HOME and no HF token found."
    touch "$CELL_DIR/failed.flag"
    exit 3
fi

if [ ! -e "$STABLEWM_HOME/datasets" ] && [ -d "$STABLEWM_HOME/dataset" ]; then
    ln -s "$STABLEWM_HOME/dataset" "$STABLEWM_HOME/datasets"
fi

cd "$WORK_BIND_DIR"

echo "uv sync ..."
uv sync --extra train --extra env || { echo "FATAL: uv sync failed"; touch "$CELL_DIR/failed.flag"; exit 4; }

CELL_LOG="$CELL_DIR/eval.log"
START_TIME=$(date +%s)

echo "Launching baseline planning at epoch=$EPOCH seed=$SEED ..."
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
    seed="$SEED" \
    eval.img_size=256 \
    "+output.video_path=$CELL_DIR/videos" 2>&1 | tee "$CELL_LOG"
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

# Shared sweep CSV (flock-protected)
(
    flock -x 200
    if [ ! -f "$CSV" ]; then
        echo "epoch,seed,SR,elapsed_s" > "$CSV"
    fi
    printf '%s,%s,%s,%s\n' "$EPOCH" "$SEED" "$SR" "$ELAPSED" >> "$CSV"
) 200>"$CSV.lock"

echo "Epoch $EPOCH seed $SEED baseline SR=$SR elapsed=${ELAPSED}s"
echo "=================================================="
echo "End: $(date)  exit=$EXIT_CODE"
echo "=================================================="
exit "$EXIT_CODE"
