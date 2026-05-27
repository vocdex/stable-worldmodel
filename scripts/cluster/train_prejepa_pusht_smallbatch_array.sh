#!/bin/bash
#SBATCH -J swm_pusht_actiondiag
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-12:00
#SBATCH --mem=200G
#SBATCH --array=0-1
#SBATCH --output=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_pusht_actiondiag_%A_%a.out
#SBATCH --error=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/swm_pusht_actiondiag_%A_%a.err

# -----------------------------------------------------------------------------
# Two PushT PreJEPA retrains to debug action-signal shortcut.
# Cell 0: bs=32, lr=5e-4  (matches upstream SWM/DINO-WM exactly)
# Cell 1: bs=256, lr=4e-3 (linear LR scaling for our big batch)
#
# If cell 0 reaches ~70%+ SR and cell 1 doesn't → batch size was the issue.
# If both reach ~70%+ → linear LR scaling rescues large batch.
# If neither → deeper issue (cached features, action embedding dim, ...).
# -----------------------------------------------------------------------------

set -u

CONFIGS=(
  "bs32_lr5e-4|32|5e-4"
  "bs256_lr4e-3|256|4e-3"
)
ENTRY="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
LABEL="${ENTRY%%|*}"
REST="${ENTRY#*|}"
BATCH_SIZE="${REST%%|*}"
LR="${REST##*|}"

echo "Cell ${SLURM_ARRAY_TASK_ID}: label=${LABEL} batch_size=${BATCH_SIZE} lr=${LR}"
echo "Start: $(date)"

set +u
source ~/.bashrc
conda activate /mnt/lustre/work/martius/mot956/.conda/swm
set -u

WORK_BIND_DIR=/mnt/lustre/work/martius/mot956/stable-worldmodel
export STABLEWM_HOME=$WORK_BIND_DIR
export WANDB_DIR=$WORK_BIND_DIR/wandb
export HF_HOME=$WORK_BIND_DIR/hf
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=$WORK_BIND_DIR/torch_hub
export PYTHONUNBUFFERED=1

if [ ! -e "$STABLEWM_HOME/datasets" ] && [ -d "$STABLEWM_HOME/dataset" ]; then
    ln -s "$STABLEWM_HOME/dataset" "$STABLEWM_HOME/datasets"
fi

cd "$WORK_BIND_DIR"
uv sync --extra train --extra env

OUTPUT_NAME="pusht_dinov2_small_actiononly_cachedfeats_psmall__${LABEL}"

srun --kill-on-bad-exit=1 --unbuffered uv run python scripts/train/prejepa.py \
    --config-name prejepa_pusht_features \
    dataset_name=pusht_expert_train_features \
    cache_dir=$STABLEWM_HOME \
    trainer.max_epochs=10 \
    +trainer.limit_val_batches=0 \
    batch_size=${BATCH_SIZE} \
    optimizer.lr=${LR} \
    num_workers=16 \
    output_model_name=${OUTPUT_NAME} \
    wandb.enable=true \
    wandb.entity=vocdex \
    wandb.project=swm-pusht

EXIT_CODE=$?

sleep 2
LEFTOVER=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v '^$' || true)
if [ -n "$LEFTOVER" ]; then
    echo "Killing GPU stragglers: $LEFTOVER"
    echo "$LEFTOVER" | xargs -r kill -KILL 2>/dev/null || true
fi

echo "End: $(date)  exit=$EXIT_CODE"
exit $EXIT_CODE
