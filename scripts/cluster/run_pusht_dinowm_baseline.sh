#!/bin/bash
# Smoke test inside an interactive srun on the cluster:
#   srun --partition=2080-galvani --gres=gpu:2080:1 --cpus-per-task=4 \
#        --mem=16G --time=00:30:00 --pty bash
#   cd /mnt/lustre/work/martius/mot956/stable-worldmodel
#   bash scripts/cluster/run_pusht_dinowm_baseline.sh
#
# 30-min budget: just verify the integration loads and runs to first SR.
# Bump knobs via env vars if you want more (NUM_EVAL=10 BATCH_SIZE=2 etc).

set +u
source ~/.bashrc
set -u

WORK_BIND_DIR=/mnt/lustre/work/martius/mot956/stable-worldmodel
DINO_WM_SRC=/mnt/lustre/work/martius/mot956/dino_wm
DINO_WM_CKPT=$DINO_WM_SRC/checkpoints/outputs/pusht

# Tiny so it finishes within 30 min on a 2080:
# 5 envs x N=100 x 30 iters x 2 MPC iters ~= 8-12 min on 2080.
NUM_EVAL=${NUM_EVAL:-5}
BATCH_SIZE=${BATCH_SIZE:-2}
NUM_SAMPLES=${NUM_SAMPLES:-100}
N_STEPS=${N_STEPS:-30}
TOPK=${TOPK:-30}
SEED=${SEED:-42}

conda activate /mnt/lustre/work/martius/mot956/.conda/swm

export STABLEWM_HOME=$WORK_BIND_DIR
export HF_HOME=$WORK_BIND_DIR/hf
export TORCH_HOME=$WORK_BIND_DIR/torch_hub
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$WORK_BIND_DIR"
uv sync --extra train --extra env
uv pip install accelerate

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Running: num_eval=$NUM_EVAL N=$NUM_SAMPLES bs=$BATCH_SIZE"

uv run python scripts/plan/eval_wm.py \
    --config-name pusht_dinowm \
    policy="$DINO_WM_CKPT" \
    dino_wm_src="$DINO_WM_SRC" \
    eval.dataset_name=pusht_expert_train \
    cache_dir="$STABLEWM_HOME" \
    eval.num_eval="$NUM_EVAL" \
    solver.batch_size="$BATCH_SIZE" \
    solver.num_samples="$NUM_SAMPLES" \
    solver.n_steps="$N_STEPS" \
    solver.topk="$TOPK" \
    seed="$SEED"
