#!/bin/bash
#SBATCH -J swm_collect_cube
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:1
#SBATCH --time=0-03:00          # plan_oracle, 224²; ~1-2h each with 3500/2500 eps
#SBATCH --mem=48G
#SBATCH --output=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/collect_cube_%x_%j.out
#SBATCH --error=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/collect_cube_%x_%j.err

# Collect expert demonstrations for one OGBench-Cube env_type.
# Uses plan_oracle (smooth trajectories, matches single-cube action distribution).
# Episode steps scale with cube count (N×200); episode counts chosen for ~2M frames each:
#   triple:    3500 eps × 601 steps ≈ 2.1M frames
#   quadruple: 2500 eps × 801 steps ≈ 2.0M frames
#
#   cd /mnt/lustre/work/martius/mot956/stable-worldmodel
#   sbatch --job-name=collect_triple    --export=ENV_TYPE=triple    scripts/cluster/data/collect_cube_multi.sh
#   sbatch --job-name=collect_quadruple --export=ENV_TYPE=quadruple scripts/cluster/data/collect_cube_multi.sh
#
# Output: ~/.stable_worldmodel/datasets/ogbench/cube_{env_type}_expert.h5

ENV_TYPE=${ENV_TYPE:-double}

case "$ENV_TYPE" in
    triple)    MAX_STEPS=600; NUM_TRAJ=3500 ;;
    quadruple) MAX_STEPS=800; NUM_TRAJ=2500 ;;
    *)         MAX_STEPS=200; NUM_TRAJ=5000 ;;
esac

echo "=================================================="
echo "SLURM Job: $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Node: $SLURM_NODELIST  Partition: $SLURM_JOB_PARTITION"
echo "ENV_TYPE: $ENV_TYPE  MAX_STEPS: $MAX_STEPS"
echo "Start: $(date)"
echo "=================================================="

source ~/.bashrc

CONDA_ENV_PATH="/mnt/lustre/work/martius/mot956/.conda/swm"
conda activate "$CONDA_ENV_PATH"

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

WORK_DIR=/mnt/lustre/work/martius/mot956/stable-worldmodel
cd "$WORK_DIR"

uv sync --extra env

OUT_H5="$HOME/.stable_worldmodel/datasets/ogbench/cube_${ENV_TYPE}_expert.h5"
if [ -f "$OUT_H5" ]; then
    echo "ERROR: output already exists: $OUT_H5 — refusing to overwrite."
    exit 1
fi
mkdir -p "$HOME/.stable_worldmodel/datasets/ogbench"
mkdir -p "$WORK_DIR/logs"

echo "Collecting cube_${ENV_TYPE}_expert: ${NUM_TRAJ} episodes × ${MAX_STEPS} steps × 10 envs → $OUT_H5"

srun --kill-on-bad-exit=1 --unbuffered "$CONDA_ENV_PATH/bin/uv" run python scripts/data/collect_cube.py \
    env_type="$ENV_TYPE" \
    num_traj=$NUM_TRAJ \
    cache_dir="$HOME/.stable_worldmodel" \
    world.num_envs=10 \
    world.max_episode_steps=$MAX_STEPS \
    world.image_shape=[224,224] \
    ++policy_type=plan_oracle

EXIT_CODE=$?
echo "=================================================="
echo "Exit: $EXIT_CODE  End: $(date)"
echo "=================================================="
exit $EXIT_CODE
