#!/bin/bash
#SBATCH -J swm_collect_cube
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --partition=cpu-galvani
#SBATCH --time=0-06:00          # 2500 eps × 200 steps × 10 envs ≈ 2-4h per count
#SBATCH --mem=48G
#SBATCH --output=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/collect_cube_%x_%j.out
#SBATCH --error=/mnt/lustre/work/martius/mot956/stable-worldmodel/logs/collect_cube_%x_%j.err

# Collect expert demonstrations for one OGBench-Cube env_type.
# Run as a SLURM array (one job per count) from the stable-worldmodel root:
#
#   cd /mnt/lustre/work/martius/mot956/stable-worldmodel
#   sbatch --job-name=collect_double --export=ENV_TYPE=double scripts/cluster/data/collect_cube_multi.sh
#   sbatch --job-name=collect_triple --export=ENV_TYPE=triple scripts/cluster/data/collect_cube_multi.sh
#   sbatch --job-name=collect_quadruple --export=ENV_TYPE=quadruple scripts/cluster/data/collect_cube_multi.sh
#
# Output: ~/.stable_worldmodel/datasets/ogbench/cube_{env_type}_expert.h5
# Config: 200 steps/ep, 256x256, num_envs=10, 2500 episodes, visualize_info=False.

ENV_TYPE=${ENV_TYPE:-double}

echo "=================================================="
echo "SLURM Job: $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Node: $SLURM_NODELIST  Partition: $SLURM_JOB_PARTITION"
echo "ENV_TYPE: $ENV_TYPE"
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

echo "Collecting cube_${ENV_TYPE}_expert: 2500 episodes × 200 steps × 10 envs → $OUT_H5"

srun --kill-on-bad-exit=1 --unbuffered uv run python scripts/data/collect_cube.py \
    env_type="$ENV_TYPE" \
    num_traj=2500 \
    cache_dir="$HOME/.stable_worldmodel" \
    world.num_envs=10 \
    world.max_episode_steps=200 \
    world.image_shape=[256,256]

EXIT_CODE=$?
echo "=================================================="
echo "Exit: $EXIT_CODE  End: $(date)"
echo "=================================================="
exit $EXIT_CODE
