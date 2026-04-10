"""Extract object segmentation masks for OGBench Cube dataset via MuJoCo.

Replays qpos/qvel from the H5 dataset through the simulator and renders
per-pixel segmentation masks.  Three semantic classes:
    0 = background (floor, walls, target ghost)
    1 = cube
    2 = robot arm (UR5e + Robotiq gripper)

Writes a new 'segmentation' dataset (uint8, shape (N, H, W)) into the
same H5 file.  Resolution defaults to the pixel size stored in the H5
but can be overridden with --size (e.g. --size 256 for 256x256).

Usage:
    MUJOCO_GL=egl python scripts/data/extract_cube_masks.py \
        --h5-path ~/.stable_worldmodel/cube_single/cube_single_expert.h5

    # Render at 256x256 instead:
    MUJOCO_GL=egl python scripts/data/extract_cube_masks.py \
        --h5-path ~/.stable_worldmodel/cube_single/cube_single_expert.h5 \
        --size 256
"""
import argparse
import os

os.environ.setdefault('MUJOCO_GL', 'egl')

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from scipy.ndimage import binary_dilation
import tqdm

from stable_worldmodel.envs.ogbench.cube_env import CubeEnv

LABEL_BG = 0
LABEL_CUBE = 1
LABEL_ARM = 2


def build_geom_to_label(model):
    """Map each MuJoCo geom id to a semantic label."""
    lut = np.zeros(model.ngeom, dtype=np.uint8)
    for gid in range(model.ngeom):
        name = model.geom(gid).name
        if name.startswith('object_') and not name.startswith('target_object_'):
            lut[gid] = LABEL_CUBE
        elif name.startswith('ur5e/'):
            lut[gid] = LABEL_ARM
    return lut


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5-path', required=True)
    parser.add_argument('--size', type=int, default=None,
                        help='render resolution (default: match pixels in H5)')
    parser.add_argument('--batch', type=int, default=1000,
                        help='timesteps to buffer before flushing to disk')
    args = parser.parse_args()

    with h5py.File(args.h5_path, 'r') as f:
        h, w = f['pixels'].shape[1], f['pixels'].shape[2]
    if args.size is not None:
        h = w = args.size

    env = CubeEnv(
        env_type='single',
        ob_type='pixels',
        width=w,
        height=h,
        visualize_info=False,
        terminate_at_goal=False,
        mode='data_collection',
        pixel_transparent_arm=False,
    )
    env.reset()
    # Disable antialiasing for segmentation rendering to avoid border artifacts.
    # See: https://github.com/google-deepmind/dm_control/issues/395
    env._model.vis.quality.offsamples = 0
    lut = build_geom_to_label(env._model)

    ds_name = 'segmentation' if args.size is None else f'segmentation_{args.size}'

    with h5py.File(args.h5_path, 'a') as f:
        n = f['qpos'].shape[0]

        if ds_name in f:
            del f[ds_name]
        seg_ds = f.create_dataset(
            ds_name, shape=(n, h, w), dtype=np.uint8,
            chunks=(min(args.batch, n), h, w),
        )

        buf = np.empty((args.batch, h, w), dtype=np.uint8)
        buf_idx = 0

        for i in tqdm.tqdm(range(n), desc='Rendering segmentation'):
            qpos = f['qpos'][i]
            qvel = f['qvel'][i]
            env.set_state(qpos, qvel)
            raw = env.render(segmentation=True)  # (H, W, 2): [geom_id, geom_type]
            geom_ids = raw[:, :, 0]
            geom_ids = np.clip(geom_ids, -1, len(lut) - 1)
            mask = np.where(geom_ids >= 0, lut[geom_ids], LABEL_BG)
            # Relabel stray arm pixels at cube border (z-fighting residuals).
            cube_dilated = binary_dilation(mask == LABEL_CUBE)
            mask[(mask == LABEL_ARM) & cube_dilated] = LABEL_CUBE
            buf[buf_idx] = mask
            buf_idx += 1

            if buf_idx == args.batch:
                start = i - args.batch + 1
                seg_ds[start:start + args.batch] = buf
                buf_idx = 0

        if buf_idx > 0:
            start = n - buf_idx
            seg_ds[start:n] = buf[:buf_idx]

    env.close()
    print(f'Wrote {ds_name} ({n}, {h}, {w}) to {args.h5_path}')


if __name__ == '__main__':
    main()
