"""Rebuild OGBench Cube H5 with re-rendered RGB pixels and segmentation masks.

Reads qpos/qvel from an original cube H5, replays every timestep through
MuJoCo, and writes a new H5 containing:

- `pixels` re-rendered at the chosen resolution (default 256x256),
  blosc/zstd compressed to match the original codec.
- `segmentation` rendered at the same resolution, gzip compressed.
  Three semantic classes:
      0 = background (floor, walls, target ghost)
      1 = cube
      2 = robot arm (UR5e + Robotiq gripper)
- All other datasets (qpos, qvel, actions, observations, episode metadata,
  etc.) copied verbatim from the input.

Usage:
    MUJOCO_GL=egl python scripts/data/extract_cube_masks.py \
        --in-path  ~/.stable_worldmodel/cube_single/cube_single_expert.h5 \
        --out-path ~/.stable_worldmodel/cube_single/cube_single_expert_256.h5
"""
import argparse
import os

os.environ.setdefault('MUJOCO_GL', 'egl')

import h5py
import hdf5plugin
import mujoco
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
    p = argparse.ArgumentParser()
    p.add_argument('--in-path', required=True)
    p.add_argument('--out-path', required=True)
    p.add_argument('--size', type=int, default=256,
                   help='render resolution (default: 256)')
    p.add_argument('--no-shadow', action='store_true',
                   help='disable shadows when rendering pixels')
    p.add_argument('--batch', type=int, default=500,
                   help='frames buffered before flushing to disk')
    p.add_argument('--gzip-level', type=int, default=6,
                   help='gzip level for segmentation')
    args = p.parse_args()

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
    if args.no_shadow:
        env._scene_option.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    lut = build_geom_to_label(env._model)

    chunk_px = (min(args.batch, 64), h, w, 3)
    chunk_seg = (min(args.batch, 64), h, w)

    with h5py.File(args.in_path, 'r') as fin, h5py.File(args.out_path, 'w') as fout:
        n = fin['qpos'].shape[0]

        # Copy all datasets except `pixels` (which we rerender).
        for name in fin.keys():
            if name == 'pixels':
                continue
            fin.copy(name, fout)
            print(f'copied {name}')

        px_ds = fout.create_dataset(
            'pixels', shape=(n, h, w, 3), dtype=np.uint8,
            chunks=chunk_px,
            **hdf5plugin.Blosc(cname='zstd', clevel=5,
                               shuffle=hdf5plugin.Blosc.SHUFFLE),
        )
        seg_ds = fout.create_dataset(
            'segmentation', shape=(n, h, w), dtype=np.uint8,
            chunks=chunk_seg,
            compression='gzip', compression_opts=args.gzip_level, shuffle=True,
        )

        px_buf = np.empty((args.batch, h, w, 3), dtype=np.uint8)
        seg_buf = np.empty((args.batch, h, w), dtype=np.uint8)
        buf_idx = 0

        for i in tqdm.tqdm(range(n), desc=f'Rendering @ {h}x{w}'):
            env.set_state(fin['qpos'][i], fin['qvel'][i])

            raw = env.render(segmentation=True)
            geom_ids = np.clip(raw[:, :, 0], -1, len(lut) - 1)
            mask = np.where(geom_ids >= 0, lut[geom_ids], LABEL_BG)
            cube_dilated = binary_dilation(mask == LABEL_CUBE)
            mask[(mask == LABEL_ARM) & cube_dilated] = LABEL_CUBE
            seg_buf[buf_idx] = mask

            px_buf[buf_idx] = env.render()

            buf_idx += 1
            if buf_idx == args.batch:
                start = i - args.batch + 1
                px_ds[start:start + args.batch] = px_buf
                seg_ds[start:start + args.batch] = seg_buf
                buf_idx = 0

        if buf_idx > 0:
            start = n - buf_idx
            px_ds[start:n] = px_buf[:buf_idx]
            seg_ds[start:n] = seg_buf[:buf_idx]

    env.close()
    print(f'Done. Wrote pixels ({n}, {h}, {w}, 3) and segmentation ({n}, {h}, {w}) to {args.out_path}')


if __name__ == '__main__':
    main()
