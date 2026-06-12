"""Rebuild OGBench Cube H5 with re-rendered RGB pixels and segmentation masks.

Reads qpos/qvel from an original cube H5, replays every timestep through
MuJoCo, and writes a new H5 containing:

- `pixels` re-rendered at the chosen resolution (default 256x256),
  blosc/zstd compressed to match the original codec.
- `segmentation` rendered at the same resolution, gzip compressed.
  Semantic classes:
      0 = background (floor, walls, target ghost)
      1 = cube
      2 = robot arm (UR5e + Robotiq gripper)
      3 = shadow (only with --shadow-class; extracted by rendering each
          frame twice, with and without shadows, and marking background
          pixels that brighten)
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
import numpy as np
import tqdm

from cube_seg_utils import add_shadow_class, build_geom_to_label, seg_from_render
from stable_worldmodel.envs.ogbench.cube_env import CubeEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in-path', required=True)
    p.add_argument('--out-path', required=True)
    p.add_argument('--size', type=int, default=256,
                   help='render resolution (default: 256)')
    p.add_argument('--no-shadow', action='store_true',
                   help='disable shadows when rendering pixels')
    p.add_argument('--shadow-class', action='store_true',
                   help='add shadow as segmentation class 3')
    p.add_argument('--shadow-tau', type=float, default=4.0,
                   help='luminance threshold for the shadow mask')
    p.add_argument('--max-frames', type=int, default=None,
                   help='only process the first N frames (for smoke tests)')
    p.add_argument('--batch', type=int, default=500,
                   help='frames buffered before flushing to disk')
    p.add_argument('--gzip-level', type=int, default=6,
                   help='gzip level for segmentation')
    args = p.parse_args()

    if args.shadow_class and args.no_shadow:
        p.error('--shadow-class and --no-shadow are contradictory')
    in_path = os.path.abspath(os.path.expanduser(args.in_path))
    out_path = os.path.abspath(os.path.expanduser(args.out_path))
    if out_path == in_path or os.path.exists(out_path):
        p.error(f'refusing to overwrite existing file: {out_path}')

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
    saved_castshadow = env._model.light_castshadow.copy()
    if args.no_shadow:
        env._model.light_castshadow[:] = 0
    lut = build_geom_to_label(env._model)

    chunk_px = (min(args.batch, 64), h, w, 3)
    chunk_seg = (min(args.batch, 64), h, w)

    with h5py.File(in_path, 'r') as fin, h5py.File(out_path, 'w') as fout:
        n_src = fin['qpos'].shape[0]
        n = n_src if args.max_frames is None else min(args.max_frames, n_src)

        ep_offset = fin['ep_offset'][:]
        ep_len = fin['ep_len'][:]
        if n < n_src:
            m = int(np.searchsorted(ep_offset, n, side='left'))
            ep_offset = ep_offset[:m]
            ep_len = ep_len[:m].copy()
            ep_len[-1] = n - ep_offset[-1]
        fout.create_dataset('ep_offset', data=ep_offset)
        fout.create_dataset('ep_len', data=ep_len)

        # Copy all other datasets except `pixels`/`segmentation` (rerendered),
        # truncating frame-indexed ones to n.
        for name in fin.keys():
            if name in ('pixels', 'segmentation', 'ep_offset', 'ep_len'):
                continue
            ds = fin[name]
            if ds.shape and ds.shape[0] == n_src:
                fout.create_dataset(name, data=ds[:n])
            else:
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
            mask = seg_from_render(raw, lut)
            rgb = env.render()
            if args.shadow_class:
                env._model.light_castshadow[:] = 0
                rgb_ns = env.render()
                env._model.light_castshadow[:] = saved_castshadow
                mask = add_shadow_class(mask, rgb, rgb_ns, tau=args.shadow_tau)

            px_buf[buf_idx] = rgb
            seg_buf[buf_idx] = mask

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
