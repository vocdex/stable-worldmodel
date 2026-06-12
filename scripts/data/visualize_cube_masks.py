"""Visualize OGBench Cube episodes with their segmentation masks.

Each output MP4 stacks `[RGB | colorized segmentation | overlay]`
horizontally so mask quality (including the optional shadow class) can be
eyeballed with one play.  For the first picked episode every 20th frame is
also written as PNG.  Per-episode cube motion stats are printed so motion
variety is checkable numerically.

Run:
    python scripts/data/visualize_cube_masks.py \\
        --h5-path /tmp/cube_indep_smoke.h5 --episodes 3
"""
import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 — required for the pixels dataset codec
import imageio
import numpy as np

from cube_seg_utils import LABEL_ARM, LABEL_BG, LABEL_CUBE, LABEL_SHADOW

PALETTE = np.zeros((256, 3), dtype=np.uint8)
PALETTE[LABEL_BG] = (160, 160, 160)
PALETTE[LABEL_CUBE] = (220, 40, 40)
PALETTE[LABEL_ARM] = (40, 80, 220)
PALETTE[LABEL_SHADOW] = (230, 210, 40)

CUBE_QPOS_START = 14


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h5-path', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--out-dir', type=Path, default=Path('visualizations/cube_masks'))
    p.add_argument('--fps', type=int, default=20)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = args.h5_path.stem

    with h5py.File(args.h5_path, 'r') as f:
        ep_len = f['ep_len'][:]
        ep_offset = f['ep_offset'][:]
        n_total = len(ep_len)
        if n_total < args.episodes:
            picked = list(range(n_total))
        else:
            picked = np.linspace(0, n_total - 1, args.episodes).astype(int).tolist()

        for k, ep_i in enumerate(picked):
            start = int(ep_offset[ep_i])
            length = int(ep_len[ep_i])
            end = start + length
            pixels = f['pixels'][start:end]
            seg = f['segmentation'][start:end]

            seg_rgb = PALETTE[seg]
            overlay = (0.55 * pixels + 0.45 * seg_rgb).astype(np.uint8)
            bg = seg == LABEL_BG
            overlay[bg] = pixels[bg]

            frames = np.concatenate([pixels, seg_rgb, overlay], axis=2)
            out_path = args.out_dir / f'{name}_ep{ep_i:04d}.mp4'
            writer = imageio.get_writer(out_path, fps=args.fps, codec='libx264')
            for fr in frames:
                writer.append_data(fr)
            writer.close()

            if k == 0:
                for t in range(0, length, 20):
                    imageio.imwrite(
                        args.out_dir / f'{name}_ep{ep_i:04d}_t{t:04d}.png', frames[t])

            cube_xy = f['qpos'][start:end, CUBE_QPOS_START:CUBE_QPOS_START + 2]
            speed = np.linalg.norm(np.diff(cube_xy, axis=0), axis=1)
            shadow_frac = (seg == LABEL_SHADOW).mean()
            print(f'ep {ep_i:4d}: {length} frames | idle {100 * (speed < 1e-5).mean():.0f}% '
                  f'| max speed {speed.max():.4f} m/frame '
                  f'| shadow {100 * shadow_frac:.2f}% of pixels '
                  f'| labels {sorted(np.unique(seg).tolist())} '
                  f'-> {out_path.name}')


if __name__ == '__main__':
    main()
