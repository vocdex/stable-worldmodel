"""Extract sample episodes from PushTMulti H5 datasets to MP4.

Writes `--n-episodes` evenly-spaced episodes per dataset as side-by-side
[live RGB | colorized segmentation | goal RGB] MP4s under
`demo_videos/<dataset_stem>/ep_<i>.mp4`. Opens H5 in SWMR mode so it
can read a dataset that's still being written by an active collection.

Usage:
    python scripts/visualization/extract_pusht_multi_episodes.py \\
        ~/.stable_worldmodel/datasets/pusht_multi_AB_cem.h5 \\
        ~/.stable_worldmodel/datasets/pusht_multi_BC_cem.h5 \\
        ~/.stable_worldmodel/datasets/pusht_multi_ABC_cem.h5 \\
        --n-episodes 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import hdf5plugin  # noqa: F401 — needed for compressed seg
import imageio
import numpy as np

from stable_worldmodel.envs.pusht_multi import LABEL_AGENT, LABEL_BG, OBJECT_LIBRARY


def _build_palette() -> np.ndarray:
    """Same palette as scripts/visualization/visualize_pusht_multi.py so
    the per-label hues are consistent across all our renderings."""
    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[LABEL_BG] = (240, 240, 240)
    palette[LABEL_AGENT] = (60, 60, 200)
    for spec in OBJECT_LIBRARY.values():
        hue = (spec.label * 360 / 8) % 360
        c, x = 1.0, 1 - abs(((hue / 60) % 2) - 1)
        if hue < 60:    rgb = (c, x, 0)
        elif hue < 120: rgb = (x, c, 0)
        elif hue < 180: rgb = (0, c, x)
        elif hue < 240: rgb = (0, x, c)
        elif hue < 300: rgb = (x, 0, c)
        else:           rgb = (c, 0, x)
        palette[spec.label] = tuple(int(v * 220) + 35 for v in rgb)
    return palette


PALETTE = _build_palette()


def colorize_seg(seg: np.ndarray) -> np.ndarray:
    return PALETTE[seg]


def extract_dataset(h5_path: Path, out_root: Path, n_episodes: int, fps: int) -> None:
    out_dir = out_root / h5_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'\n=== {h5_path.name} ===')

    with h5py.File(h5_path, 'r', libver='latest', swmr=True) as f:
        if 'ep_len' not in f:
            print(f'  no episodes recorded yet, skipping')
            return
        ep_len = f['ep_len'][:]
        ep_off = f['ep_offset'][:]
        n_total = len(ep_len)
        if n_total == 0:
            print(f'  empty, skipping')
            return

        # Pick episodes evenly across the dataset so the user sees the
        # full range of trajectories, not just the first or last few.
        n_pick = min(n_episodes, n_total)
        idx = np.linspace(0, n_total - 1, n_pick).astype(int)
        print(f'  {n_total} eps in file; extracting {n_pick}: {idx.tolist()}')

        for k, ep_i in enumerate(idx):
            start = int(ep_off[ep_i])
            length = int(ep_len[ep_i])
            end = start + length

            pixels = f['pixels'][start:end]            # (T, H, W, 3) uint8
            seg = f['segmentation'][start:end]         # (T, H', W') uint8
            goal = f['goal'][start:end]                # (T, H, W, 3) uint8 (same every step)

            h_render = pixels.shape[1]
            seg_rgb = colorize_seg(seg)
            if seg_rgb.shape[1] != h_render:
                seg_rgb = np.stack([
                    cv2.resize(s, (h_render, h_render), interpolation=cv2.INTER_NEAREST)
                    for s in seg_rgb
                ])
            if goal.shape[1] != h_render:
                goal = np.stack([
                    cv2.resize(g, (h_render, h_render)) for g in goal
                ])

            frames = np.concatenate([pixels, seg_rgb, goal], axis=2)

            out_path = out_dir / f'ep_{k:02d}.mp4'
            writer = imageio.get_writer(out_path, fps=fps, codec='libx264')
            try:
                for frame in frames:
                    writer.append_data(frame)
            finally:
                writer.close()
            print(f'  → {out_path.relative_to(out_root.parent)} ({length} frames)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+', type=Path,
                    help='Dataset H5 files (one or more).')
    ap.add_argument('--out-dir', type=Path, default=Path('demo_videos'),
                    help='Root output directory (default: demo_videos/).')
    ap.add_argument('--n-episodes', type=int, default=10,
                    help='Episodes per dataset (default: 10).')
    ap.add_argument('--fps', type=int, default=15,
                    help='Output video framerate.')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for p in args.paths:
        extract_dataset(p, args.out_dir, args.n_episodes, args.fps)


if __name__ == '__main__':
    main()
