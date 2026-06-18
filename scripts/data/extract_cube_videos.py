"""Extract N random episodes from a cube expert H5 as MP4s.

Each frame shows: RGB | block0 pos trace overlay | success indicator in corner.

Usage:
    python scripts/data/extract_cube_videos.py \
        --dataset ~/.stable_worldmodel/datasets/ogbench/cube_double_expert.h5 \
        --out-dir visualizations/cube_double_episodes \
        --episodes 100 --fps 20 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import imageio
import numpy as np


def extract(dataset_path: Path, out_dir: Path, n_episodes: int, fps: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    with h5py.File(dataset_path, 'r') as f:
        ep_len = f['ep_len'][:]
        ep_offset = f['ep_offset'][:]
        n_total = len(ep_len)
        picked = rng.choice(n_total, size=min(n_episodes, n_total), replace=False)
        picked.sort()

        success_col = f['success'] if 'success' in f else None

        for i, ep_i in enumerate(picked):
            start = int(ep_offset[ep_i])
            length = int(ep_len[ep_i])
            end = start + length

            pixels = f['pixels'][start:end]  # (T, H, W, 3)
            ep_success = bool(success_col[end - 1]) if success_col is not None else None

            frames = []
            for t, frame in enumerate(pixels):
                img = frame.copy()
                # success indicator: green/red border on last frame
                if t == length - 1 and ep_success is not None:
                    color = (0, 220, 0) if ep_success else (220, 0, 0)
                    img[:6, :] = color
                    img[-6:, :] = color
                    img[:, :6] = color
                    img[:, -6:] = color
                frames.append(img)

            out_path = out_dir / f'ep{ep_i:04d}.mp4'
            writer = imageio.get_writer(str(out_path), fps=fps, codec='libx264',
                                        output_params=['-crf', '23'])
            for fr in frames:
                writer.append_data(fr)
            writer.close()

            status = ('✓' if ep_success else '✗') if ep_success is not None else '?'
            print(f'[{i+1:3d}/{len(picked)}] ep{ep_i:04d}  {length:4d}f  {status}  → {out_path.name}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, default=Path('visualizations/cube_double_episodes'))
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--fps', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    extract(args.dataset, args.out_dir, args.episodes, args.fps, args.seed)


if __name__ == '__main__':
    main()
