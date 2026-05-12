"""Extract a handful of episodes from PushTMulti H5 datasets as MP4.

Each output MP4 stacks `[RGB live | colorized segmentation | goal frame]`
horizontally so reviewers can eyeball the data with one play.

Run:
    python scripts/data/extract_pusht_multi_videos.py \\
        --dataset ~/.stable_worldmodel/datasets/pusht_multi_AB.h5 \\
        --out-dir demo_videos/pusht_multi_episodes \\
        --episodes 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import hdf5plugin  # noqa: F401 — required for the seg dataset codec
import imageio
import numpy as np

from stable_worldmodel.envs.pusht_multi import (
    LABEL_AGENT, LABEL_BG, OBJECT_LIBRARY,
)


def _build_palette() -> np.ndarray:
    """Distinct flat RGBs per segmentation label. Same scheme as the
    rollout visualizer so videos look consistent across scripts.
    """
    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[LABEL_BG] = (240, 240, 240)
    palette[LABEL_AGENT] = (60, 60, 200)
    for spec in OBJECT_LIBRARY.values():
        h = (spec.label * 360 / 8) % 360
        c = 1.0
        x = 1 - abs(((h / 60) % 2) - 1)
        if h < 60:    rgb = (c, x, 0)
        elif h < 120: rgb = (x, c, 0)
        elif h < 180: rgb = (0, c, x)
        elif h < 240: rgb = (0, x, c)
        elif h < 300: rgb = (x, 0, c)
        else:         rgb = (c, 0, x)
        palette[spec.label] = tuple(int(v * 220) + 35 for v in rgb)
    return palette


PALETTE = _build_palette()


def extract(dataset_path: Path, out_dir: Path, n_episodes: int, fps: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = dataset_path.stem
    print(f'\n[{name}] reading {dataset_path}')

    with h5py.File(dataset_path, 'r') as f:
        ep_len = f['ep_len'][:]
        ep_offset = f['ep_offset'][:]
        n_total = len(ep_len)
        # Pick episodes evenly spread through the file (first/last/middle, etc.)
        # so the sample isn't biased toward the start.
        if n_total < n_episodes:
            picked = list(range(n_total))
        else:
            picked = np.linspace(0, n_total - 1, n_episodes).astype(int).tolist()

        for ep_i in picked:
            start = int(ep_offset[ep_i])
            length = int(ep_len[ep_i])
            end = start + length
            pixels = f['pixels'][start:end]              # (T, H, W, 3) uint8
            seg = f['segmentation'][start:end]           # (T, Hs, Ws) uint8
            goal = f['goal'][start:end]                  # (T, H, W, 3) uint8 (same every step)

            H, W = pixels.shape[1], pixels.shape[2]
            # Resize seg & goal to match pixels' resolution.
            seg_rgb = PALETTE[seg]                       # (T, Hs, Ws, 3)
            if seg_rgb.shape[1] != H:
                seg_rgb = np.stack([
                    cv2.resize(s, (W, H), interpolation=cv2.INTER_NEAREST)
                    for s in seg_rgb
                ])
            if goal.shape[1] != H:
                goal = np.stack([cv2.resize(g, (W, H)) for g in goal])

            frames = np.concatenate([pixels, seg_rgb, goal], axis=2)  # along W
            out_path = out_dir / f'{name}_ep{ep_i:04d}.mp4'
            writer = imageio.get_writer(out_path, fps=fps, codec='libx264')
            for fr in frames:
                writer.append_data(fr)
            writer.close()
            print(f'  wrote {out_path.name}  ({length} frames)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, required=True,
                    help='Dataset H5 file to extract from')
    ap.add_argument('--out-dir', type=Path, default=Path('demo_videos/pusht_multi_episodes'))
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--fps', type=int, default=10)
    args = ap.parse_args()
    extract(args.dataset, args.out_dir, args.episodes, args.fps)


if __name__ == '__main__':
    main()
