"""Record rollouts of PushTMulti across compositional splits.

For each split (e.g. {A,B}, {B,C}, {A,B,C}), drive `MultiObjectWeakPolicy`
for a few episodes and save an MP4 that triples up:

    | RGB live | seg (colorized) | RGB goal |

Useful as a sanity check that:
  - enabled_objects subset selection actually controls which bodies appear,
  - segmentation labels are stable across subsets,
  - goal-pose rendering is sensible.

Usage:
    python scripts/visualization/visualize_pusht_multi.py \
        --out-dir demo_videos/pusht_multi --steps 60 --episodes 2

The output directory is created if it doesn't exist.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import imageio
import numpy as np

import stable_worldmodel as swm
from stable_worldmodel.envs.pusht_multi import (
    OBJECT_LIBRARY,
    LABEL_AGENT,
    LABEL_BG,
    MultiObjectWeakPolicy,
)


# A fixed RGB palette indexed by segmentation label, so any object's color
# is identical across all videos (== reading the figures becomes easy).
def _build_palette() -> np.ndarray:
    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[LABEL_BG] = (240, 240, 240)
    palette[LABEL_AGENT] = (60, 60, 200)
    for spec in OBJECT_LIBRARY.values():
        # Distinct, saturated colors per identity. Hash the label for variety
        # but keep deterministic.
        rng = np.random.default_rng(spec.label * 1337)
        hue = (spec.label * 360 / 8) % 360
        # Convert HSV -> RGB the simple way.
        c = 1.0
        x = 1 - abs(((hue / 60) % 2) - 1)
        if hue < 60:    rgb = (c, x, 0)
        elif hue < 120: rgb = (x, c, 0)
        elif hue < 180: rgb = (0, c, x)
        elif hue < 240: rgb = (0, x, c)
        elif hue < 300: rgb = (x, 0, c)
        else:           rgb = (c, 0, x)
        palette[spec.label] = tuple(int(v * 220) + 35 for v in rgb)
    return palette


PALETTE = _build_palette()


def colorize_segmentation(seg: np.ndarray) -> np.ndarray:
    return PALETTE[seg]


def _build_options(objects: tuple[str, ...], enabled: tuple[str, ...]) -> dict:
    return {
        'variation': [
            'agent.start_position',
            *[f'obj.{oid}.start_position' for oid in objects],
            *[f'obj.{oid}.angle' for oid in objects],
            *[f'obj.{oid}.goal_position' for oid in objects],
            *[f'obj.{oid}.goal_angle' for oid in objects],
            *[f'obj.{oid}.enabled' for oid in objects],
        ],
        'variation_values': {
            f'obj.{oid}.enabled': int(oid in set(enabled)) for oid in objects
        },
    }


def _label_strip(seg: np.ndarray, enabled: tuple[str, ...]) -> str:
    present = sorted(int(v) for v in np.unique(seg))
    name_by_label = {spec.label: oid for oid, spec in OBJECT_LIBRARY.items()}
    parts = []
    for v in present:
        if v == LABEL_BG: parts.append('bg')
        elif v == LABEL_AGENT: parts.append('agent')
        else: parts.append(name_by_label.get(v, f'?{v}'))
    return ', '.join(parts)


def rollout_split(
    objects: tuple[str, ...],
    enabled: tuple[str, ...],
    out_path: Path,
    steps: int,
    episodes: int,
    image_size: int,
    seed: int,
) -> None:
    print(f'\n[{enabled}]  -> {out_path}')
    world = swm.World(
        'swm/PushTMulti-v1',
        num_envs=1,
        max_episode_steps=steps,
        image_shape=(image_size, image_size),
        objects=list(objects),
        enabled_objects=list(enabled),
        verbose=0,
    )
    world.set_policy(
        MultiObjectWeakPolicy(
            dist_constraint=100, switch_every=10, p_wedge=0.4, seed=seed
        )
    )
    opts = _build_options(objects, enabled)

    writer = imageio.get_writer(out_path, fps=10, codec='libx264')
    try:
        for ep in range(episodes):
            world.reset(seed=seed + ep * 7, options=opts)
            import cv2
            for t in range(steps):
                # Wrapped infos have shape (n_envs, history, ...) — pick env 0
                # and the most recent history frame.
                def _take(key):
                    v = world.infos[key][0]
                    while v.ndim >= 3 and v.shape[0] == 1 and key != 'pixels' and key != 'goal':
                        # 2D maps come in as (history, H, W); pick the last.
                        pass
                    return v[-1] if v.ndim in (3, 4) and v.shape[0] != image_size else v

                pixels = world.infos['pixels'][0, -1]
                seg = world.infos['segmentation'][0, -1]
                goal = world.infos['goal'][0, -1]

                if t == 0:
                    seg_strip = _label_strip(seg, enabled)
                    print(f'  episode {ep}: seg labels present at t=0 = [{seg_strip}]')

                seg_rgb = colorize_segmentation(seg)
                h = pixels.shape[0]
                if seg_rgb.shape[0] != h:
                    seg_rgb = cv2.resize(seg_rgb, (h, h), interpolation=cv2.INTER_NEAREST)
                if goal.shape[0] != h:
                    goal = cv2.resize(goal, (h, h))
                frame = np.concatenate([pixels, seg_rgb, goal], axis=1)
                writer.append_data(frame)
                world.step()
                if np.any(world.terminateds) or np.any(world.truncateds):
                    break
    finally:
        writer.close()
        world.close()
    print(f'  wrote {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='demo_videos/pusht_multi')
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--episodes', type=int, default=2)
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    objects = ('A', 'B', 'C')
    splits = {
        'AB': ('A', 'B'),
        'BC': ('B', 'C'),
        'ABC': ('A', 'B', 'C'),
    }
    # Also throw in a high-clutter sanity rollout with all 6 identities so we
    # can eyeball the seg renderer's distinctness on a hard scene.
    objects_full = tuple(OBJECT_LIBRARY.keys())
    splits_extra = {
        'ABCDEF': objects_full,
    }

    for name, enabled in splits.items():
        rollout_split(
            objects=objects, enabled=enabled,
            out_path=out_dir / f'pusht_multi_{name}.mp4',
            steps=args.steps, episodes=args.episodes,
            image_size=args.image_size, seed=args.seed,
        )

    for name, enabled in splits_extra.items():
        rollout_split(
            objects=objects_full, enabled=enabled,
            out_path=out_dir / f'pusht_multi_{name}.mp4',
            steps=args.steps, episodes=args.episodes,
            image_size=args.image_size, seed=args.seed,
        )


if __name__ == '__main__':
    main()
