"""Inspect PushTMulti collected datasets for compositionality invariants.

Loads one or more dataset H5 files and computes:

  1. Schema invariance — same key set across all datasets.
  2. Identity↔label stability — B's seg label is 3 in every dataset
     where B is enabled.
  3. Enabled flag consistency — `enabled.<oid>` matches expected
     per-dataset enabled set.
  4. NaN sentinel for disabled objects — `pose.<oid>` is NaN exactly
     when `enabled.<oid>` is False.
  5. Per-object pose coverage — workspace exploration histograms.
  6. Pairwise proximity-based contact frequency — for every dataset and
     every ordered (i, j) pair of enabled objects, fraction of frames
     where bodies are within (r_i + r_j + epsilon) of each other.
  7. **Critical RQ2 invariant**: A↔C contact frequency is ZERO in
     AB and BC datasets (since C and A respectively are disabled), and
     non-zero in ABC.

Run:
    python scripts/data/inspect_pusht_multi.py \\
        ~/.stable_worldmodel/datasets/pusht_multi_AB.h5 \\
        ~/.stable_worldmodel/datasets/pusht_multi_BC.h5 \\
        ~/.stable_worldmodel/datasets/pusht_multi_ABC.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 — needed to read compressed segmentation
import numpy as np

from stable_worldmodel.envs.pusht_multi import (
    LABEL_AGENT,
    LABEL_BG,
    OBJECT_LIBRARY,
)


IDS = list(OBJECT_LIBRARY)  # canonical order: A, B, C, D, E, F


def _enabled_set_from_h5(f: h5py.File) -> set[str]:
    """Read which identities were enabled at recording time."""
    enabled = set()
    for oid in IDS:
        if f[f'enabled.{oid}'][0]:
            enabled.add(oid)
    return enabled


def _summary_table(rows: list[list[str]], header: list[str]) -> str:
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    fmt = ' | '.join(f'{{:<{w}}}' for w in widths)
    sep = '-+-'.join('-' * w for w in widths)
    return '\n'.join([fmt.format(*header), sep] + [fmt.format(*r) for r in rows])


def inspect_one(path: Path) -> dict:
    """Per-dataset stats. Returns a dict the summary can format."""
    with h5py.File(path, 'r') as f:
        n_frames = int(f['ep_len'][:].sum())
        n_eps = int(f['ep_len'].shape[0])
        keys = set(f.keys())
        enabled = _enabled_set_from_h5(f)

        # 1. Sample some frames for label & geometry stats. 2000 frames
        # is plenty — enough for stable proximity statistics, cheap to load.
        sample_n = min(2000, n_frames)
        sample_idx = np.linspace(0, n_frames - 1, sample_n).astype(np.int64)

        seg_sample = f['segmentation'][sample_idx]   # (S, H, W) uint8
        # Build a per-frame label-presence matrix.
        labels_present = {
            lbl: np.array([lbl in np.unique(seg_sample[i]) for i in range(sample_n)])
            for lbl in (LABEL_BG, LABEL_AGENT, *[s.label for s in OBJECT_LIBRARY.values()])
        }

        # 2. Per-object pose stats (full sample).
        poses = {
            oid: f[f'pose.{oid}'][sample_idx, :2] for oid in IDS  # (S, 2)
        }
        goal_poses = {
            oid: f[f'goal_pose.{oid}'][sample_idx, :2] for oid in IDS
        }

        # 3. Pairwise proximity-based contact frequency. Two bodies "in
        # contact" if their center distance is within (r_i + r_j) * 1.1
        # — a 10% tolerance over the rotation-invariant bounding-radius
        # sum. With distinct shapes this is a generous-but-honest proxy
        # for actual collisions (true SAT-collision would be tighter).
        radii = {oid: OBJECT_LIBRARY[oid].bounding_radius for oid in IDS}
        contact_threshold_mult = 1.1
        pair_contact_count = {}
        for i, oid_i in enumerate(IDS):
            for oid_j in IDS[i + 1:]:
                pos_i = poses[oid_i]
                pos_j = poses[oid_j]
                # If either is disabled, all NaN → never count as contact.
                d = np.linalg.norm(pos_i - pos_j, axis=-1)
                threshold = (radii[oid_i] + radii[oid_j]) * contact_threshold_mult
                valid = ~np.isnan(d)
                in_contact = np.where(valid, d < threshold, False)
                pair_contact_count[(oid_i, oid_j)] = int(in_contact.sum())

        # 4. Agent ↔ object contact frequency.
        agent_pos = f['pos_agent'][sample_idx]
        agent_radius = 15.0  # default scale 40 → 0.375*40
        agent_contact = {}
        for oid in IDS:
            pos = poses[oid]
            d = np.linalg.norm(pos - agent_pos, axis=-1)
            valid = ~np.isnan(d)
            threshold = (radii[oid] + agent_radius) * contact_threshold_mult
            in_contact = np.where(valid, d < threshold, False)
            agent_contact[oid] = int(in_contact.sum())

        # 5. n_contacts histogram from the recorded info column.
        n_contacts = f['n_contacts'][:]
        contact_pct = float((n_contacts > 0).mean() * 100)

        return dict(
            path=str(path),
            n_eps=n_eps,
            n_frames=n_frames,
            sample_n=sample_n,
            enabled=enabled,
            keys=keys,
            labels_present={
                lbl: int(labels_present[lbl].sum())
                for lbl in labels_present
            },
            pair_contact_count=pair_contact_count,
            agent_contact=agent_contact,
            contact_frame_pct=contact_pct,
            poses=poses,
            goal_poses=goal_poses,
            radii=radii,
        )


def print_summary(stats: dict[str, dict]) -> None:
    print('\n' + '=' * 76)
    print(' DATASET COMPOSITIONALITY INSPECTION')
    print('=' * 76)

    # ---- 1. Overview ----
    print('\n## 1. Overview\n')
    rows = []
    for name, s in stats.items():
        rows.append([
            name,
            str(s['n_eps']),
            f"{s['n_frames']:,}",
            ','.join(sorted(s['enabled'])),
            f"{s['contact_frame_pct']:.1f}%",
        ])
    print(_summary_table(
        rows, ['dataset', 'episodes', 'frames', 'enabled', 'contact-frames'],
    ))

    # ---- 2. Schema invariance ----
    print('\n## 2. Schema invariance\n')
    keysets = [s['keys'] for s in stats.values()]
    common = set.intersection(*keysets)
    union = set.union(*keysets)
    diff = union - common
    if diff:
        print(f'⚠ Key set differs across datasets: {sorted(diff)}')
    else:
        print(f'✅ All {len(stats)} datasets share the same {len(common)} keys')
        per_obj_keys = sorted(k for k in common if any(k.startswith(p) for p in ('pose.', 'goal_pose.', 'enabled.')))
        print(f'   Per-object keys present: {per_obj_keys}')

    # ---- 3. Enabled-flag consistency + NaN sentinel ----
    print('\n## 3. Enabled flag / NaN sentinel for disabled\n')
    rows = []
    for name, s in stats.items():
        for oid in IDS:
            poses = s['poses'][oid]
            all_nan = bool(np.all(np.isnan(poses)))
            any_nan = bool(np.any(np.isnan(poses)))
            is_enabled = oid in s['enabled']
            ok = (is_enabled and not any_nan) or (not is_enabled and all_nan)
            rows.append([
                name, oid,
                'on' if is_enabled else 'off',
                'all-NaN' if all_nan else ('any-NaN' if any_nan else 'finite'),
                '✅' if ok else '❌',
            ])
    print(_summary_table(rows, ['dataset', 'oid', 'enabled', 'pose', 'ok']))

    # ---- 4. Label stability ----
    print('\n## 4. Identity↔label stability\n')
    rows = []
    for name, s in stats.items():
        for oid in IDS:
            lbl = OBJECT_LIBRARY[oid].label
            present_frames = s['labels_present'][lbl]
            is_enabled = oid in s['enabled']
            ok = (is_enabled and present_frames > 0) or (not is_enabled and present_frames == 0)
            rows.append([
                name, oid, str(lbl),
                'on' if is_enabled else 'off',
                f'{present_frames}/{s["sample_n"]}',
                '✅' if ok else '❌',
            ])
    print(_summary_table(rows, ['dataset', 'oid', 'label', 'enabled', 'seg-frames', 'ok']))

    # ---- 5. Pairwise proximity contacts ----
    print('\n## 5. Pairwise proximity-based contact (frames where '
          'd(i,j) < 1.1·(r_i+r_j))\n')
    rows = []
    pair_keys = [(a, b) for i, a in enumerate(IDS) for b in IDS[i + 1:]]
    header = ['dataset'] + [f'{a}{b}' for a, b in pair_keys]
    for name, s in stats.items():
        cells = [name]
        for pair in pair_keys:
            c = s['pair_contact_count'][pair]
            pct = c / s['sample_n'] * 100
            if c == 0:
                cells.append('0')
            else:
                cells.append(f'{c} ({pct:.1f}%)')
        rows.append(cells)
    print(_summary_table(rows, header))

    # ---- 6. Agent contacts ----
    print('\n## 6. Agent ↔ object proximity contact\n')
    rows = []
    for name, s in stats.items():
        cells = [name]
        for oid in IDS:
            c = s['agent_contact'][oid]
            if c == 0:
                cells.append('0')
            else:
                pct = c / s['sample_n'] * 100
                cells.append(f'{c} ({pct:.1f}%)')
        rows.append(cells)
    print(_summary_table(rows, ['dataset'] + [f'agent↔{o}' for o in IDS]))

    # ---- 7. CRITICAL: RQ2 compositionality invariant ----
    print('\n## 7. RQ2 invariant: A↔C contact frequency\n')
    print('   AB ∪ BC training datasets must NOT contain A↔C interactions.')
    print('   ABC test dataset MUST contain A↔C interactions.\n')
    rows = []
    for name, s in stats.items():
        c = s['pair_contact_count'].get(('A', 'C'), 0)
        ac_pct = c / s['sample_n'] * 100
        # Expected based on which identities were active during collection,
        # not the dataset's filename — so `pusht_multi_AB_cem` etc. work too.
        enabled = s['enabled']
        expected_zero = not ({'A', 'C'} <= enabled)
        ok = (expected_zero and c == 0) or (not expected_zero and c > 0)
        rows.append([
            name,
            str(c),
            f'{ac_pct:.2f}%',
            'must be 0' if expected_zero else 'should be > 0',
            '✅' if ok else '❌',
        ])
    print(_summary_table(rows, ['dataset', 'A↔C frames', 'pct', 'expected', 'ok']))

    # ---- 9. Out-of-bounds detection ----
    print('\n## 9. Out-of-bounds object positions (workspace = [0, 512])\n')
    print('   Walls at 5–506. Object centers should stay roughly inside.')
    print('   Significant excursions indicate physics tunneling or')
    print('   high-impulse pushes the solver can\'t resolve.\n')
    rows = []
    for name, s in stats.items():
        for oid in IDS:
            if oid not in s['enabled']:
                continue
            poses = s['poses'][oid]
            x_min, x_max = float(np.nanmin(poses[:, 0])), float(np.nanmax(poses[:, 0]))
            y_min, y_max = float(np.nanmin(poses[:, 1])), float(np.nanmax(poses[:, 1]))
            ws_lo, ws_hi = -10, 522  # small tolerance for objects touching walls
            n_oob = int(np.sum(
                (poses[:, 0] < ws_lo) | (poses[:, 0] > ws_hi) |
                (poses[:, 1] < ws_lo) | (poses[:, 1] > ws_hi)
            ))
            severe = (x_min < -50) or (x_max > 560) or (y_min < -50) or (y_max > 560)
            rows.append([
                name, oid,
                f'[{x_min:.0f}, {x_max:.0f}]',
                f'[{y_min:.0f}, {y_max:.0f}]',
                f'{n_oob}/{s["sample_n"]}',
                '🚨' if severe else ('⚠' if n_oob else '✅'),
            ])
    print(_summary_table(rows, ['dataset', 'oid', 'x-range', 'y-range', 'oob-frames', 'flag']))

    # ---- 8. Pose coverage (workspace exploration) ----
    print('\n## 8. Per-object pose coverage (x range, y range, sampled finite)\n')
    rows = []
    for name, s in stats.items():
        for oid in IDS:
            if oid not in s['enabled']:
                continue
            poses = s['poses'][oid]
            x_min, x_max = float(np.nanmin(poses[:, 0])), float(np.nanmax(poses[:, 0]))
            y_min, y_max = float(np.nanmin(poses[:, 1])), float(np.nanmax(poses[:, 1]))
            rows.append([
                name, oid,
                f'[{x_min:.0f}, {x_max:.0f}]',
                f'[{y_min:.0f}, {y_max:.0f}]',
            ])
    print(_summary_table(rows, ['dataset', 'oid', 'x-range', 'y-range']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+', type=Path,
                    help='Dataset H5 files to inspect (in order).')
    args = ap.parse_args()

    stats = {}
    for p in args.paths:
        # Name shorthand: strip 'pusht_multi_' prefix and '.h5' suffix.
        name = p.stem.replace('pusht_multi_', '')
        print(f'Loading {p.name} ...', flush=True)
        stats[name] = inspect_one(p)

    print_summary(stats)


if __name__ == '__main__':
    main()
