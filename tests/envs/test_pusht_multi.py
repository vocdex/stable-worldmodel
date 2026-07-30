"""Smoke tests for PushTMulti.

Covers the invariants the compositional-generalization experiment depends
on:

1. Stable identity→label mapping: B has the canonical label 3 whether the
   active subset is {A,B}, {B,C}, or {A,B,C}.
2. Disabled objects produce no segmentation pixels.
3. Per-object `pose.<oid>` / `goal_pose.<oid>` keys are present in `info`
   for *every* identity in `objects=`, regardless of which are enabled —
   so the H5 schema is identical across subset datasets.
4. Single-object configuration produces sensible obs / step output (the
   parity check vs. swm/PushT-v1, by mirrored interface — we don't
   compare pixel-equal because PushTMulti changes damping and goal
   rendering).
"""

import gymnasium as gym
import numpy as np
import pytest

import stable_worldmodel  # noqa: F401  (registers swm/PushTMulti-v1)
from stable_worldmodel.envs.pusht_multi import (
    LABEL_AGENT,
    LABEL_BG,
    OBJECT_LIBRARY,
    PushTMulti,
)


# A reusable env factory: passes `objects=` so the variation space sizes
# match across subset settings (which is the schema-stability test).
def _make_env(enabled, objects=('A', 'B', 'C')):
    return PushTMulti(
        objects=objects, enabled_objects=enabled, resolution=96,
    )


def _step_once(env, seed=0):
    obs, info = env.reset(seed=seed)
    obs, rew, term, trunc, info = env.step(env.action_space.sample())
    return obs, info


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_env_is_registered():
    env = gym.make('swm/PushTMulti-v1')
    try:
        obs, info = env.reset(seed=0)
        assert 'segmentation' in info
        assert 'pose.A' in info and 'pose.B' in info and 'pose.C' in info
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Label stability (RQ2 invariant)
# ---------------------------------------------------------------------------


def test_label_stability_across_subsets():
    """B has label 3 (OBJECT_LIBRARY['B'].label) regardless of subset."""
    expected_B = OBJECT_LIBRARY['B'].label
    assert expected_B == 3, 'OBJECT_LIBRARY changed; update the test'

    for subset in [('A', 'B'), ('B', 'C'), ('A', 'B', 'C')]:
        env = _make_env(enabled=subset)
        try:
            _, info = env.reset(seed=42)
            seg = info['segmentation']
            assert expected_B in np.unique(seg), (
                f'B label missing from segmentation when enabled={subset}: '
                f'unique={np.unique(seg)}'
            )
        finally:
            env.close()


def test_disabled_objects_have_no_seg_pixels():
    env = _make_env(enabled=('A', 'B'))
    try:
        _, info = env.reset(seed=7)
        seg_labels = set(int(v) for v in np.unique(info['segmentation']))
        # 'C' is disabled — its label must not appear.
        assert OBJECT_LIBRARY['C'].label not in seg_labels
        # 'A' and 'B' should appear, alongside agent + bg.
        assert OBJECT_LIBRARY['A'].label in seg_labels
        assert OBJECT_LIBRARY['B'].label in seg_labels
        assert LABEL_AGENT in seg_labels
        assert LABEL_BG in seg_labels
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Info schema invariance (RQ1/RQ2 dataloader invariant)
# ---------------------------------------------------------------------------


def test_info_schema_identical_across_subsets():
    """All info keys are present for every identity in `objects=`,
    independent of which subset is enabled. Disabled object slots get
    NaN so downstream HDF5 schemas remain identical across splits.
    """
    objects = ('A', 'B', 'C')
    keysets = []
    for subset in [('A', 'B'), ('B', 'C'), objects]:
        env = _make_env(enabled=subset, objects=objects)
        try:
            _, info = env.reset(seed=11)
            for oid in objects:
                assert f'pose.{oid}' in info
                assert f'goal_pose.{oid}' in info
                assert f'enabled.{oid}' in info
            # Disabled objects have NaN poses; enabled ones don't.
            for oid in objects:
                pose = info[f'pose.{oid}']
                if oid in subset:
                    assert not np.any(np.isnan(pose)), (
                        f'enabled {oid} pose unexpectedly contains NaN'
                    )
                else:
                    assert np.all(np.isnan(pose)), (
                        f'disabled {oid} pose should be NaN, got {pose}'
                    )
            keysets.append(set(info.keys()))
        finally:
            env.close()
    # All three info dicts must have the exact same set of keys.
    assert keysets[0] == keysets[1] == keysets[2], (
        f'info keys diverge across subsets: '
        f'{keysets[0] ^ keysets[1]} / {keysets[1] ^ keysets[2]}'
    )


# ---------------------------------------------------------------------------
# Single-object parity (sanity check; not pixel-equal to PushT)
# ---------------------------------------------------------------------------


def test_single_object_runs_and_steps():
    env = _make_env(enabled=('A',))
    try:
        obs, info = env.reset(seed=0)
        # state layout: 2 + 3*N + 2 where N = len(objects) = 3
        assert obs['state'].shape == (2 + 3 * 3 + 2,)
        # Slot 0 = A is enabled → no NaN for its triple
        assert not np.any(np.isnan(obs['state'][2:5]))
        # Slots B (5:8) and C (8:11) disabled → NaN
        assert np.all(np.isnan(obs['state'][5:8]))
        assert np.all(np.isnan(obs['state'][8:11]))
        # Step doesn't raise and reward is finite negative (or zero).
        action = np.array([0.1, -0.1], dtype=np.float32)
        obs, rew, term, trunc, info = env.step(action)
        assert np.isfinite(rew) and rew <= 0
    finally:
        env.close()


def test_goal_segmentation_present_and_aligned():
    env = _make_env(enabled=('A', 'B'))
    try:
        _, info = env.reset(seed=3)
        gseg = info['goal_segmentation']
        seg = info['segmentation']
        # Same shape.
        assert gseg.shape == seg.shape
        # Goal segmentation contains the enabled object labels.
        ulabels = set(int(v) for v in np.unique(gseg))
        assert OBJECT_LIBRARY['A'].label in ulabels
        assert OBJECT_LIBRARY['B'].label in ulabels
        # Disabled identity must not appear at the goal either.
        assert OBJECT_LIBRARY['C'].label not in ulabels
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Non-overlap rejection sampling (start + goal)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('seed', [0, 1, 2, 3, 4])
def test_no_overlap_in_starts_or_goals(seed):
    """Sampled start/goal positions must respect each pair's
    sum-of-bounding-radii separation (with a small slack to account for
    the `max_iters` early-exit and floating-point noise)."""
    objects = ('A', 'B', 'C')
    env = _make_env(enabled=objects)
    try:
        _, info = env.reset(seed=seed)
        radii = np.array(
            [OBJECT_LIBRARY[o].bounding_radius for o in objects],
            dtype=np.float64,
        )
        min_sep = radii[:, None] + radii[None, :]
        np.fill_diagonal(min_sep, 0.0)

        for arr, name in (
            (np.stack([info[f'pose.{o}'][:2] for o in objects]), 'start'),
            (np.stack([info[f'goal_pose.{o}'][:2] for o in objects]), 'goal'),
        ):
            d = np.linalg.norm(arr[:, None] - arr[None, :], axis=-1)
            np.fill_diagonal(d, np.inf)
            slack = d - min_sep
            # Allow a small tolerance: rejection sampler aims for
            # +8 px margin, accept anything within -2 px of zero (in
            # case max_iters was exhausted; physics will resolve).
            assert slack.min() >= -2, (
                f'{name} bodies overlap at seed={seed}: '
                f'min_slack={slack.min():.1f} px'
            )
    finally:
        env.close()
