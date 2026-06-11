"""Trust-chain tests for the OOD planning eval (`World.evaluate_from_dataset`).

The eval chain (H5 frames -> variation reset -> callables -> planner inputs)
silently produced wrong goal frames for weeks (the 2026-06 cube bug: under a
variation override the planner targeted a reset-time random task goal instead
of the H5 goal). These tests make every link of the chain an explicit, atomic
invariant:

    T1  goal frame given to the planner == re-render of the H5 goal state
        under the active variation
    T2  start frame given to the planner == re-render of the H5 start state
        under the active variation
    T3  the perturbation is actually visible in the planner's frames
    T4  the env's numerical success target equals the H5 goal state,
        override or not
    T5  without an override, planner frames are byte-identical to the H5
    T6  every variation_space leaf changes the rendered pixels (no-op sweep)
    T7  during the rollout, the frames the policy sees track the live env

plus regression detectors that re-create the missing-goal-callable bug and
assert that both the T1 invariant and the SWM_EVAL_TRUST_CHECKS runtime hook
catch it. Cube variants are marked `slow` (MuJoCo EGL).
"""

import os

os.environ.setdefault('MUJOCO_GL', 'egl')

from copy import deepcopy

import gymnasium as gym
import h5py
import numpy as np
import pytest

import stable_worldmodel as swm
from stable_worldmodel import spaces as swm_spaces
from stable_worldmodel.policy import BasePolicy
from stable_worldmodel.utils import get_in


RES = 128
N_ENVS = 2
EPISODES_IDX = [0, 1]
START_STEPS = [2, 3]
GOAL_OFFSET = 6

BLOCK_RED = {
    'variation': ['block.color'],
    'variation_values': {
        'block.color': np.array([255, 0, 0], dtype=np.uint8)
    },
}

DISTRACTOR_MOVING = {
    'variation': [
        'distractor.color',
        'distractor.scale',
        'distractor.position',
        'distractor.motion',
    ],
    'variation_values': {
        'distractor.color': np.array([255, 0, 255], dtype=np.uint8),
        'distractor.scale': np.float32(25),
        'distractor.position': np.array([200.0, 200.0]),
        'distractor.motion': np.array([40.0, 20.0]),
    },
}

PUSHT_CALLABLES = [
    {'method': '_set_state', 'args': {'state': {'value': 'state'}}},
    {
        'method': '_set_goal_state',
        'args': {'goal_state': {'value': 'goal_state'}},
    },
]

CUBE_RED = {
    'variation': ['cube.color'],
    'variation_values': {
        'cube.color': np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    },
}

CUBE_GOAL_OFFSET = 4
CUBE_START_STEPS = [1, 2]

CUBE_CALLABLES = [
    {
        'method': 'set_state',
        'args': {'qpos': {'value': 'qpos'}, 'qvel': {'value': 'qvel'}},
    },
    {
        'method': 'set_target_pos',
        'args': {
            'cube_id': {'value': 0, 'in_dataset': False},
            'target_pos': {'value': 'goal_privileged_block_0_pos'},
            'target_quat': {'value': 'goal_privileged_block_0_quat'},
        },
    },
    {
        'method': 'render_goal_scene',
        'args': {
            'qpos': {'value': 'goal_qpos'},
            'qvel': {'value': 'goal_qvel'},
        },
    },
]

CUBE_ENV_KWARGS = dict(
    env_type='single',
    ob_type='states',
    multiview=False,
    width=224,
    height=224,
    visualize_info=False,
)


def mad(a, b):
    """Mean absolute difference in 0-255 units."""
    return float(np.mean(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))))


class SpyPolicy(BasePolicy):
    """Records the planner inputs (`pixels`, `goal`) of every call and the
    live env renders at call time; acts with zero actions."""

    def __init__(self):
        super().__init__()
        self.calls = []
        self.live_renders = []

    def get_action(self, info_dict, **kwargs):
        self.calls.append(
            {
                k: np.asarray(v).copy()
                for k, v in info_dict.items()
                if k in ('pixels', 'goal')
            }
        )
        self.live_renders.append(
            np.stack([e.unwrapped.render() for e in self.env.envs])
        )
        return np.zeros(self.env.action_space.shape, dtype=np.float32)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope='module')
def pusht_dataset(tmp_path_factory):
    cache = tmp_path_factory.mktemp('swm_cache_pusht')
    world = swm.World(
        'swm/PushT-v1',
        num_envs=N_ENVS,
        image_shape=(RES, RES),
        max_episode_steps=START_STEPS[-1] + GOAL_OFFSET + 3,
        resolution=RES,
        verbose=0,
    )
    world.set_policy(swm.policy.RandomPolicy(seed=0))
    world.record_dataset('trust_pusht', episodes=2, seed=123, cache_dir=cache)
    world.close()

    dataset = swm.data.HDF5Dataset('trust_pusht', cache_dir=cache)
    h5 = h5py.File(dataset.h5_path, 'r', swmr=True, libver='latest')
    yield dataset, h5
    h5.close()


# function scope: the StackedWrapper freezes its info-key set (incl.
# `variation.*` columns) at the FIRST reset, so a world cannot be reused
# across tests with different variation lists.
@pytest.fixture
def pusht_world():
    world = swm.World(
        'swm/PushT-v1',
        num_envs=N_ENVS,
        image_shape=(RES, RES),
        max_episode_steps=4 * GOAL_OFFSET,
        resolution=RES,
        verbose=0,
    )
    yield world
    world.close()


@pytest.fixture(scope='module')
def cube_dataset(tmp_path_factory):
    cache = tmp_path_factory.mktemp('swm_cache_cube')
    world = swm.World(
        'swm/OGBCube-v0',
        num_envs=N_ENVS,
        image_shape=(224, 224),
        max_episode_steps=CUBE_START_STEPS[-1] + CUBE_GOAL_OFFSET + 2,
        verbose=0,
        terminate_at_goal=False,
        mode='data_collection',
        **CUBE_ENV_KWARGS,
    )
    world.set_policy(swm.policy.RandomPolicy(seed=0))
    world.record_dataset('trust_cube', episodes=2, seed=7, cache_dir=cache)
    world.close()

    dataset = swm.data.HDF5Dataset('trust_cube', cache_dir=cache)
    h5 = h5py.File(dataset.h5_path, 'r', swmr=True, libver='latest')
    yield dataset, h5
    h5.close()


@pytest.fixture
def cube_world():
    world = swm.World(
        'swm/OGBCube-v0',
        num_envs=N_ENVS,
        image_shape=(224, 224),
        max_episode_steps=4 * CUBE_GOAL_OFFSET,
        verbose=0,
        terminate_at_goal=True,
        **CUBE_ENV_KWARGS,
    )
    yield world
    world.close()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def h5_row(h5, col, ep, step):
    return h5[col][h5['ep_offset'][ep] + step]


def goal_row(start, goal_offset):
    """Dataset row used as the goal by `evaluate_from_dataset`.

    `load_chunk(ep, start, start + goal_offset)` slices with an EXCLUSIVE
    end (dataset.py `_load_slice`), so the goal is the state
    `goal_offset - 1` steps after the start — pinned here on purpose; a
    change in this semantic silently shifts every goal in every eval.
    """
    return start + goal_offset - 1


def run_eval(world, dataset, *, overrides, callables, start_steps, goal_offset, budget=2):
    spy = SpyPolicy()
    world.set_policy(spy)
    results = world.evaluate_from_dataset(
        dataset,
        episodes_idx=EPISODES_IDX,
        start_steps=start_steps,
        goal_offset_steps=goal_offset,
        eval_budget=budget,
        callables=deepcopy(callables),
        save_video=False,
        variation_overrides=deepcopy(overrides),
    )
    return spy, results


def run_pusht(world, dataset, overrides=None, callables=PUSHT_CALLABLES, budget=2):
    return run_eval(
        world,
        dataset,
        overrides=overrides,
        callables=callables,
        start_steps=START_STEPS,
        goal_offset=GOAL_OFFSET,
        budget=budget,
    )


def run_cube(world, dataset, overrides=None, callables=CUBE_CALLABLES, budget=2):
    return run_eval(
        world,
        dataset,
        overrides=overrides,
        callables=callables,
        start_steps=CUBE_START_STEPS,
        goal_offset=CUBE_GOAL_OFFSET,
        budget=budget,
    )


def independent_pusht_frames(h5, overrides):
    """Start/goal frames for each eval window, rendered on a fresh env —
    the ground truth the eval pipeline must reproduce."""
    env = gym.make('swm/PushT-v1', resolution=RES).unwrapped
    options = None if overrides is None else deepcopy(overrides)
    starts, goals = [], []
    for ep, start in zip(EPISODES_IDX, START_STEPS):
        env.reset(seed=0, options=deepcopy(options))
        env._set_state(h5_row(h5, 'state', ep, start))
        env._set_goal_state(h5_row(h5, 'state', ep, goal_row(start, GOAL_OFFSET)))
        goals.append(env._goal.copy())
        starts.append(env.render().copy())
    env.close()
    return np.stack(starts), np.stack(goals)


def independent_cube_frames(h5, overrides):
    env = gym.make('swm/OGBCube-v0', **CUBE_ENV_KWARGS).unwrapped
    options = None if overrides is None else deepcopy(overrides)
    starts, goals = [], []
    for ep, start in zip(EPISODES_IDX, CUBE_START_STEPS):
        env.reset(seed=0, options=deepcopy(options))
        env.set_state(
            h5_row(h5, 'qpos', ep, start), h5_row(h5, 'qvel', ep, start)
        )
        env.render_goal_scene(
            h5_row(h5, 'qpos', ep, goal_row(start, CUBE_GOAL_OFFSET)),
            h5_row(h5, 'qvel', ep, goal_row(start, CUBE_GOAL_OFFSET)),
        )
        goals.append(np.asarray(env._goal).copy())
        starts.append(env.render().copy())
    env.close()
    return np.stack(starts), np.stack(goals)


# --------------------------------------------------------------------------
# T1/T2 — planner frames correspond to the H5 states under the variation
# --------------------------------------------------------------------------


def test_t1_goal_frame_matches_h5_goal_under_override(pusht_world, pusht_dataset):
    dataset, h5 = pusht_dataset
    spy, _ = run_pusht(pusht_world, dataset, overrides=BLOCK_RED)
    _, expected_goals = independent_pusht_frames(h5, BLOCK_RED)
    got = spy.calls[0]['goal'][:, -1]
    assert mad(got, expected_goals) < 2.0


def test_t2_start_frame_matches_h5_start_under_override(pusht_world, pusht_dataset):
    dataset, h5 = pusht_dataset
    spy, _ = run_pusht(pusht_world, dataset, overrides=BLOCK_RED)
    expected_starts, _ = independent_pusht_frames(h5, BLOCK_RED)
    got = spy.calls[0]['pixels'][:, -1]
    assert mad(got, expected_starts) < 2.0


@pytest.mark.parametrize(
    'overrides', [BLOCK_RED, DISTRACTOR_MOVING], ids=['block_red', 'distractor_moving']
)
def test_t1_t2_planner_frames_match_h5_states(pusht_world, pusht_dataset, overrides):
    """T1+T2 for every override cell used in the OOD matrix (extend the
    parametrize list whenever a new cell/variation entry is added — T8)."""
    dataset, h5 = pusht_dataset
    spy, _ = run_pusht(pusht_world, dataset, overrides=overrides)
    expected_starts, expected_goals = independent_pusht_frames(h5, overrides)
    assert mad(spy.calls[0]['goal'][:, -1], expected_goals) < 2.0
    assert mad(spy.calls[0]['pixels'][:, -1], expected_starts) < 2.0


# --------------------------------------------------------------------------
# T3 — the perturbation is visible in the planner's frames
# --------------------------------------------------------------------------


def test_t3_override_visibly_changes_planner_frames(pusht_world, pusht_dataset):
    dataset, _ = pusht_dataset
    spy_clean, _ = run_pusht(pusht_world, dataset, overrides=None)
    spy_pert, _ = run_pusht(pusht_world, dataset, overrides=BLOCK_RED)
    for key in ('pixels', 'goal'):
        d = mad(spy_clean.calls[0][key][:, -1], spy_pert.calls[0][key][:, -1])
        assert d > 0.5, f'{key} frames unchanged under override (diff {d:.3f})'


# --------------------------------------------------------------------------
# T4 — numerical success target equals the H5 goal state
# --------------------------------------------------------------------------


@pytest.mark.parametrize('overrides', [None, BLOCK_RED], ids=['clean', 'override'])
def test_t4_success_target_matches_h5_goal_state(pusht_world, pusht_dataset, overrides):
    dataset, h5 = pusht_dataset
    run_pusht(pusht_world, dataset, overrides=overrides)
    for i, (ep, start) in enumerate(zip(EPISODES_IDX, START_STEPS)):
        expected = h5_row(h5, 'state', ep, goal_row(start, GOAL_OFFSET))
        actual = pusht_world.envs.envs[i].unwrapped.goal_state
        assert np.allclose(np.asarray(actual), expected, atol=1e-6)


# --------------------------------------------------------------------------
# T5 — baseline (no override) planner frames are byte-identical to the H5
# --------------------------------------------------------------------------


def test_t5_baseline_frames_identical_to_h5(pusht_world, pusht_dataset):
    dataset, h5 = pusht_dataset
    spy, _ = run_pusht(pusht_world, dataset, overrides=None)
    for i, (ep, start) in enumerate(zip(EPISODES_IDX, START_STEPS)):
        assert np.array_equal(
            spy.calls[0]['pixels'][i, -1], h5_row(h5, 'pixels', ep, start)
        )
        assert np.array_equal(
            spy.calls[0]['goal'][i, -1],
            h5_row(h5, 'pixels', ep, goal_row(start, GOAL_OFFSET)),
        )


# --------------------------------------------------------------------------
# T6 — every variation leaf changes the rendered pixels (no-op sweep)
# --------------------------------------------------------------------------

# Documented non-visual leaves: changing them is EXPECTED to leave the frame
# untouched. Anything else rendering identically is a silent no-op bug
# (the class fixed in a1437f7: inverted agent-color check, hardcoded T scale).
PUSHT_KNOWN_NOOPS = {
    'agent.velocity': 'start velocity is not visible in a single frame',
    'agent.angle': 'the agent is a circle; rotation is invisible',
    'goal.scale': 'goal marker is drawn from the block shapes; scale unused',
}


def _leaf_paths(space, prefix=''):
    if hasattr(space, 'spaces'):
        for key, sub in space.spaces.items():
            yield from _leaf_paths(sub, f'{prefix}{key}.')
    else:
        yield prefix[:-1]


def _probe_value(space):
    """A value visibly different from the leaf's init value (deterministic).

    37% of the way toward the farthest bound — avoids degenerate extremes
    (e.g. angle 0 -> 2*pi renders identically).
    """
    if isinstance(space, swm_spaces.Discrete):
        return int(
            space.start + (int(space.init_value) - space.start + 1) % space.n
        )
    init = np.asarray(space.init_value, dtype=np.float64)
    high = np.asarray(space.high, dtype=np.float64)
    low = np.asarray(space.low, dtype=np.float64)
    toward = np.where(high - init >= init - low, high, low)
    return (init + 0.37 * (toward - init)).astype(space.dtype)


def _init_value(space):
    if isinstance(space, swm_spaces.Discrete):
        return int(space.init_value)
    return np.asarray(space.init_value, dtype=space.dtype)


def _render_leaf(env, leaf, value):
    env.reset(
        seed=11,
        options={'variation': [leaf], 'variation_values': {leaf: value}},
    )
    return env.render().copy()


def _pusht_leaves():
    env = gym.make('swm/PushT-v1', resolution=64).unwrapped
    leaves = list(_leaf_paths(env.variation_space))
    env.close()
    return leaves


@pytest.mark.parametrize('leaf', _pusht_leaves())
def test_t6_pusht_variation_changes_pixels(leaf):
    env = gym.make('swm/PushT-v1', resolution=RES).unwrapped
    space = get_in(env.variation_space, leaf.split('.'))
    base = _render_leaf(env, leaf, _init_value(space))
    probe = _render_leaf(env, leaf, _probe_value(space))
    env.close()

    d = mad(base, probe)
    if d <= 0.1 and leaf in PUSHT_KNOWN_NOOPS:
        pytest.xfail(f'known non-visual leaf: {PUSHT_KNOWN_NOOPS[leaf]}')
    assert d > 0.1, (
        f'variation {leaf!r} produced identical pixels (mean abs diff '
        f'{d:.4f}) — silent no-op'
    )


# --------------------------------------------------------------------------
# T7 — frames the policy sees track the live (perturbed) env
# --------------------------------------------------------------------------


def test_t7_policy_frames_track_live_env_under_override(pusht_world, pusht_dataset):
    dataset, _ = pusht_dataset
    spy, _ = run_pusht(pusht_world, dataset, overrides=BLOCK_RED, budget=3)
    assert len(spy.calls) >= 2
    for call, live in zip(spy.calls, spy.live_renders):
        assert mad(call['pixels'][:, -1], live) < 2.0


# --------------------------------------------------------------------------
# Regression detectors — the missing-goal-callable bug class (2026-06 cube)
# --------------------------------------------------------------------------


def test_regression_missing_goal_callable_detected(pusht_world, pusht_dataset):
    """Without the goal re-render callable, the planner's goal frame is the
    env's reset-time random goal. The T1 invariant must flag it."""
    dataset, h5 = pusht_dataset
    spy, _ = run_pusht(
        pusht_world, dataset, overrides=BLOCK_RED, callables=[PUSHT_CALLABLES[0]]
    )
    _, expected_goals = independent_pusht_frames(h5, BLOCK_RED)
    got = spy.calls[0]['goal'][:, -1]
    assert mad(got, expected_goals) > 2.0, (
        'goal frame unexpectedly matched the H5 goal even though the '
        'goal-state callable was removed — the regression detector is dead'
    )


def test_runtime_hook_catches_missing_goal_callable(pusht_world, pusht_dataset, monkeypatch):
    monkeypatch.setenv('SWM_EVAL_TRUST_CHECKS', '1')
    dataset, _ = pusht_dataset
    with pytest.raises(RuntimeError, match='[Tt]rust check'):
        run_pusht(
            pusht_world,
            dataset,
            overrides=BLOCK_RED,
            callables=[PUSHT_CALLABLES[0]],
        )


def test_runtime_hook_passes_on_correct_setup(pusht_world, pusht_dataset, monkeypatch):
    monkeypatch.setenv('SWM_EVAL_TRUST_CHECKS', '1')
    dataset, _ = pusht_dataset
    spy, _ = run_pusht(pusht_world, dataset, overrides=BLOCK_RED)
    assert spy.calls  # completed without the hook raising


# --------------------------------------------------------------------------
# Cube (slow: MuJoCo EGL + model recompiles)
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_t1_cube_goal_frame_matches_h5_goal_under_override(cube_world, cube_dataset):
    dataset, h5 = cube_dataset
    spy, _ = run_cube(cube_world, dataset, overrides=CUBE_RED)
    _, expected_goals = independent_cube_frames(h5, CUBE_RED)
    got = spy.calls[0]['goal'][:, -1]
    assert mad(got, expected_goals) < 2.0


@pytest.mark.slow
def test_t2_cube_start_frame_matches_h5_start_under_override(cube_world, cube_dataset):
    dataset, h5 = cube_dataset
    spy, _ = run_cube(cube_world, dataset, overrides=CUBE_RED)
    expected_starts, _ = independent_cube_frames(h5, CUBE_RED)
    got = spy.calls[0]['pixels'][:, -1]
    assert mad(got, expected_starts) < 2.0


@pytest.mark.slow
def test_regression_cube_missing_goal_callable_detected(cube_world, cube_dataset):
    """The exact 2026-06 bug: cube.yaml without render_goal_scene leaves the
    planner targeting a random predefined task goal."""
    dataset, h5 = cube_dataset
    spy, _ = run_cube(
        cube_world, dataset, overrides=CUBE_RED, callables=CUBE_CALLABLES[:2]
    )
    _, expected_goals = independent_cube_frames(h5, CUBE_RED)
    got = spy.calls[0]['goal'][:, -1]
    assert mad(got, expected_goals) > 2.0


@pytest.mark.slow
def test_runtime_hook_catches_cube_missing_goal_callable(cube_world, cube_dataset, monkeypatch):
    monkeypatch.setenv('SWM_EVAL_TRUST_CHECKS', '1')
    dataset, _ = cube_dataset
    with pytest.raises(RuntimeError, match='[Tt]rust check'):
        run_cube(
            cube_world,
            dataset,
            overrides=CUBE_RED,
            callables=CUBE_CALLABLES[:2],
        )


@pytest.mark.slow
def test_runtime_hook_passes_on_correct_cube_setup(cube_world, cube_dataset, monkeypatch):
    monkeypatch.setenv('SWM_EVAL_TRUST_CHECKS', '1')
    dataset, _ = cube_dataset
    spy, _ = run_cube(cube_world, dataset, overrides=CUBE_RED)
    assert spy.calls
