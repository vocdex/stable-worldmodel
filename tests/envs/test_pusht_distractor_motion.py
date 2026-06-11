"""Tests for the PushT moving-distractor variation (`distractor.motion`).

The distractor is purely visual (no collisions). `distractor.motion` is
(amplitude_px, period_steps): the distractor moves on a circle of the given
amplitude around `distractor.position`, advancing one phase step per env
step (cosine phase, so the amplitude is already visible at t=0).
amplitude == 0 (the default) reproduces the legacy static distractor.
"""

import os

os.environ.setdefault('MUJOCO_GL', 'egl')

import gymnasium as gym
import numpy as np
import pytest

import stable_worldmodel  # noqa: F401  (registers envs)


RES = 128

MOVING = {
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


def static_options():
    opts = {
        'variation': list(MOVING['variation']),
        'variation_values': dict(MOVING['variation_values']),
    }
    opts['variation_values']['distractor.motion'] = np.array([0.0, 20.0])
    return opts


def mad(a, b):
    return float(
        np.mean(
            np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32))
        )
    )


def make_env():
    return gym.make('swm/PushT-v1', resolution=RES).unwrapped


def rollout_frames(options, n_steps=5, seed=3):
    env = make_env()
    env.reset(seed=seed, options=options)
    frames = [env.render().copy()]
    for _ in range(n_steps):
        env.step(np.zeros(2, dtype=np.float32))
        frames.append(env.render().copy())
    env.close()
    return frames


def test_distractor_moves_during_rollout():
    frames = rollout_frames(MOVING)
    assert mad(frames[0], frames[3]) > 0.05, (
        'distractor.motion amplitude > 0 but the scene did not change '
        'over 3 zero-action steps'
    )


def test_zero_amplitude_is_static():
    frames = rollout_frames(static_options())
    assert mad(frames[0], frames[-1]) < 0.05, (
        'distractor moved although motion amplitude is 0 (legacy static '
        'cell must be unaffected)'
    )


def test_motion_is_deterministic():
    a = rollout_frames(MOVING, seed=5)
    b = rollout_frames(MOVING, seed=5)
    for fa, fb in zip(a, b):
        assert np.array_equal(fa, fb)


def test_amplitude_visible_at_reset():
    """Cosine phase: the amplitude offsets the distractor already at t=0,
    so the start/goal frames of an eval window reflect the motion cell."""
    moving = rollout_frames(MOVING, n_steps=0)[0]
    static = rollout_frames(static_options(), n_steps=0)[0]
    assert mad(moving, static) > 0.05


def test_motion_only_moves_the_distractor():
    """Zero actions + static physics: any pixel change between steps must be
    confined to the distractor's reachable disk."""
    frames = rollout_frames(MOVING, n_steps=4)
    base = MOVING['variation_values']['distractor.position']
    amp = float(MOVING['variation_values']['distractor.motion'][0])
    scale = float(MOVING['variation_values']['distractor.scale'])
    radius_px = (amp + scale * np.sqrt(2) + 4.0) * RES / 512.0
    center_px = base * RES / 512.0

    yy, xx = np.mgrid[0:RES, 0:RES]
    inside = (xx - center_px[0]) ** 2 + (
        yy - center_px[1]
    ) ** 2 <= radius_px**2

    diff = (
        np.abs(
            frames[0].astype(np.float32) - frames[3].astype(np.float32)
        ).max(axis=-1)
        > 10
    )
    leaked = np.logical_and(diff, ~inside)
    assert leaked.sum() == 0, (
        f'{leaked.sum()} changed pixels outside the distractor motion disk'
    )


# --------------------------------------------------------------------------
# distractor.shape — the distractor must not resemble the agent (blue disk)
# or the block (T); default is a 6-point star (two overlapping triangles)
# --------------------------------------------------------------------------


def shape_options(shape):
    return {
        'variation': ['distractor.shape', 'distractor.scale'],
        'variation_values': {
            'distractor.shape': shape,
            'distractor.scale': np.float32(30),
        },
    }


def render_shape(shape, seed=3):
    env = make_env()
    env.reset(seed=seed, options=shape_options(shape))
    frame = env.render().copy()
    env.close()
    return frame


def test_distractor_shapes_render_differently():
    square = render_shape(0)
    triangle = render_shape(1)
    star = render_shape(2)
    assert mad(square, triangle) > 0.05
    assert mad(square, star) > 0.05
    assert mad(triangle, star) > 0.05


def test_default_distractor_shape_is_star():
    env = make_env()
    space = env.variation_space['distractor']['shape']
    env.close()
    assert int(space.init_value) == 2
    default = render_shape(2)
    star = render_shape(2)
    assert np.array_equal(default, star)
