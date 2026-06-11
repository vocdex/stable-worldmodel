"""Tests for the OGBench-Cube natural floor texture (`floor.texture_id`).

Same indexing contract as PushT's `background.texture_id`: 1-based index into
the sorted entries of the texture dir (`texture_dir` kwarg > $SWM_TEXTURE_DIR
> swm cache `textures/`); 0 (default) keeps the procedural checker floor.
Only static image files are supported on cube (clip dirs raise — per-step
MuJoCo texture upload is not wired yet). All tests are slow (EGL + model
recompiles).
"""

import os

os.environ.setdefault('MUJOCO_GL', 'egl')

import gymnasium as gym
import numpy as np
import pytest

import stable_worldmodel  # noqa: F401  (registers envs)


pytestmark = pytest.mark.slow

ENV_KWARGS = dict(
    env_type='single',
    ob_type='states',
    multiview=False,
    width=224,
    height=224,
    visualize_info=False,
)


def mad(a, b):
    return float(
        np.mean(
            np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32))
        )
    )


@pytest.fixture(scope='module')
def texture_dir(tmp_path_factory):
    import imageio

    root = tmp_path_factory.mktemp('cube_textures')
    for name, seed in [('a_one.png', 1), ('b_two.png', 2)]:
        rng = np.random.default_rng(seed)
        imageio.imwrite(
            root / name,
            rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8),
        )
    clip = root / 'c_clip'
    clip.mkdir()
    rng = np.random.default_rng(3)
    imageio.imwrite(
        clip / '000.png',
        rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8),
    )
    return root


def make_env(texture_dir):
    return gym.make(
        'swm/OGBCube-v0', texture_dir=str(texture_dir), **ENV_KWARGS
    ).unwrapped


def floor_options(texture_id):
    return {
        'variation': ['floor.texture_id'],
        'variation_values': {'floor.texture_id': texture_id},
    }


def render_floor(env, texture_id, seed=4):
    env.reset(seed=seed, options=floor_options(texture_id))
    return env.render().copy()


def test_floor_texture_visibly_replaces_checker(texture_dir):
    env = make_env(texture_dir)
    plain = render_floor(env, 0)
    textured = render_floor(env, 1)
    env.close()
    assert mad(plain, textured) > 5.0


def test_texture_id_selects_different_files(texture_dir):
    env = make_env(texture_dir)
    one = render_floor(env, 1)
    two = render_floor(env, 2)
    env.close()
    assert mad(one, two) > 1.0


def test_reset_restores_checker_floor(texture_dir):
    """Texture on, then a plain reset: the checker floor must come back.
    Frames are compared at a pinned qpos/qvel (cube task selection is
    globally random across resets)."""
    env = make_env(texture_dir)
    render_floor(env, 0)
    qpos, qvel = env._data.qpos.copy(), env._data.qvel.copy()
    env.set_state(qpos, qvel)
    plain_before = env.render().copy()
    render_floor(env, 1)
    render_floor(env, 0)
    env.set_state(qpos, qvel)
    plain_after = env.render().copy()
    env.close()
    assert mad(plain_before, plain_after) < 1.0


def test_clip_dir_raises_not_implemented(texture_dir):
    env = make_env(texture_dir)
    with pytest.raises(NotImplementedError):
        render_floor(env, 3)
    env.close()


def test_missing_texture_raises(texture_dir):
    env = make_env(texture_dir)
    with pytest.raises(FileNotFoundError):
        render_floor(env, 9)
    env.close()


def test_rendering_is_deterministic(texture_dir):
    """Same texture + same explicit state => same frame. (Two resets are NOT
    state-identical on cube — task selection uses global np.random — so the
    state is pinned via set_state.)"""
    env_a = make_env(texture_dir)
    env_b = make_env(texture_dir)
    render_floor(env_a, 1)
    render_floor(env_b, 1)
    qpos, qvel = env_a._data.qpos.copy(), env_a._data.qvel.copy()
    env_a.set_state(qpos, qvel)
    env_b.set_state(qpos, qvel)
    a = env_a.render().copy()
    b = env_b.render().copy()
    env_a.close()
    env_b.close()
    assert mad(a, b) < 0.5
