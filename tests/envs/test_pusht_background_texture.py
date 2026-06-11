"""Tests for the PushT natural-background variation (`background.texture_id`).

`background.texture_id` indexes (1-based) the sorted entries of the texture
directory (`texture_dir` env kwarg, else $SWM_TEXTURE_DIR, else the swm cache
`textures/` folder); 0 (default) keeps the plain background color. An entry
that is an image file is a static background; an entry that is a directory is
a clip whose frames advance one per env step (looping) — the DCS-style
dynamic background. Missing textures fail loudly (no silent no-op).
"""

import os

os.environ.setdefault('MUJOCO_GL', 'egl')

import gymnasium as gym
import numpy as np
import pytest

import stable_worldmodel  # noqa: F401  (registers envs)


RES = 128


def mad(a, b):
    return float(
        np.mean(
            np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32))
        )
    )


def _write_noise_png(path, seed, size=64):
    import imageio

    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    imageio.imwrite(path, img)


@pytest.fixture(scope='module')
def texture_dir(tmp_path_factory):
    """Two static images + one 4-frame clip directory."""
    root = tmp_path_factory.mktemp('textures')
    _write_noise_png(root / 'a_static1.png', seed=1)
    _write_noise_png(root / 'b_static2.png', seed=2)
    clip = root / 'c_clip'
    clip.mkdir()
    for t in range(4):
        _write_noise_png(clip / f'{t:03d}.png', seed=100 + t)
    return root


def make_env(texture_dir):
    return gym.make(
        'swm/PushT-v1', resolution=RES, texture_dir=str(texture_dir)
    ).unwrapped


def texture_options(texture_id):
    return {
        'variation': ['background.texture_id'],
        'variation_values': {'background.texture_id': texture_id},
    }


def rollout_frames(texture_dir, texture_id, n_steps=0, seed=3):
    env = make_env(texture_dir)
    env.reset(seed=seed, options=texture_options(texture_id))
    frames = [env.render().copy()]
    for _ in range(n_steps):
        env.step(np.zeros(2, dtype=np.float32))
        frames.append(env.render().copy())
    env.close()
    return frames


def test_texture_visibly_replaces_background(texture_dir):
    plain = rollout_frames(texture_dir, 0)[0]
    textured = rollout_frames(texture_dir, 1)[0]
    assert mad(plain, textured) > 5.0


def test_texture_id_selects_different_files(texture_dir):
    one = rollout_frames(texture_dir, 1)[0]
    two = rollout_frames(texture_dir, 2)[0]
    assert mad(one, two) > 5.0


def test_objects_drawn_on_top_of_texture(texture_dir):
    """The agent must still be visible: its RoyalBlue disk survives the
    noise background."""
    frame = rollout_frames(texture_dir, 1)[0]
    royal_blue = np.array([65, 105, 225], dtype=np.float32)
    dist = np.linalg.norm(frame.astype(np.float32) - royal_blue, axis=-1)
    assert (dist < 30).sum() > 10, 'agent not visible over the texture'


def test_static_texture_constant_across_steps(texture_dir):
    frames = rollout_frames(texture_dir, 1, n_steps=3)
    assert mad(frames[0], frames[-1]) < 0.05


def test_clip_advances_per_step(texture_dir):
    frames = rollout_frames(texture_dir, 3, n_steps=2)
    assert mad(frames[0], frames[1]) > 5.0
    assert mad(frames[1], frames[2]) > 5.0


def test_clip_loops(texture_dir):
    frames = rollout_frames(texture_dir, 3, n_steps=4)
    assert mad(frames[0], frames[4]) < 0.05  # period 4: t=4 == t=0


def test_rendering_is_deterministic(texture_dir):
    a = rollout_frames(texture_dir, 3, n_steps=3, seed=5)
    b = rollout_frames(texture_dir, 3, n_steps=3, seed=5)
    for fa, fb in zip(a, b):
        assert np.array_equal(fa, fb)


def test_missing_texture_raises(texture_dir):
    env = make_env(texture_dir)
    with pytest.raises(FileNotFoundError):
        env.reset(seed=3, options=texture_options(9))
    env.close()


def test_env_var_fallback(texture_dir, monkeypatch):
    monkeypatch.setenv('SWM_TEXTURE_DIR', str(texture_dir))
    env = gym.make('swm/PushT-v1', resolution=RES).unwrapped
    env.reset(seed=3, options=texture_options(1))
    textured = env.render().copy()
    env.reset(seed=3, options=texture_options(0))
    plain = env.render().copy()
    env.close()
    assert mad(plain, textured) > 5.0
