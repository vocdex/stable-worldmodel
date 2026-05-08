"""Generate OGBench Cube dataset where cube moves independently of the arm.

For each episode:
- Arm/gripper qpos are borrowed from a random expert trajectory so the arm
  follows realistic motion patterns.
- The cube follows a smooth scripted random walk on the table, decorrelated
  from the arm.
- When the scripted cube position would overlap the arm (contact detected
  via MuJoCo's collision query), the cube is nudged along the contact
  normals until the overlap is resolved.  This produces visually plausible
  collisions: the arm appears to push the cube out of its way.

Output H5 matches the schema of cube_single_expert_256.h5 so the existing
SlotContrast converter (save_cube.py) works unchanged.

Usage:
    MUJOCO_GL=egl python scripts/data/collect_cube_independent.py \
        --src-path ~/.stable_worldmodel/cube_single/cube_single_expert_256.h5 \
        --out-path ~/.stable_worldmodel/cube_single/cube_single_indep_256.h5 \
        --num-episodes 2000 --seed 0
"""
import argparse
import os

os.environ.setdefault('MUJOCO_GL', 'egl')

import h5py
import hdf5plugin
import mujoco
import numpy as np
from scipy.ndimage import binary_dilation
import tqdm

from stable_worldmodel.envs.ogbench.cube_env import CubeEnv

LABEL_BG = 0
LABEL_CUBE = 1
LABEL_ARM = 2

CUBE_QPOS_START = 14  # qpos[14:21] = freejoint (x,y,z,qw,qx,qy,qz)

X_MIN, X_MAX = 0.3, 0.55
Y_MIN, Y_MAX = -0.3, 0.3
CUBE_Z = 0.02


def build_geom_to_label(model):
    lut = np.zeros(model.ngeom, dtype=np.uint8)
    for gid in range(model.ngeom):
        name = model.geom(gid).name
        if name.startswith('object_') and not name.startswith('target_object_'):
            lut[gid] = LABEL_CUBE
        elif name.startswith('ur5e/'):
            lut[gid] = LABEL_ARM
    return lut


def yaw_to_quat(yaw):
    c, s = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([c, 0.0, 0.0, s])


def sample_cube_trajectory(T, rng, speed=0.003, yaw_speed=0.01):
    """Smooth random walk of the cube on the table."""
    xy = np.empty((T, 2))
    vel = np.zeros(2)
    yaw_arr = np.empty(T)
    yaw_vel = 0.0

    xy[0] = np.array([rng.uniform(X_MIN, X_MAX), rng.uniform(Y_MIN, Y_MAX)])
    yaw_arr[0] = rng.uniform(-np.pi, np.pi)

    damping = 0.9
    for t in range(1, T):
        vel = damping * vel + rng.normal(0, speed, size=2)
        new_xy = xy[t - 1] + vel
        if new_xy[0] < X_MIN or new_xy[0] > X_MAX:
            vel[0] *= -1
            new_xy[0] = np.clip(new_xy[0], X_MIN, X_MAX)
        if new_xy[1] < Y_MIN or new_xy[1] > Y_MAX:
            vel[1] *= -1
            new_xy[1] = np.clip(new_xy[1], Y_MIN, Y_MAX)
        xy[t] = new_xy

        yaw_vel = damping * yaw_vel + rng.normal(0, yaw_speed)
        yaw_arr[t] = yaw_arr[t - 1] + yaw_vel

    qpos_cube = np.zeros((T, 7), dtype=np.float64)
    qpos_cube[:, 0] = xy[:, 0]
    qpos_cube[:, 1] = xy[:, 1]
    qpos_cube[:, 2] = CUBE_Z
    for t in range(T):
        qpos_cube[t, 3:7] = yaw_to_quat(yaw_arr[t])
    return qpos_cube


def resolve_cube_arm_overlap(model, data, lut, max_iters=8, margin=0.003):
    """Nudge the cube along contact normals until it no longer penetrates the arm."""
    for _ in range(max_iters):
        mujoco.mj_forward(model, data)
        push = np.zeros(2)
        has_overlap = False
        for i in range(data.ncon):
            c = data.contact[i]
            l1 = lut[c.geom1]
            l2 = lut[c.geom2]
            involves_cube_arm = {l1, l2} == {LABEL_CUBE, LABEL_ARM}
            if not involves_cube_arm or c.dist > 0:
                continue
            # c.frame[:3] is the Z-axis of the contact frame, pointing from geom1 to geom2.
            normal = c.frame[:3]
            penetration = -c.dist + margin
            # Push cube away from arm.
            if l1 == LABEL_CUBE:
                push -= normal[:2] * penetration
            else:
                push += normal[:2] * penetration
            has_overlap = True
        if not has_overlap:
            return False
        data.qpos[CUBE_QPOS_START] += push[0]
        data.qpos[CUBE_QPOS_START + 1] += push[1]
        # Stay in workspace.
        data.qpos[CUBE_QPOS_START] = np.clip(data.qpos[CUBE_QPOS_START], X_MIN, X_MAX)
        data.qpos[CUBE_QPOS_START + 1] = np.clip(data.qpos[CUBE_QPOS_START + 1], Y_MIN, Y_MAX)
    return True  # still overlapping after max_iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src-path', required=True)
    p.add_argument('--out-path', required=True)
    p.add_argument('--num-episodes', type=int, default=2000)
    p.add_argument('--episode-len', type=int, default=None)
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--batch', type=int, default=500)
    p.add_argument('--gzip-level', type=int, default=6)
    args = p.parse_args()

    h = w = args.size
    rng = np.random.default_rng(args.seed)

    env = CubeEnv(
        env_type='single', ob_type='pixels', width=w, height=h,
        visualize_info=False, terminate_at_goal=False, mode='data_collection',
        pixel_transparent_arm=False,
    )
    env.reset()
    env._model.vis.quality.offsamples = 0
    lut = build_geom_to_label(env._model)

    with h5py.File(args.src_path, 'r') as fsrc:
        src_ep_offset = fsrc['ep_offset'][:]
        src_ep_len = fsrc['ep_len'][:]
        src_n_eps = len(src_ep_len)

        ep_source_idx = rng.integers(0, src_n_eps, size=args.num_episodes)
        ep_lens = np.array([
            args.episode_len if args.episode_len is not None else int(src_ep_len[i])
            for i in ep_source_idx
        ], dtype=np.int32)
        ep_offsets = np.concatenate([[0], np.cumsum(ep_lens[:-1])]).astype(np.int64)
        n_total = int(ep_lens.sum())

        print(f'Generating {args.num_episodes} episodes, {n_total} total frames @ {h}x{w}')

        with h5py.File(args.out_path, 'w') as fout:
            chunk_px = (min(args.batch, 64), h, w, 3)
            chunk_seg = (min(args.batch, 64), h, w)

            px_ds = fout.create_dataset(
                'pixels', shape=(n_total, h, w, 3), dtype=np.uint8,
                chunks=chunk_px,
                **hdf5plugin.Blosc(cname='zstd', clevel=5,
                                   shuffle=hdf5plugin.Blosc.SHUFFLE),
            )
            seg_ds = fout.create_dataset(
                'segmentation', shape=(n_total, h, w), dtype=np.uint8,
                chunks=chunk_seg, compression='gzip',
                compression_opts=args.gzip_level, shuffle=True,
            )
            qpos_ds = fout.create_dataset(
                'qpos', shape=(n_total, env._model.nq), dtype=np.float64)
            qvel_ds = fout.create_dataset(
                'qvel', shape=(n_total, env._model.nv), dtype=np.float64)
            fout.create_dataset('action', shape=(n_total, 5), dtype=np.float32)
            fout.create_dataset('observation', shape=(n_total, 28), dtype=np.float64)
            fout.create_dataset('ep_offset', data=ep_offsets)
            fout.create_dataset('ep_len', data=ep_lens)

            px_buf = np.empty((args.batch, h, w, 3), dtype=np.uint8)
            seg_buf = np.empty((args.batch, h, w), dtype=np.uint8)
            qpos_buf = np.empty((args.batch, env._model.nq), dtype=np.float64)
            qvel_buf = np.empty((args.batch, env._model.nv), dtype=np.float64)
            buf_idx = 0
            global_frame = 0

            pbar = tqdm.tqdm(total=n_total, desc='Rendering')

            def flush(start, count):
                if count == 0:
                    return
                px_ds[start:start + count] = px_buf[:count]
                seg_ds[start:start + count] = seg_buf[:count]
                qpos_ds[start:start + count] = qpos_buf[:count]
                qvel_ds[start:start + count] = qvel_buf[:count]

            for ep_i in range(args.num_episodes):
                src_i = int(ep_source_idx[ep_i])
                src_off = int(src_ep_offset[src_i])
                T = int(ep_lens[ep_i])
                src_len = int(src_ep_len[src_i])
                idxs = src_off + (np.arange(T) % src_len)
                arm_qpos = fsrc['qpos'][idxs][:, :CUBE_QPOS_START]
                arm_qvel = fsrc['qvel'][idxs][:, :CUBE_QPOS_START]

                scripted_cube = sample_cube_trajectory(T, rng)
                # Use deltas between consecutive scripted frames and integrate
                # from the previous *resolved* cube position.  That way a push
                # from the arm sticks instead of being undone next frame.
                scripted_delta_xy = np.diff(scripted_cube[:, :2], axis=0,
                                            prepend=scripted_cube[:1, :2])
                scripted_delta_yaw = np.diff(scripted_cube[:, 3:7], axis=0,
                                             prepend=scripted_cube[:1, 3:7])

                # Cube's current state (starts at scripted initial pose).
                cur_cube_qpos = scripted_cube[0].copy()

                for t in range(T):
                    # Advance cube by scripted delta from its current position.
                    cur_cube_qpos[:2] += scripted_delta_xy[t]
                    cur_cube_qpos[:2] = np.clip(
                        cur_cube_qpos[:2],
                        [X_MIN, Y_MIN], [X_MAX, Y_MAX])
                    cur_cube_qpos[2] = CUBE_Z
                    # Rotation: just copy scripted yaw quat (small deltas).
                    cur_cube_qpos[3:7] = scripted_cube[t, 3:7]

                    qpos = np.empty(env._model.nq, dtype=np.float64)
                    qpos[:CUBE_QPOS_START] = arm_qpos[t]
                    qpos[CUBE_QPOS_START:] = cur_cube_qpos
                    qvel = np.zeros(env._model.nv, dtype=np.float64)
                    qvel[:CUBE_QPOS_START] = arm_qvel[t]

                    env._data.qpos[:] = qpos
                    env._data.qvel[:] = qvel
                    mujoco.mj_forward(env._model, env._data)

                    # Resolve any arm-cube penetration by nudging the cube.
                    resolve_cube_arm_overlap(env._model, env._data, lut)

                    # Remember the resolved pose for the next step.
                    cur_cube_qpos = env._data.qpos[CUBE_QPOS_START:].copy()

                    qpos_buf[buf_idx] = env._data.qpos
                    qvel_buf[buf_idx] = env._data.qvel

                    raw = env.render(segmentation=True)
                    geom_ids = np.clip(raw[:, :, 0], -1, len(lut) - 1)
                    mask = np.where(geom_ids >= 0, lut[geom_ids], LABEL_BG)
                    cube_dilated = binary_dilation(mask == LABEL_CUBE)
                    mask[(mask == LABEL_ARM) & cube_dilated] = LABEL_CUBE

                    px_buf[buf_idx] = env.render()
                    seg_buf[buf_idx] = mask

                    buf_idx += 1
                    global_frame += 1
                    pbar.update(1)

                    if buf_idx == args.batch:
                        flush(global_frame - args.batch, args.batch)
                        buf_idx = 0

            flush(global_frame - buf_idx, buf_idx)
            pbar.close()

    env.close()
    print(f'Done. Wrote {args.num_episodes} episodes ({n_total} frames) to {args.out_path}')


if __name__ == '__main__':
    main()
