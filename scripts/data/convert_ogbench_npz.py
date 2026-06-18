"""Convert OGBench .npz datasets to SWM H5 format.

OGBench provides state datasets (qpos/qvel/obs/actions/terminals) at
https://rail.eecs.berkeley.edu/datasets/ogbench/.

This script converts them to the SWM H5 schema so extract_cube_masks.py
can re-render pixels + segmentation on top in a second pass.

Output H5 fields written here (subset of full SWM schema):
    qpos          (N, qpos_dim)   float32
    qvel          (N, qvel_dim)   float32
    action        (N, action_dim) float32
    observation   (N, obs_dim)    float32
    ep_offset     (n_eps,)        int64
    ep_len        (n_eps,)        int64

All other SWM fields (pixels, segmentation, privileged_*, proprio_*, etc.)
are added by extract_cube_masks.py.

Usage:
    python scripts/data/convert_ogbench_npz.py \\
        --npz ~/.stable_worldmodel/ogbench_datasets/cube-double-play-v0.npz \\
        --out ~/.stable_worldmodel/datasets/ogbench/cube_double_ogb.h5
"""
import argparse
import os

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import tqdm


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    npz_path = os.path.abspath(os.path.expanduser(args.npz))
    out_path = os.path.abspath(os.path.expanduser(args.out))

    if os.path.exists(out_path):
        p.error(f'refusing to overwrite: {out_path}')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f'Loading {npz_path}...')
    f = np.load(npz_path)

    terminals = f['terminals'].astype(bool)
    ep_ends = np.where(terminals)[0]
    ep_starts = np.concatenate([[0], ep_ends[:-1] + 1])
    ep_len = (ep_ends - ep_starts + 1).astype(np.int64)
    ep_offset = ep_starts.astype(np.int64)
    n_eps = len(ep_len)
    n_steps = len(terminals)

    print(f'Episodes: {n_eps}, steps: {n_steps}')
    print(f'Episode length: min={ep_len.min()}, mean={ep_len.mean():.1f}, max={ep_len.max()}')

    comp = hdf5plugin.Blosc(cname='zstd', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)

    with h5py.File(out_path, 'w') as fout:
        fout.create_dataset('ep_offset', data=ep_offset)
        fout.create_dataset('ep_len', data=ep_len)
        fout.create_dataset('qpos', data=f['qpos'].astype(np.float32),
                            chunks=(512, f['qpos'].shape[1]), **comp)
        fout.create_dataset('qvel', data=f['qvel'].astype(np.float32),
                            chunks=(512, f['qvel'].shape[1]), **comp)
        fout.create_dataset('action', data=f['actions'].astype(np.float32),
                            chunks=(512, f['actions'].shape[1]), **comp)
        fout.create_dataset('observation', data=f['observations'].astype(np.float32),
                            chunks=(512, f['observations'].shape[1]), **comp)

    print(f'Written: {out_path}')
    print(f'Next step: run extract_cube_masks.py --in-path {out_path} --env-type <type> --shadow-class')


if __name__ == '__main__':
    main()
