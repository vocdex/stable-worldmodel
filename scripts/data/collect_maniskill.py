import h5py
import hydra
import numpy as np
from loguru import logger as logging
from omegaconf import DictConfig, OmegaConf

import stable_worldmodel as swm
from stable_worldmodel.data.utils import get_cache_dir
from stable_worldmodel.envs.maniskill import ManiSkillExpertPolicy


@hydra.main(
    version_base=None, config_path='./config', config_name='maniskill',
)
def run(cfg: DictConfig):

    world = swm.World(
        cfg.env_name,
        **cfg.world,
        width=cfg.world.image_shape[1],
        height=cfg.world.image_shape[0],
    )

    options = cfg.get('options')
    options = OmegaConf.to_object(options) if options is not None else None
    rng = np.random.default_rng(cfg.seed)
    world.set_policy(ManiSkillExpertPolicy(seed=cfg.seed))

    world.record_dataset(
        cfg.dataset_name,
        episodes=cfg.num_traj,
        seed=rng.integers(0, 1_000_000).item(),
        cache_dir=cfg.cache_dir,
        options=options,
    )

    # Log success and recovery stats from recorded dataset
    ds_dir = get_cache_dir(cfg.cache_dir, sub_folder='datasets')
    h5_path = ds_dir / f'{cfg.dataset_name}.h5'
    with h5py.File(h5_path, 'r') as f:
        ep_lens = f['ep_len'][:]
        ep_offsets = f['ep_offset'][:]
        actions = f['action'][:]
        n_total = len(ep_lens)

        if 'eval_success' in f:
            success = f['eval_success'][:]
            n_success = sum(
                success[off + l - 1]
                for off, l in zip(ep_offsets, ep_lens)
            )
            logging.info(
                f'Success: {n_success}/{n_total}'
                f' ({100 * n_success / n_total:.1f}%)'
            )

        # Count pick attempts per episode via home-return detection
        # (TCP rises above 0.30 after having been below 0.15)
        from collections import Counter
        tcp = f['extra_tcp_pose'][:, 2]
        attempt_counts = []
        for off, l in zip(ep_offsets, ep_lens):
            z = tcp[off:off + l]
            attempts = 1
            was_low = False
            for s in range(l):
                if z[s] < 0.15:
                    was_low = True
                elif z[s] > 0.30 and was_low:
                    attempts += 1
                    was_low = False
            attempt_counts.append(attempts)
        dist = Counter(attempt_counts)
        n_recovery = sum(1 for a in attempt_counts if a > 1)
        logging.info(
            f'Recovery: {n_recovery}/{n_total}'
            f' ({100 * n_recovery / n_total:.1f}%)'
        )
        for k in sorted(dist):
            logging.info(f'  {k} attempt(s): {dist[k]} eps')

    logging.success(
        f'Completed data collection for {cfg.env_name}'
    )


if __name__ == '__main__':
    run()
