import os

os.environ['MUJOCO_GL'] = 'egl'
import hydra
import numpy as np
from loguru import logger as logging
from omegaconf import DictConfig, OmegaConf

import stable_worldmodel as swm
from stable_worldmodel.envs.ogbench import ExpertPolicy


@hydra.main(version_base=None, config_path='./config', config_name='ogb')
def run(cfg: DictConfig):
    """Run parallel data collection script"""

    env_type = cfg.get('env_type', 'single')
    world = swm.World(
        'swm/OGBCube-v0',
        **cfg.world,
        env_type=env_type,
        width=256,
        height=256,
        visualize_info=False,
        terminate_at_goal=False,
        mode='data_collection',
    )

    options = cfg.get('options')
    options = OmegaConf.to_object(options) if options is not None else None
    rng = np.random.default_rng(cfg.seed)
    world.set_policy(ExpertPolicy())

    dataset_name = f'ogbench/cube_{env_type}_expert'
    world.record_dataset(
        dataset_name,
        episodes=cfg.num_traj,
        seed=rng.integers(0, 1_000_000).item(),
        cache_dir=cfg.cache_dir,
        options=options,
    )

    logging.success('🎉🎉🎉 Completed data collection for ogbench cube 🎉🎉🎉')


if __name__ == '__main__':
    run()
