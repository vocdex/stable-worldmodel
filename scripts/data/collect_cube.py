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
    policy_type = cfg.get('policy_type', 'markov_oracle')
    oracle_gain = cfg.get('oracle_gain', None)

    if oracle_gain is not None:
        import ogbench.manipspace.oracles.markov.cube_markov as _cm
        _orig = _cm.CubeMarkovOracle
        _gp, _gy = float(oracle_gain), float(oracle_gain) * 0.6

        class _LowGainOracle(_orig):
            def select_action(self, ob, info):
                ep = info['proprio/effector_pos']; ey = info['proprio/effector_yaw'][0]
                tb = info['privileged/target_block']
                bp = info[f'privileged/block_{tb}_pos']
                by = self.shortest_yaw(ey, info[f'privileged/block_{tb}_yaw'][0])
                tp = info['privileged/target_block_pos']
                ty = self.shortest_yaw(ey, info['privileged/target_block_yaw'][0])
                off = np.array([0,0,0.18]); thr = 0.16
                gc = info['proprio/gripper_contact'] > 0.5; go = info['proprio/gripper_contact'] < 0.1
                above=ep[2]>thr; xya=np.linalg.norm(bp[:2]-ep[:2])<=0.04
                pa=np.linalg.norm(bp-ep)<=0.02; txya=np.linalg.norm(tp[:2]-bp[:2])<=0.04
                tpa=np.linalg.norm(tp-bp)<=0.02; fpa=np.linalg.norm(self._final_pos-ep)<=0.04
                a=np.zeros(5)
                if not tpa:
                    if not xya:   d=self.shape_diff(bp+off-ep); a[:3]=d*_gp; a[3]=(by-ey)*_gy; a[4]=-1
                    elif not pa:  d=self.shape_diff(bp-ep);     a[:3]=d*_gp; a[3]=(by-ey)*_gy; a[4]=-1
                    elif pa and not gc: d=self.shape_diff(bp-ep); a[:3]=d*_gp; a[3]=(by-ey)*_gy; a[4]=1
                    elif pa and gc and not above and not txya:
                        d=self.shape_diff(np.array([bp[0],bp[1],off[2]*2])-ep); a[:3]=d*_gp; a[3]=(ty-by)*_gy; a[4]=1
                    elif pa and gc and above and not txya:
                        d=self.shape_diff(tp+off-ep); a[:3]=d*_gp; a[3]=(ty-by)*_gy; a[4]=1
                    else: d=self.shape_diff(tp-ep); a[:3]=d*_gp; a[3]=(ty-by)*_gy; a[4]=1
                else:
                    if not go:  d=self.shape_diff(tp-ep); a[:3]=d*_gp; a[3]=(ty-by)*_gy; a[4]=-1
                    elif go and not above:
                        d=self.shape_diff(np.array([bp[0],bp[1],thr*2])-ep); a[:3]=d*_gp; a[3]=(self._final_yaw-ey)*_gy; a[4]=-1
                    else:
                        d=self.shape_diff(self._final_pos-ep); a[:3]=d*_gp; a[3]=(self._final_yaw-ey)*_gy; a[4]=-1
                    if fpa: self._done = True
                a=np.clip(a,-1,1); self._step+=1
                if self._step==self._max_step: self._done=True
                return a

        _cm.CubeMarkovOracle = _LowGainOracle

    world.set_policy(ExpertPolicy(policy_type=policy_type))

    dataset_suffix = cfg.get('dataset_suffix', '')
    dataset_name = f'ogbench/cube_{env_type}_expert{dataset_suffix}'
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
