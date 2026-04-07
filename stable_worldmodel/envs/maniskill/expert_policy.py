import numpy as np

from stable_worldmodel.envs.maniskill.oracles import (
    ComposableTabletopOracle,
    PickCubeMarkovOracle,
    StackCubeMarkovOracle,
    StackPyramidMarkovOracle,
)
from stable_worldmodel.policy import BasePolicy

ORACLE_MAP = {
    'PickCube': PickCubeMarkovOracle,
    'StackCube': StackCubeMarkovOracle,
    'StackPyramid': StackPyramidMarkovOracle,
    'ComposableTabletop': ComposableTabletopOracle,
}

# Episode mode probabilities  [clean, noisy, play, random]
DEFAULT_MODE_PROBS = [0.2, 0.5, 0.2, 0.1]


class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated noise."""

    def __init__(self, size, rng, sigma=0.15, theta=0.15, dt=1.0):
        self.size = size
        self.rng = rng
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.state = np.zeros(size)

    def reset(self, sigma=None):
        self.state = np.zeros(self.size)
        if sigma is not None:
            self.sigma = sigma

    def sample(self):
        dx = (
            -self.theta * self.state * self.dt
            + self.sigma * np.sqrt(self.dt) * self.rng.standard_normal(self.size)
        )
        self.state += dx
        return self.state.copy()


class ManiSkillExpertPolicy(BasePolicy):
    """Closed-loop Markov oracle expert with stratified episode modes.

    Each episode is one of four modes (rolled at reset):
      - clean  (20%): pure expert, no noise, no drops
      - noisy  (50%): OU position noise + discrete drop events
      - play   (20%): task then sweep near table pushing objects
      - random (10%): purely random actions

    This mixture gives the world model coverage of optimal paths, noisy
    recoveries, contact dynamics, and sub-optimal states.
    """

    def __init__(
        self,
        action_noise=0.15,
        noise_theta=0.15,
        p_drop=0.05,
        drop_steps=5,
        mode_probs=None,
        min_norm=0.1,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.type = 'maniskill_markov_expert'
        self.action_noise = action_noise
        self.noise_theta = noise_theta
        self.p_drop = p_drop
        self.drop_steps = drop_steps
        self.mode_probs = mode_probs or DEFAULT_MODE_PROBS
        self.min_norm = min_norm
        self.set_seed(seed)

    def set_seed(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def set_env(self, env):
        self.env = env
        base_env = self.env.unwrapped
        envs = base_env.envs if hasattr(base_env, 'envs') else [base_env]
        env_name = envs[0].unwrapped.env_name

        oracle_cls = ORACLE_MAP.get(env_name)
        if oracle_cls is None:
            raise ValueError(f'No oracle for env {env_name!r}')

        n = len(envs)
        # Pass shape info to ComposableTabletop oracle
        oracle_kwargs = {}
        single_env = envs[0].unwrapped
        if env_name == 'ComposableTabletop' and hasattr(single_env, '_object_specs'):
            specs = single_env._object_specs
            oracle_kwargs['task'] = 'stack' if len(specs) >= 2 else 'pick'
            oracle_kwargs['num_objects'] = len(specs)
            oracle_kwargs['shapes'] = [s.shape for s in specs]
        self._oracle_agents = [
            oracle_cls(min_norm=self.min_norm, **oracle_kwargs) for _ in range(n)
        ]
        self._ou = [
            OUNoise(3, self.rng, theta=self.noise_theta) for _ in range(n)
        ]
        self._drop_remaining = np.zeros(n, dtype=int)
        self._mode = ['noisy'] * n

    def _roll_mode(self):
        return self.rng.choice(
            ['clean', 'noisy', 'play', 'random'], p=self.mode_probs,
        )

    def get_action(self, info_dict, **kwargs):
        base_env = self.env.unwrapped
        envs = base_env.envs if hasattr(base_env, 'envs') else [base_env]

        act_shape = self.env.action_space.shape
        actions = np.zeros(act_shape, dtype=np.float32)

        for i in range(len(envs)):
            info = {
                k: v[i][0]
                for k, v in info_dict.items()
                if not k.startswith('_')
            }

            if info['step_idx'] == 0:
                self._mode[i] = self._roll_mode()
                xi = self.rng.uniform(0, self.action_noise)
                self._ou[i].reset(sigma=xi)
                self._drop_remaining[i] = 0
                self._oracle_agents[i].reset(None, info)

            mode = self._mode[i]
            oracle = self._oracle_agents[i]

            # --- Random mode: smooth OU-driven exploration ---
            if mode == 'random':
                tcp = info['extra/tcp_pose'][:3]
                action = oracle._play_action(tcp)
                action[:3] += self._ou[i].sample()
                actions[i] = np.clip(action, -1, 1)
                continue

            # --- Play mode: task first, then sweep near objects ---
            if mode == 'play':
                tcp = info['extra/tcp_pose'][:3]
                if oracle._task_done:
                    action = oracle._play_action(tcp)
                    action[:3] += self._ou[i].sample()
                else:
                    action = oracle.select_action(None, info)
                    action = np.array(action)
                    action[:3] += self._ou[i].sample()
                actions[i] = np.clip(action, -1, 1)
                continue

            # --- Clean / Noisy modes: oracle-driven ---
            action = oracle.select_action(None, info)
            action = np.array(action)

            if mode == 'noisy':
                # OU position noise
                action[:3] += self._ou[i].sample()

                # Discrete drop events while grasped
                if not oracle._task_done:
                    grasped = info.get(
                        'eval/is_cube_grasped',
                        info.get('eval/is_cubeA_grasped',
                        info.get('eval/is_obj_0_grasped', False)),
                    )
                    if self._drop_remaining[i] > 0:
                        action[3] = 1.0
                        self._drop_remaining[i] -= 1
                    elif grasped and self.rng.uniform() < self.p_drop:
                        action[3] = 1.0
                        self._drop_remaining[i] = self.drop_steps

            # clean mode: no noise at all, just oracle output

            actions[i] = np.clip(action, -1, 1)

        return actions
