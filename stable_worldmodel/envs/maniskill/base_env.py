import gymnasium as gym
import numpy as np
import torch

from stable_worldmodel import spaces as swm_spaces


class ManiSkillSWMEnv(gym.Env):
    """Base adapter wrapping ManiSkill3 environments for stable-worldmodel.

    Uses composition to hold an internal ManiSkill3 env in CPU single-env mode,
    converting all torch tensor I/O to numpy for SWM compatibility.
    """

    metadata = {'render_modes': ['rgb_array'], 'render_fps': 20}

    def __init__(
        self,
        ms_env_id,
        height=224,
        width=224,
        control_mode='pd_ee_delta_pos',
        render_mode='rgb_array',
        ms_max_episode_steps=200,
        **kwargs,
    ):
        import mani_skill.envs  # noqa: F401 — triggers env registration
        import stable_worldmodel.envs.maniskill.static_camera  # noqa: F401

        self._ms_env_id = ms_env_id
        self._height = height
        self._width = width
        self.render_mode = render_mode
        self.env_name = ms_env_id

        self._ms_env = gym.make(
            ms_env_id,
            obs_mode='rgb+segmentation',
            control_mode=control_mode,
            render_mode='rgb_array',
            shader_dir='default',
            sensor_configs=dict(width=width, height=height),
            num_envs=1,
            max_episode_steps=ms_max_episode_steps,
        )

        ms_action = self._ms_env.action_space
        self.action_space = gym.spaces.Box(
            low=ms_action.low.item() if ms_action.low.ndim == 0 else float(ms_action.low.min()),
            high=ms_action.high.item() if ms_action.high.ndim == 0 else float(ms_action.high.max()),
            shape=ms_action.shape,
            dtype=np.float32,
        )

        self._num_objects = 1
        self._default_object_colors = np.array([[1.0, 0.0, 0.0]])
        self._object_actors = []

        self.observation_space = gym.spaces.Dict({
            'proprio': gym.spaces.Box(-np.inf, np.inf, shape=(18,), dtype=np.float32),
            'state': gym.spaces.Box(-np.inf, np.inf, shape=(18,), dtype=np.float32),
        })

        self.variation_space = self._build_variation_space()

    def _build_variation_space(self):
        return swm_spaces.Dict({
            'object': swm_spaces.Dict({
                'color': swm_spaces.Box(
                    low=0.0, high=1.0,
                    shape=(self._num_objects, 3),
                    dtype=np.float64,
                    init_value=self._default_object_colors.copy(),
                ),
                'size_scale': swm_spaces.Box(
                    low=0.8, high=1.2,
                    shape=(self._num_objects,),
                    dtype=np.float64,
                    init_value=np.ones(self._num_objects),
                ),
                'start_position': swm_spaces.Box(
                    low=np.tile([-0.2, -0.2], (self._num_objects, 1)),
                    high=np.tile([0.2, 0.2], (self._num_objects, 1)),
                    shape=(self._num_objects, 2),
                    dtype=np.float64,
                    init_value=np.zeros((self._num_objects, 2)),
                ),
            }),
            'camera': swm_spaces.Dict({
                'angle_delta': swm_spaces.Box(
                    low=-10.0, high=10.0,
                    shape=(1, 2),
                    dtype=np.float64,
                    init_value=np.zeros((1, 2)),
                ),
            }),
            'light': swm_spaces.Dict({
                'intensity': swm_spaces.Box(
                    low=0.3, high=1.5,
                    shape=(1,),
                    dtype=np.float64,
                    init_value=np.array([1.0]),
                ),
                'color': swm_spaces.Box(
                    low=0.7, high=1.0,
                    shape=(3,),
                    dtype=np.float64,
                    init_value=np.array([1.0, 1.0, 1.0]),
                ),
            }),
            'floor': swm_spaces.Dict({
                'color': swm_spaces.Box(
                    low=0.0, high=1.0,
                    shape=(3,),
                    dtype=np.float64,
                    init_value=np.array([0.5, 0.5, 0.5]),
                ),
            }),
        })

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        if isinstance(x, dict):
            return {k: self._to_numpy(v) for k, v in x.items()}
        return x

    def _build_obs(self, ms_obs):
        agent = ms_obs['agent']
        qpos = self._to_numpy(agent['qpos']).squeeze(0)
        qvel = self._to_numpy(agent['qvel']).squeeze(0)
        proprio = np.concatenate([qpos, qvel]).astype(np.float32)
        return {'proprio': proprio, 'state': proprio.copy()}

    def _build_info(self, ms_obs, ms_info):
        info = {}
        extra = ms_obs.get('extra', {})
        for k, v in extra.items():
            val = self._to_numpy(v)
            if val.ndim > 0:
                val = val.squeeze(0)
            info[f'extra/{k}'] = val

        # Add privileged object poses (not in rgb+segmentation obs)
        uw = self._ms_env.unwrapped
        if hasattr(uw, 'cube'):
            info['extra/obj_pose'] = self._to_numpy(
                uw.cube.pose.raw_pose
            ).squeeze(0).astype(np.float32)
        if hasattr(uw, 'cubeA'):
            info['extra/cubeA_pose'] = self._to_numpy(
                uw.cubeA.pose.raw_pose
            ).squeeze(0).astype(np.float32)
        if hasattr(uw, 'cubeB'):
            info['extra/cubeB_pose'] = self._to_numpy(
                uw.cubeB.pose.raw_pose
            ).squeeze(0).astype(np.float32)
        if hasattr(uw, 'cubeC'):
            info['extra/cubeC_pose'] = self._to_numpy(
                uw.cubeC.pose.raw_pose
            ).squeeze(0).astype(np.float32)

        # Evaluation metrics (success, is_grasped, etc.)
        eval_info = uw.evaluate()
        for k, v in eval_info.items():
            val = self._to_numpy(v)
            if val.ndim > 0:
                val = val.squeeze()
            info[f'eval/{k}'] = val
        info['success'] = bool(info.get('eval/success', False))

        # Per-cube grasp detection (not always in evaluate())
        agent = uw.agent
        for attr in ('cube', 'cubeA', 'cubeB', 'cubeC'):
            if hasattr(uw, attr):
                obj = getattr(uw, attr)
                val = self._to_numpy(agent.is_grasping(obj))
                if val.ndim > 0:
                    val = val.squeeze()
                info[f'eval/is_{attr}_grasped'] = bool(val)

        # Segmentation mask from main camera
        sensor = self._get_sensor_data()
        seg = self._to_numpy(sensor['segmentation']).squeeze(0).squeeze(-1)
        info['segmentation'] = seg

        info['env_name'] = self.env_name
        return info

    def _get_sensor_data(self):
        obs = self._ms_env.unwrapped.get_obs()
        return obs['sensor_data']['base_camera']

    def _apply_variations(self):
        try:
            scene = self._ms_env.unwrapped.scene
        except AttributeError:
            return

        vs = self.variation_space

        # Object colors
        if self._object_actors:
            colors = vs['object']['color'].value
            for i, actor in enumerate(self._object_actors):
                if i >= len(colors):
                    break
                color = colors[i]
                try:
                    import sapien
                    for comp in actor.components:
                        if isinstance(comp, sapien.render.RenderBodyComponent):
                            for shape in comp.render_shapes:
                                mat = shape.material
                                mat.base_color = np.append(color, 1.0).astype(np.float32)
                except Exception:
                    pass

        # Light intensity and color
        try:
            intensity = vs['light']['intensity'].value[0]
            light_color = vs['light']['color'].value
            for light in scene.get_all_lights():
                light.color = (light_color * intensity).astype(np.float32)
        except Exception:
            pass

    def _discover_object_actors(self):
        self._object_actors = []
        try:
            env_unwrapped = self._ms_env.unwrapped
            if hasattr(env_unwrapped, 'obj') and env_unwrapped.obj is not None:
                actors = env_unwrapped.obj
                if not isinstance(actors, (list, tuple)):
                    actors = [actors]
                self._object_actors = list(actors)
            elif hasattr(env_unwrapped, 'cubeA'):
                self._object_actors = [env_unwrapped.cubeA]
                if hasattr(env_unwrapped, 'cubeB'):
                    self._object_actors.append(env_unwrapped.cubeB)
        except Exception:
            pass

    @property
    def np_random_seed(self):
        return getattr(self, '_last_seed', 0)

    def reset(self, seed=None, options=None):
        self._last_seed = seed if seed is not None else 0
        options = options or {}

        swm_spaces.reset_variation_space(
            self.variation_space,
            seed=seed,
            options=options,
            default_variations=self._get_default_variations(),
        )

        ms_obs, ms_info = self._ms_env.reset(seed=seed)
        ms_obs = self._to_numpy(ms_obs)
        ms_info = self._to_numpy(ms_info)

        self._discover_object_actors()
        self._apply_variations()

        obs = self._build_obs(ms_obs)
        info = self._build_info(ms_obs, ms_info)
        return obs, info

    def step(self, action):
        action_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        ms_obs, reward, terminated, truncated, ms_info = self._ms_env.step(action_t)

        ms_obs = self._to_numpy(ms_obs)
        ms_info = self._to_numpy(ms_info)
        reward = float(self._to_numpy(reward).squeeze())
        # Never terminate on success — episodes run to max steps so the
        # oracle can execute post-task phases and subtask chaining.
        terminated = False
        truncated = bool(self._to_numpy(truncated).squeeze())

        obs = self._build_obs(ms_obs)
        info = self._build_info(ms_obs, ms_info)
        return obs, reward, terminated, truncated, info

    def render(self):
        sensor = self._get_sensor_data()
        rgb = self._to_numpy(sensor['rgb']).squeeze(0)
        return rgb

    def close(self):
        self._ms_env.close()

    def _get_default_variations(self):
        return ('object.start_position',)
