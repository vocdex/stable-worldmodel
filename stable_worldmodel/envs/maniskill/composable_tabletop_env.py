from dataclasses import dataclass, field

import numpy as np
import sapien

from stable_worldmodel.envs.maniskill.base_env import ManiSkillSWMEnv


COLOR_MAP = {
    'red': [1.0, 0.0, 0.0],
    'green': [0.0, 1.0, 0.0],
    'blue': [0.0, 0.0, 1.0],
}


@dataclass
class ObjectSpec:
    shape: str = 'cube'
    color: str | tuple = 'red'
    scale: float = 1.0

    @property
    def rgb(self):
        if isinstance(self.color, str):
            return COLOR_MAP[self.color]
        return list(self.color)


class ComposableTabletopSWMEnv(ManiSkillSWMEnv):
    """SWM wrapper for the composable tabletop environment.

    Accepts a list of ObjectSpec defining each object's shape, color, and
    scale. The inner ManiSkill env handles shape switching; this wrapper
    applies colors and extracts per-object privileged info.
    """

    def __init__(self, object_specs=None, height=224, width=224, **kwargs):
        kwargs.setdefault('control_mode', 'pd_ee_delta_pos')

        if object_specs is None:
            object_specs = [
                ObjectSpec(shape='cube', color='red'),
                ObjectSpec(shape='cube', color='green'),
            ]
        self._object_specs = object_specs

        super().__init__(
            ms_env_id='ComposableTabletop-Static-v1',
            height=height,
            width=width,
            **kwargs,
        )
        self.env_name = 'ComposableTabletop'
        self._num_objects = len(object_specs)
        self._default_object_colors = np.array([s.rgb for s in object_specs])
        self.variation_space = self._build_variation_space()

    def reset(self, seed=None, options=None):
        options = options or {}

        ms_options = {
            'object_configs': [
                {'shape': s.shape, 'scale': s.scale}
                for s in self._object_specs
            ],
        }

        self._last_seed = seed if seed is not None else 0

        from stable_worldmodel import spaces as swm_spaces
        swm_spaces.reset_variation_space(
            self.variation_space,
            seed=seed,
            options=options,
            default_variations=self._get_default_variations(),
        )

        ms_obs, ms_info = self._ms_env.reset(seed=seed, options=ms_options)

        # Let robot PD controller settle, then freeze all objects
        import torch
        zero_action = torch.zeros((1, self._ms_env.action_space.shape[-1]))
        for _ in range(30):
            self._ms_env.step(zero_action)
        uw = self._ms_env.unwrapped
        uw.elapsed_steps.zero_()
        for actor in uw._active:
            if actor is not None:
                actor.set_linear_velocity(torch.zeros((uw.num_envs, 3)))
                actor.set_angular_velocity(torch.zeros((uw.num_envs, 3)))
        ms_obs = uw.get_obs()

        ms_obs = self._to_numpy(ms_obs)
        ms_info = self._to_numpy(ms_info)

        self._discover_object_actors()
        self._apply_variations()

        obs = self._build_obs(ms_obs)
        info = self._build_info(ms_obs, ms_info)
        return obs, info

    def _discover_object_actors(self):
        uw = self._ms_env.unwrapped
        self._object_actors = [
            obj for obj in uw._active if obj is not None
        ]

    def _build_info(self, ms_obs, ms_info):
        info = {}

        # Extra keys from MS observation
        extra = ms_obs.get('extra', {})
        for k, v in extra.items():
            val = self._to_numpy(v)
            if val.ndim > 0:
                val = val.squeeze(0)
            info[f'extra/{k}'] = val

        # Per-object poses from active actors
        uw = self._ms_env.unwrapped
        for i in range(self._num_objects):
            actor = uw._active[i]
            if actor is not None:
                info[f'extra/obj_{i}_pose'] = self._to_numpy(
                    actor.pose.raw_pose
                ).squeeze(0).astype(np.float32)

        # Evaluation metrics
        eval_info = uw.evaluate()
        for k, v in eval_info.items():
            val = self._to_numpy(v)
            if val.ndim > 0:
                val = val.squeeze()
            info[f'eval/{k}'] = val
        info['success'] = bool(info.get('eval/success', False))

        # Per-object grasp detection
        agent = uw.agent
        for i in range(self._num_objects):
            actor = uw._active[i]
            if actor is not None:
                val = self._to_numpy(agent.is_grasping(actor))
                if val.ndim > 0:
                    val = val.squeeze()
                info[f'eval/is_obj_{i}_grasped'] = bool(val)

        # Segmentation
        sensor = self._get_sensor_data()
        seg = self._to_numpy(sensor['segmentation']).squeeze(0).squeeze(-1)
        info['segmentation'] = seg

        info['env_name'] = self.env_name
        return info

    def _apply_variations(self):
        # Skip super — we handle colors directly via specs
        from stable_worldmodel.envs.maniskill.composable_tabletop_ms_env import (
            _set_actor_visible,
        )
        uw = self._ms_env.unwrapped
        for i, spec in enumerate(self._object_specs):
            actor = uw._active[i]
            if actor is None:
                continue
            rgba = np.array([*spec.rgb, 1.0], dtype=np.float32)
            _set_actor_visible(actor, True)
            for obj in actor._objs:
                for comp in obj.components:
                    if isinstance(comp, sapien.render.RenderBodyComponent):
                        for rs in comp.render_shapes:
                            rs.material.base_color = rgba
