import sapien

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.envs.tasks.tabletop.stack_pyramid import StackPyramidEnv

# Move camera closer to the table (10% closer) for tighter framing of the workspace
_CAMERA_DISTANCE_SCALE = 1.0


class StaticCameraMixin:
    """Override sensor config to use a closer camera for tighter framing
    of the tabletop workspace."""

    @property
    def _default_sensor_configs(self):
        human_cfg = super()._default_human_render_camera_configs
        if isinstance(human_cfg, list):
            target_cfg = human_cfg[0]
        else:
            target_cfg = human_cfg

        import numpy as np
        p = np.asarray(target_cfg.pose.p, dtype=np.float32).flatten() * _CAMERA_DISTANCE_SCALE
        q = np.asarray(target_cfg.pose.q, dtype=np.float32).flatten()
        closer_pose = sapien.Pose(p, q)

        return [
            CameraConfig(
                'base_camera',
                pose=closer_pose,
                width=512,
                height=512,
                fov=target_cfg.fov,
                near=target_cfg.near,
                far=target_cfg.far,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        human_cfg = super()._default_human_render_camera_configs
        if isinstance(human_cfg, list):
            target_cfg = human_cfg[0]
        else:
            target_cfg = human_cfg
        target_cfg.width = 512
        target_cfg.height = 512
        return target_cfg


@register_env('PickCube-Static-v1', max_episode_steps=500)
class PickCubeStaticEnv(StaticCameraMixin, PickCubeEnv):
    pass


@register_env('StackCube-Static-v1', max_episode_steps=500)
class StackCubeStaticEnv(StaticCameraMixin, StackCubeEnv):
    pass


@register_env('StackPyramid-Static-v1', max_episode_steps=500)
class StackPyramidStaticEnv(StaticCameraMixin, StackPyramidEnv):
    pass
