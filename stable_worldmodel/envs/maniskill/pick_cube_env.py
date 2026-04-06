import numpy as np

from stable_worldmodel.envs.maniskill.base_env import ManiSkillSWMEnv


class PickCubeEnv(ManiSkillSWMEnv):

    def __init__(self, height=224, width=224, **kwargs):
        kwargs.setdefault('control_mode', 'pd_ee_delta_pos')
        super().__init__(
            ms_env_id='PickCube-Static-v1',
            height=height,
            width=width,
            **kwargs,
        )
        self.env_name = 'PickCube'
        self._num_objects = 1
        self._default_object_colors = np.array([[1.0, 0.0, 0.0]])
        self.variation_space = self._build_variation_space()
