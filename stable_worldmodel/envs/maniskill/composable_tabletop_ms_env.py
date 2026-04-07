from typing import Any, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

SHAPES = ('cube', 'sphere', 'peg')

# Exact dimensions from source ManiSkill envs
CUBE_HALF_SIZE = 0.02             # from StackCube-v1
SPHERE_RADIUS = 0.035             # from RollBall-v1
PEG_LENGTH = 0.12                 # from PokeCube-v1 (half_length param)
PEG_WIDTH = 0.025                 # from PokeCube-v1 (half_width param)

NUM_SLOTS = 3
BANISH_Z = -10.0

_DEFAULT_COLOR = [0.5, 0.5, 0.5, 1.0]


def _rest_z(shape):
    """Height of object center when resting on the table."""
    if shape == 'cube':
        return CUBE_HALF_SIZE
    if shape == 'sphere':
        return SPHERE_RADIUS
    if shape == 'peg':
        return PEG_WIDTH  # resting on its side
    return 0.02


def _placement_radius(shape):
    """Collision radius for placement sampling (generous to avoid overlaps)."""
    if shape == 'sphere':
        return SPHERE_RADIUS + 0.04
    if shape == 'peg':
        return PEG_LENGTH + 0.04
    return CUBE_HALF_SIZE * 1.414 + 0.03


def _set_actor_visible(actor, visible):
    """Show or hide an actor's render components."""
    for obj in actor._objs:
        for comp in obj.components:
            if isinstance(comp, sapien.render.RenderBodyComponent):
                comp.visibility = 1.0 if visible else 0.0


class ComposableTabletopMSEnv(BaseEnv):
    """ManiSkill3 tabletop env with configurable object shapes.

    Pre-creates one actor per shape per object slot (3 slots x 3 shapes = 9).
    Shapes use exact dimensions from source ManiSkill environments:
      - cube: 4cm block (StackCube-v1)
      - sphere: 7cm diameter ball (RollBall-v1)
      - peg: 24cm x 5cm two-color bar (PokeCube-v1)

    At episode init, the desired shape is placed on the table. Inactive
    variants are banished below the scene and hidden from rendering.
    """

    SUPPORTED_ROBOTS = ['panda']
    agent: Panda

    def __init__(self, *args, robot_uids='panda', robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig('base_camera', pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig('render_camera', pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise,
        )
        self.table_scene.build()

        # Invisible collision walls to keep objects in the workspace
        wall_height = 0.10
        wall_thick = 0.005
        extent = 0.30  # workspace boundary
        walls = [
            ([extent, 0, wall_height], [wall_thick, extent, wall_height]),   # +x
            ([-extent, 0, wall_height], [wall_thick, extent, wall_height]),  # -x
            ([0, extent, wall_height], [extent, wall_thick, wall_height]),   # +y
            ([0, -extent, wall_height], [extent, wall_thick, wall_height]),  # -y
        ]
        for i, (pos, half_size) in enumerate(walls):
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=half_size)
            # No visual — invisible walls
            builder.initial_pose = sapien.Pose(p=pos)
            builder.build_static(name=f'wall_{i}')

        self._variants = {}
        for i in range(NUM_SLOTS):
            # Cube: exact StackCube-v1 dimensions
            self._variants[(i, 'cube')] = actors.build_cube(
                self.scene,
                half_size=CUBE_HALF_SIZE,
                color=_DEFAULT_COLOR,
                name=f'obj_{i}_cube',
                initial_pose=sapien.Pose(p=[0, 0, BANISH_Z]),
            )
            # Sphere: exact RollBall-v1 dimensions
            self._variants[(i, 'sphere')] = actors.build_sphere(
                self.scene,
                radius=SPHERE_RADIUS,
                color=_DEFAULT_COLOR,
                name=f'obj_{i}_sphere',
                initial_pose=sapien.Pose(p=[0, 0, BANISH_Z]),
            )
            # Peg: exact PokeCube-v1 dimensions (build_twocolor_peg)
            self._variants[(i, 'peg')] = actors.build_twocolor_peg(
                self.scene,
                length=PEG_LENGTH,
                width=PEG_WIDTH,
                color_1=_DEFAULT_COLOR,
                color_2=_DEFAULT_COLOR,
                name=f'obj_{i}_peg',
                initial_pose=sapien.Pose(p=[0, 0, BANISH_Z]),
            )

        # Hide all variants initially
        for actor in self._variants.values():
            _set_actor_visible(actor, False)

        self._active = [None] * NUM_SLOTS
        self._num_active = 0

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            configs = options.get('object_configs', [{'shape': 'cube'}])
            self._num_active = len(configs)

            # Banish and hide all variants
            banish = Pose.create_from_pq(
                p=torch.tensor([[0, 0, BANISH_Z]]).expand(b, -1),
            )
            for actor in self._variants.values():
                actor.set_pose(banish)
                _set_actor_visible(actor, False)

            # Place and show active shapes
            region = [[-0.08, -0.12], [0.08, 0.12]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device,
            )

            for i, cfg in enumerate(configs):
                shape = cfg.get('shape', 'cube')
                actor = self._variants[(i, shape)]
                self._active[i] = actor

                radius = _placement_radius(shape)
                xy = sampler.sample(radius, 100, verbose=False)
                z = _rest_z(shape)
                xyz = torch.zeros((b, 3))
                xyz[:, :2] = xy
                xyz[:, 2] = z

                # Cubes get random z-rotation; spheres/pegs stay fixed
                # (peg must keep long axis along x so gripper can grasp the narrow side)
                if shape == 'cube':
                    qs = randomization.random_quaternions(
                        b, lock_x=True, lock_y=True, lock_z=False,
                    )
                else:
                    qs = torch.tensor([[1, 0, 0, 0]]).float().expand(b, -1)
                actor.set_pose(Pose.create_from_pq(p=xyz, q=qs))
                actor.set_linear_velocity(torch.zeros((b, 3)))
                actor.set_angular_velocity(torch.zeros((b, 3)))
                _set_actor_visible(actor, True)

            for i in range(len(configs), NUM_SLOTS):
                self._active[i] = None

    def evaluate(self):
        results = {
            'success': torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        }
        for i in range(self._num_active):
            actor = self._active[i]
            if actor is not None:
                results[f'is_obj_{i}_grasped'] = self.agent.is_grasping(actor)
        return results

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if 'state' in self.obs_mode:
            for i in range(self._num_active):
                actor = self._active[i]
                if actor is not None:
                    obs[f'obj_{i}_pose'] = actor.pose.raw_pose
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info)
