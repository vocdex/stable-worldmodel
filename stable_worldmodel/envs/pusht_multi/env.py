"""Multi-object PushT environment for compositional generalization studies.

Differs from `swm/PushT-v1`:
  - N named objects (canonical identities A..F, see `objects.py`) with stable
    identity↔segmentation-label mapping.
  - Each enabled object has its own goal pose (x, y, θ); success requires all
    enabled objects to be within tolerance of their respective goals.
  - Per-identity physics (mass, friction) plumbed in from `OBJECT_LIBRARY`.
  - `space.damping = 0.85` (un-contacted objects come to rest).
  - `info` exposes per-object pose/goal_pose/enabled with fixed-shape keys
    (NaN sentinel for disabled objects), plus a per-pixel `segmentation` label
    map and its goal-frame counterpart `goal_segmentation`.

The existing `swm/PushT-v1` env and `WeakPolicy` are not modified.
"""

from __future__ import annotations

import cv2
import gymnasium as gym
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from gymnasium import spaces
from pymunk.vec2d import Vec2d

from stable_worldmodel import spaces as swm_spaces

from ..utils import DrawOptions
from .objects import (
    LABEL_AGENT,
    LABEL_BG,
    OBJECT_LIBRARY,
    ObjectSpec,
    agent_bounding_radius,
)


# Window is the pymunk simulation canvas (pre-resize). Same as PushT.
WINDOW_SIZE = 512


# ---------------------------------------------------------------------------
# Shape constructors. Module-level helpers parameterised by mass and friction.
# Mirror the polygon definitions in pusht/env.py but accept per-identity
# physics so that A, B, C, D, E, F can have genuinely distinct dynamics.
# ---------------------------------------------------------------------------


def _add_kinematic_circle(
    space: pymunk.Space, position, scale: float, color: str = 'RoyalBlue'
):
    base_radius = 0.375
    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    body.position = position
    body.friction = 1
    shape = pymunk.Circle(body, base_radius * scale)
    shape.color = pygame.Color(color)
    space.add(body, shape)
    return body


def _add_dynamic_circle(
    space: pymunk.Space, position, angle, scale, color, mass, friction
):
    radius = 0.375 * scale
    moment = pymunk.moment_for_circle(mass, 0, radius)
    body = pymunk.Body(mass, moment)
    body.position = position
    body.angle = angle
    shape = pymunk.Circle(body, radius)
    shape.color = pygame.Color(color)
    shape.friction = friction
    space.add(body, shape)
    return body


def _add_box(space, position, angle, scale, color, mass, friction):
    verts = [(-scale, -scale), (-scale, scale), (scale, scale), (scale, -scale)]
    moment = pymunk.moment_for_poly(mass, verts)
    body = pymunk.Body(mass, moment)
    body.angle = angle
    body.position = position
    shape = pymunk.Poly(body, verts)
    shape.color = pygame.Color(color)
    shape.friction = friction
    space.add(body, shape)
    return body


def _add_tee(space, position, angle, scale, color, mass, friction):
    length = 4
    v1 = [
        (-length * scale / 2, scale),
        (length * scale / 2, scale),
        (length * scale / 2, 0),
        (-length * scale / 2, 0),
    ]
    v2 = [
        (-scale / 2, scale),
        (-scale / 2, length * scale),
        (scale / 2, length * scale),
        (scale / 2, scale),
    ]
    moment = pymunk.moment_for_poly(mass, v1) + pymunk.moment_for_poly(mass, v2)
    body = pymunk.Body(mass, moment)
    s1 = pymunk.Poly(body, v1)
    s2 = pymunk.Poly(body, v2)
    for s in (s1, s2):
        s.color = pygame.Color(color)
        s.friction = friction
    # Order is load-bearing: pymunk rotates the body around its
    # center_of_gravity, and `body.angle` reads body.position relative to
    # the rotated origin. With a non-trivial COG, setting position before
    # angle makes body.position read back at a different world location
    # than was passed in. Set COG → angle → position so the body ends up
    # exactly at `position` with `angle`.
    body.center_of_gravity = (s1.center_of_gravity + s2.center_of_gravity) / 2
    body.angle = angle
    body.position = position
    space.add(body, s1, s2)
    return body


def _add_small_tee(space, position, angle, scale, color, mass, friction):
    v1 = [
        (-3 * scale / 2, scale),
        (3 * scale / 2, scale),
        (3 * scale / 2, 0),
        (-3 * scale / 2, 0),
    ]
    v2 = [
        (-scale / 2, scale),
        (-scale / 2, 2 * scale),
        (scale / 2, 2 * scale),
        (scale / 2, scale),
    ]
    moment = pymunk.moment_for_poly(mass, v1) + pymunk.moment_for_poly(mass, v2)
    body = pymunk.Body(mass, moment)
    s1 = pymunk.Poly(body, v1)
    s2 = pymunk.Poly(body, v2)
    for s in (s1, s2):
        s.color = pygame.Color(color)
        s.friction = friction
    body.center_of_gravity = (s1.center_of_gravity + s2.center_of_gravity) / 2
    body.angle = angle
    body.position = position
    space.add(body, s1, s2)
    return body


def _add_plus(space, position, angle, scale, color, mass, friction):
    v1 = [
        (-3 * scale / 2, scale / 2),
        (3 * scale / 2, scale / 2),
        (3 * scale / 2, -scale / 2),
        (-3 * scale / 2, -scale / 2),
    ]
    v2 = [
        (-scale / 2, scale / 2),
        (-scale / 2, 3 * scale / 2),
        (scale / 2, 3 * scale / 2),
        (scale / 2, scale / 2),
    ]
    v3 = [
        (-scale / 2, -scale / 2),
        (-scale / 2, -3 * scale / 2),
        (scale / 2, -3 * scale / 2),
        (scale / 2, -scale / 2),
    ]
    moment = sum(pymunk.moment_for_poly(mass, v) for v in (v1, v2, v3))
    body = pymunk.Body(mass, moment)
    shapes = [pymunk.Poly(body, v) for v in (v1, v2, v3)]
    for s in shapes:
        s.color = pygame.Color(color)
        s.friction = friction
    body.center_of_gravity = sum(
        (s.center_of_gravity for s in shapes), Vec2d(0, 0)
    ) / 3
    body.angle = angle
    body.position = position
    space.add(body, *shapes)
    return body


def _add_L(space, position, angle, scale, color, mass, friction):
    length = 2
    v1 = [
        (0, 0),
        (0, scale * length),
        (scale * length / 2, scale * length),
        (scale * length / 2, 0),
    ]
    v2 = [
        (0, 0),
        (scale * length, 0),
        (scale * length, -scale * length / 2),
        (0, -scale * length / 2),
    ]
    moment = pymunk.moment_for_poly(mass, v1) + pymunk.moment_for_poly(mass, v2)
    body = pymunk.Body(mass, moment)
    s1, s2 = pymunk.Poly(body, v1), pymunk.Poly(body, v2)
    for s in (s1, s2):
        s.color = pygame.Color(color)
        s.friction = friction
    body.center_of_gravity = (s1.center_of_gravity + s2.center_of_gravity) / 2
    body.angle = angle
    body.position = position
    space.add(body, s1, s2)
    return body


def _add_Z(space, position, angle, scale, color, mass, friction):
    length = 2
    v1 = [
        (0, 0),
        (0, length * scale / 2),
        (length * scale, length * scale / 2),
        (length * scale, 0),
    ]
    v2 = [
        (-length * scale / 2, 0),
        (length * scale / 2, 0),
        (length * scale / 2, -length * scale / 2),
        (-length * scale / 2, -length * scale / 2),
    ]
    moment = pymunk.moment_for_poly(mass, v1) + pymunk.moment_for_poly(mass, v2)
    body = pymunk.Body(mass, moment)
    s1, s2 = pymunk.Poly(body, v1), pymunk.Poly(body, v2)
    for s in (s1, s2):
        s.color = pygame.Color(color)
        s.friction = friction
    body.center_of_gravity = (s1.center_of_gravity + s2.center_of_gravity) / 2
    body.angle = angle
    body.position = position
    space.add(body, s1, s2)
    return body


def _add_I(space, position, angle, scale, color, mass, friction):
    verts = [
        (-scale / 2, -scale * 2),
        (-scale / 2, scale * 2),
        (scale / 2, scale * 2),
        (scale / 2, -scale * 2),
    ]
    moment = pymunk.moment_for_poly(mass, verts)
    body = pymunk.Body(mass, moment)
    shape = pymunk.Poly(body, verts)
    shape.color = pygame.Color(color)
    shape.friction = friction
    body.center_of_gravity = shape.center_of_gravity
    body.angle = angle
    body.position = position
    space.add(body, shape)
    return body


_SHAPE_DISPATCH = {
    'T': _add_tee,
    'small_tee': _add_small_tee,
    'L': _add_L,
    'Z': _add_Z,
    'square': _add_box,
    'I': _add_I,
    '+': _add_plus,
    'o': _add_dynamic_circle,
}


def _add_object_body(
    space, spec_shape, position, angle, scale, color, mass, friction
):
    """Dispatch to the right per-shape constructor."""
    fn = _SHAPE_DISPATCH.get(spec_shape)
    if fn is None:
        raise ValueError(f'Unknown shape type: {spec_shape}')
    return fn(space, position, angle, scale, color, mass, friction)


# ---------------------------------------------------------------------------
# Variation-space construction
# ---------------------------------------------------------------------------


def _make_agent_subspace(ws: int) -> swm_spaces.Dict:
    # Agent is a kinematic circle, radius 0.375*scale. At the max scale of
    # 60 the radius is only 22.5 px, so its safe spawn range is wide.
    sp_low, sp_high = _safe_spawn_range(agent_bounding_radius(60.0), ws)
    return swm_spaces.Dict(
        {
            'color': swm_spaces.RGBBox(
                init_value=np.array(pygame.Color('RoyalBlue')[:3], dtype=np.uint8)
            ),
            'scale': swm_spaces.Box(
                low=20, high=60, init_value=40, shape=(), dtype=np.float32
            ),
            'angle': swm_spaces.Box(
                low=-2 * np.pi, high=2 * np.pi, init_value=0.0,
                shape=(), dtype=np.float64,
            ),
            'start_position': swm_spaces.Box(
                low=sp_low, high=sp_high,
                init_value=np.array((256, 400), dtype=np.float64),
                shape=(2,), dtype=np.float64,
            ),
            'velocity': swm_spaces.Box(
                low=0, high=ws,
                init_value=np.array((0.0, 0.0), dtype=np.float64),
                shape=(2,), dtype=np.float64,
            ),
        }
    )


def _safe_spawn_range(bounding_radius: float, ws: int, margin: float = 10.0) -> tuple[float, float]:
    """Box bounds for a body's center so its bounding circle starts
    fully inside the rendered canvas `[0, ws]`.

    The env has no walls — once an episode runs, the policy is free to
    push objects off-frame — but we want the *initial* pose to be
    visible so the encoder and planner have a well-defined starting
    scene.
    """
    low = bounding_radius + margin
    high = ws - bounding_radius - margin
    if low >= high:
        # Object is too big for the workspace at this scale; collapse to
        # a single feasible point (the geometric center).
        c = (low + high) / 2
        return (c, c)
    return (float(low), float(high))


def _make_object_subspace(spec: ObjectSpec, enabled_default: bool, ws: int) -> swm_spaces.Dict:
    # Per-shape spawn box so the bounding circle starts inside the
    # canvas (purely to make the initial scene visible — no walls).
    sp_low, sp_high = _safe_spawn_range(spec.bounding_radius, ws)
    center = (sp_low + sp_high) / 2
    return swm_spaces.Dict(
        {
            'enabled': swm_spaces.Discrete(2, init_value=int(enabled_default)),
            'color': swm_spaces.RGBBox(
                init_value=np.array(pygame.Color(spec.color)[:3], dtype=np.uint8)
            ),
            'scale': swm_spaces.Box(
                low=20, high=60, init_value=float(spec.scale),
                shape=(), dtype=np.float32,
            ),
            'angle': swm_spaces.Box(
                low=-2 * np.pi, high=2 * np.pi, init_value=0.0,
                shape=(), dtype=np.float64,
            ),
            'start_position': swm_spaces.Box(
                low=sp_low, high=sp_high,
                init_value=np.array((center, center), dtype=np.float64),
                shape=(2,), dtype=np.float64,
            ),
            'goal_position': swm_spaces.Box(
                low=sp_low, high=sp_high,
                init_value=np.array((center, center), dtype=np.float64),
                shape=(2,), dtype=np.float64,
            ),
            'goal_angle': swm_spaces.Box(
                low=-2 * np.pi, high=2 * np.pi, init_value=np.pi / 4,
                shape=(), dtype=np.float64,
            ),
        }
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class PushTMulti(gym.Env):
    """Multi-object PushT.

    Args:
        objects: Tuple of canonical identity names available in this env.
            Must be a subset of `OBJECT_LIBRARY` keys.
        enabled_objects: Default set of *active* objects (rest are present
            in the variation space but `enabled=0` so they're not added to
            the pymunk scene). Can be overridden per-episode via the
            variation-values reset option.
        success_pos_tol: Per-object position tolerance (pixels).
        success_angle_tol: Per-object angle tolerance (radians); ignored
            for objects with `has_orientation=False`.
        resolution: Output image side length (square).
        relative: If True, agent action is a delta added to agent position.
        render_mode: 'rgb_array' or 'human'.
        init_value: Optional variation-space init override.
    """

    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10,
        'render_fps': 10,
    }
    reward_range = (-np.inf, 0.0)

    def __init__(
        self,
        objects: tuple[str, ...] = ('A', 'B', 'C'),
        enabled_objects: tuple[str, ...] | None = None,
        subset_mode: str = 'fixed',
        subset_k_range: tuple[int, int] = (1, 6),
        success_pos_tol: float = 20.0,
        success_angle_tol: float = np.pi / 9,
        resolution: int = 224,
        relative: bool = True,
        render_mode: str = 'rgb_array',
        init_value=None,
    ) -> None:
        # Validate identities and freeze the canonical ordering.
        for oid in objects:
            if oid not in OBJECT_LIBRARY:
                raise ValueError(
                    f'Unknown object id {oid!r}, expected one of {list(OBJECT_LIBRARY)}'
                )
        if len(set(objects)) != len(objects):
            raise ValueError(f'objects must be unique, got {objects}')
        self.object_ids: tuple[str, ...] = tuple(objects)
        self.specs: dict[str, ObjectSpec] = {
            oid: OBJECT_LIBRARY[oid] for oid in self.object_ids
        }

        # Default-enabled set: if user passes None, all available identities
        # are enabled. The variation space lets us override per-episode.
        if enabled_objects is None:
            enabled_objects = self.object_ids
        for oid in enabled_objects:
            if oid not in self.object_ids:
                raise ValueError(
                    f'enabled_objects contains {oid!r} not in objects={self.object_ids}'
                )
        self._default_enabled: set[str] = set(enabled_objects)

        # Per-episode subset randomization. 'fixed' (default) keeps the
        # existing behavior: enabled set comes from the constructor /
        # reset-options. 'random' samples k ~ U[k_min, k_max] then picks
        # a uniform subset of size k from `objects` on every reset,
        # overriding any obj.<oid>.enabled values. Used to build a single
        # SC training dataset that contains scenes with varying k and
        # varying identity combinations.
        if subset_mode not in ('fixed', 'random'):
            raise ValueError(
                f'subset_mode must be "fixed" or "random", got {subset_mode!r}'
            )
        self.subset_mode = subset_mode
        k_min, k_max = subset_k_range
        # Clamp k_max to the number of available identities so callers can
        # leave the default (1, 6) and have it adapt to `objects` of any
        # size without hand-tuning.
        k_max = min(int(k_max), len(self.object_ids))
        k_min = max(1, int(k_min))
        if not (1 <= k_min <= k_max <= len(self.object_ids)):
            raise ValueError(
                f'subset_k_range {subset_k_range} resolves to invalid '
                f'(k_min={k_min}, k_max={k_max}) for objects={self.object_ids}'
            )
        self.subset_k_range = (k_min, k_max)

        self.window_size = ws = WINDOW_SIZE
        self.render_size = resolution
        self.relative = relative
        self.action_scale = 100

        # Physics
        self.control_hz = self.metadata['render_fps']
        self.k_p, self.k_v = 100, 20
        self.dt = 0.01
        # Pymunk damping = fraction of velocity *retained* per second of sim
        # (formula: v_new = v * damping**dt). With dt=0.01 and 10 sub-steps
        # per env-step, damping=0.1 → ~20% velocity loss per env-step, which
        # makes bodies come to rest within ~5 env-steps of leaving contact —
        # the tabletop-friction feel we want. Original PushT uses 0 (no
        # damping, objects coast forever) but it doesn't show because the
        # agent stays in contact with the single block. With several lighter
        # objects and a fast-switching focus policy that breaks down.
        self._damping = 0.1

        # Success tolerances
        self.success_pos_tol = float(success_pos_tol)
        self.success_angle_tol = float(success_angle_tol)

        # --- gym spaces ---
        # state layout: [agent_xy(2)] + [per-object x,y,theta(3) * N] + [agent_vxvy(2)]
        n_obj = len(self.object_ids)
        state_dim = 2 + 3 * n_obj + 2
        proprio_dim = 4  # agent xy + agent vxvy
        low_state = np.full(state_dim, -np.inf, dtype=np.float64)
        high_state = np.full(state_dim, np.inf, dtype=np.float64)
        self.observation_space = spaces.Dict(
            {
                'proprio': spaces.Box(
                    low=np.array([0, 0, -ws, -ws]),
                    high=np.array([ws, ws, ws, ws]),
                    dtype=np.float64,
                ),
                'state': spaces.Box(
                    low=low_state, high=high_state, dtype=np.float64
                ),
            }
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # --- variation space ---
        # Per-object subspaces live under a single top-level 'obj' Dict, so
        # the swm dot-path traversal (`get_in`) resolves
        # `obj.A.start_position` → space['obj']['A']['start_position'].
        obj_root = swm_spaces.Dict(
            {
                oid: _make_object_subspace(
                    self.specs[oid],
                    enabled_default=(oid in self._default_enabled),
                    ws=ws,
                )
                for oid in self.object_ids
            },
            sampling_order=list(self.object_ids),
        )

        self.variation_space = swm_spaces.Dict(
            {
                'agent': _make_agent_subspace(ws),
                'obj': obj_root,
                'background': swm_spaces.Dict(
                    {
                        'color': swm_spaces.RGBBox(
                            init_value=np.array([255, 255, 255], dtype=np.uint8)
                        )
                    }
                ),
                'rendering': swm_spaces.Dict(
                    # Default OFF for PushTMulti: drawing a faded goal-pose
                    # outline into the live RGB would create static
                    # object-shaped "ghosts" that an object-centric encoder
                    # (e.g. SlotContrast) would bind slots to — they have no
                    # corresponding segmentation label, never move, and would
                    # eat slot capacity. The goal is delivered separately via
                    # info['goal']; re-enable here only if a downstream task
                    # genuinely needs the goal blended into the observation.
                    {'render_goal': swm_spaces.Discrete(2, init_value=0)}
                ),
            },
            sampling_order=['background', 'obj', 'agent', 'rendering'],
        )
        if init_value is not None:
            self.variation_space.set_init_value(init_value)

        # Default variations applied on every reset (everything that gives
        # the dataset its diversity). Per-object enabled flags are *not*
        # in this list — those come from reset options when collecting
        # subset-specific datasets.
        self._default_variations: tuple[str, ...] = (
            'agent.start_position',
            *[f'obj.{oid}.start_position' for oid in self.object_ids],
            *[f'obj.{oid}.angle' for oid in self.object_ids],
            *[f'obj.{oid}.goal_position' for oid in self.object_ids],
            *[f'obj.{oid}.goal_angle' for oid in self.object_ids],
        )

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.screen = None

        self.space: pymunk.Space | None = None
        self.agent = None
        self.bodies: dict[str, pymunk.Body] = {}
        self.body_labels: dict[int, int] = {}  # id(body) -> seg label
        self.enabled: set[str] = set()
        self.goal_pose: dict[str, np.ndarray] = {}  # oid -> (3,)
        self.goal_state: np.ndarray | None = None

        self.render_buffer = None
        self.latest_action = None
        self.n_contact_points = 0
        self.env_name = 'PushTMulti'
        self._goal_image = None
        self._goal_seg = None

    # ------------------------------------------------------------------
    # gym.Env interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.rng = np.random.default_rng(seed)
        options = options or {}

        swm_spaces.reset_variation_space(
            self.variation_space, seed, options, self._default_variations,
        )

        # In 'random' subset mode the enabled flags from the variation
        # values are deliberately overridden — sample k uniformly from
        # the configured range, then a uniform subset of that size from
        # the available identity pool. Applied after reset_variation_space
        # so per-episode `enabled` is the dominant signal (not whatever
        # the variation space happened to sample).
        if self.subset_mode == 'random':
            k_min, k_max = self.subset_k_range
            k = int(self.rng.integers(k_min, k_max + 1))
            chosen = self.rng.choice(
                list(self.object_ids), size=k, replace=False
            )
            chosen_set = set(chosen.tolist())
            for oid in self.object_ids:
                flag = int(oid in chosen_set)
                self.variation_space['obj'][oid]['enabled'].set_value(flag)

        # Resolve the active set of enabled objects after the variation
        # space has been (re-)sampled. The 'enabled' Discrete defaults to
        # the constructor-level enabled_objects; callers can override via
        # variation_values={'obj.X.enabled': 0}.
        self.enabled = {
            oid for oid in self.object_ids
            if int(self.variation_space['obj'][oid]['enabled'].value) == 1
        }
        if len(self.enabled) == 0:
            raise RuntimeError(
                'PushTMulti.reset(): at least one object must be enabled.'
            )

        # Enforce non-overlap on per-object start positions and goal
        # positions independently, by resampling overlapping objects.
        # The agent's start position participates in the start-pose check
        # (so the pusher doesn't spawn inside an object). Goal positions
        # are object-only (the agent has no goal pose to satisfy).
        self._resolve_overlap('start_position', include_agent=True)
        self._resolve_overlap('goal_position', include_agent=False)

        # Build the pymunk scene from the resolved variation values.
        self._setup_scene()

        # Compute the start-state and goal-state arrays. Bodies are
        # already placed at the sampled start poses by `_setup_scene`, so
        # we only need to *re-apply* state if the caller is overriding it
        # via reset options — otherwise calling _apply_state would just
        # re-place at the same poses and then step physics, generating
        # cached contact info that pollutes the goal-frame render.
        cur_state = self._compose_full_state(at_goal=False)
        goal_state = self._compose_full_state(at_goal=True)

        if 'state' in options:
            cur_state = np.asarray(options['state'], dtype=np.float64)
            self._apply_state(cur_state)
        if 'goal_state' in options:
            goal_state = np.asarray(options['goal_state'], dtype=np.float64)

        self.goal_state = goal_state
        self._goal_image, self._goal_seg = self._render_goal_frame(goal_state)

        observation = self._build_observation()
        info = self._get_info()
        return observation, info

    def step(self, action):
        self.n_contact_points = 0
        n_steps = int(1 / (self.dt * self.control_hz))
        self.latest_action = action
        if self.relative:
            action = self.agent.position + np.asarray(action) * self.action_scale
        # Clamp the PD target to the rendered canvas so the agent
        # itself stays on-screen — the encoder/planner need a visible
        # pusher to track. Objects may drift off-frame under hard
        # pushes (no walls), but the agent shouldn't lead them there.
        agent_r = agent_bounding_radius(
            float(self.variation_space['agent']['scale'].value)
        )
        inner_low = agent_r
        inner_high = WINDOW_SIZE - agent_r
        action = np.clip(action, inner_low, inner_high)

        for _ in range(n_steps):
            acc = self.k_p * (action - self.agent.position) + self.k_v * (
                Vec2d(0, 0) - self.agent.velocity
            )
            self.agent.velocity += acc * self.dt
            self.space.step(self.dt)

        observation = self._build_observation()
        info = self._get_info()

        cur_state = observation['state']
        terminated, reward = self._joint_eval(cur_state)
        return observation, reward, terminated, False, info

    # ------------------------------------------------------------------
    # Variation helpers
    # ------------------------------------------------------------------

    def _resolve_overlap(
        self,
        key: str,
        include_agent: bool = False,
        per_entity_iters: int = 500,
        margin: float = 8.0,
    ) -> None:
        """Sequential non-overlap placement.

        Strategy: sort participating entities by bounding radius
        descending (place the biggest first), then for each entity
        resample its position until it satisfies the min-separation
        constraint against every already-placed entity. This converges
        in O(N) resampling rounds rather than the O(2^N)-ish behaviour
        of pure joint-rejection sampling — important once a scene has a
        T (radius ~121) plus several other large polygons.

        Min separation for any pair (i, j) is `r_i + r_j + margin`, where
        `r_*` is the rotation-invariant circumscribed-circle radius from
        `ObjectSpec.bounding_radius`. Conservative — bodies in some
        orientations could sit closer without colliding — but guarantees
        no overlap at *any* sampled angle.

        `key`:
          - 'start_position' — agent participates iff `include_agent=True`.
          - 'goal_position'  — agent does not have a goal pose.

        If a given entity can't be placed within `per_entity_iters` tries
        (highly cluttered scene), accept the last sample. Pymunk's first
        physics step will resolve any residual penetration.
        """
        # Build participants list with radii.
        entities: list[tuple[tuple[str, ...], float]] = []
        for oid in sorted(self.enabled):
            entities.append((('obj', oid), self.specs[oid].bounding_radius))
        if include_agent:
            agent_scale = float(self.variation_space['agent']['scale'].value)
            entities.append((('agent',), agent_bounding_radius(agent_scale)))
        if len(entities) < 2:
            return

        # Largest first — easier to find space for smaller ones around it.
        entities.sort(key=lambda e: e[1], reverse=True)

        def _position(path: tuple[str, ...]) -> np.ndarray:
            if path[0] == 'agent':
                return self.variation_space['agent']['start_position'].value
            return self.variation_space['obj'][path[1]][key].value

        def _resample(path: tuple[str, ...]) -> None:
            if path[0] == 'agent':
                self.variation_space['agent']['start_position'].sample()
            else:
                self.variation_space['obj'][path[1]][key].sample()

        placed: list[tuple[np.ndarray, float]] = []  # (position, radius)
        for path, r in entities:
            for _ in range(per_entity_iters):
                pos = _position(path)
                ok = True
                for prev_pos, prev_r in placed:
                    if np.linalg.norm(pos - prev_pos) < (r + prev_r + margin):
                        ok = False
                        break
                if ok:
                    placed.append((np.asarray(pos, dtype=np.float64).copy(), r))
                    break
                _resample(path)
            else:
                # per_entity_iters exhausted — accept whatever's there.
                placed.append((np.asarray(_position(path), dtype=np.float64).copy(), r))

    # ------------------------------------------------------------------
    # Scene construction / state
    # ------------------------------------------------------------------

    def _setup_scene(self) -> None:
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = self._damping
        self.render_buffer = []
        self.bodies = {}
        self.body_labels = {}

        # No walls. Agent is kinematic (walls don't affect it anyway), and
        # bounding the scene only created visual clutter for the encoder
        # and the illusion of containment that high-impulse CEM pushes
        # could break via tunneling. Goal-distance reward already penalises
        # off-frame drift.

        # Agent (kinematic circle, PD-controlled).
        agent_v = self.variation_space['agent']
        self.agent = _add_kinematic_circle(
            self.space,
            position=agent_v['start_position'].value.tolist(),
            scale=float(agent_v['scale'].value),
            color=tuple(int(c) for c in agent_v['color'].value),
        )
        self.body_labels[id(self.agent)] = LABEL_AGENT

        # Objects.
        self.goal_pose = {}
        for oid in self.object_ids:
            if oid not in self.enabled:
                continue
            spec = self.specs[oid]
            v = self.variation_space['obj'][oid]
            body = _add_object_body(
                self.space,
                spec_shape=spec.shape,
                position=v['start_position'].value.tolist(),
                angle=float(v['angle'].value),
                scale=float(v['scale'].value),
                color=tuple(int(c) for c in v['color'].value),
                mass=spec.mass,
                friction=spec.friction,
            )
            self.bodies[oid] = body
            self.body_labels[id(body)] = spec.label

            self.goal_pose[oid] = np.array(
                [
                    *v['goal_position'].value.tolist(),
                    float(v['goal_angle'].value),
                ],
                dtype=np.float64,
            )

        self.space.on_collision(0, 0, post_solve=self._handle_collision)
        self.n_contact_points = 0

    def _compose_full_state(self, at_goal: bool) -> np.ndarray:
        """Return a fixed-shape state vector regardless of which objects
        are enabled. Disabled object slots are filled with NaN.
        """
        agent_v = self.variation_space['agent']
        agent_pos = agent_v['start_position'].value.astype(np.float64)
        agent_vel = agent_v['velocity'].value.astype(np.float64)

        obj_block = np.full(3 * len(self.object_ids), np.nan, dtype=np.float64)
        for i, oid in enumerate(self.object_ids):
            if oid not in self.enabled:
                continue
            v = self.variation_space['obj'][oid]
            if at_goal:
                obj_block[3 * i:3 * i + 2] = v['goal_position'].value
                obj_block[3 * i + 2] = float(v['goal_angle'].value)
            else:
                obj_block[3 * i:3 * i + 2] = v['start_position'].value
                obj_block[3 * i + 2] = float(v['angle'].value)

        return np.concatenate([agent_pos, obj_block, agent_vel])

    def _apply_state(self, state: np.ndarray) -> None:
        """Set agent and bodies to the given fixed-shape state."""
        state = np.asarray(state, dtype=np.float64)
        self.agent.position = state[0:2].tolist()
        for i, oid in enumerate(self.object_ids):
            if oid not in self.enabled:
                continue
            body = self.bodies[oid]
            ox, oy, oa = state[2 + 3 * i:2 + 3 * i + 3]
            # Pymunk: set angle BEFORE position when bodies have a non-trivial
            # center_of_gravity (the T-block does). See PushT._set_goal_state.
            body.angle = float(oa)
            body.position = (float(ox), float(oy))
            body.velocity = (0.0, 0.0)
            body.angular_velocity = 0.0
        self.agent.velocity = state[-2:].tolist()
        # Step once so the constraint/contact solver picks up the new poses.
        self.space.step(self.dt)

    def _build_observation(self) -> dict:
        state = self._current_state()
        proprio = np.concatenate([state[0:2], state[-2:]])
        return {'proprio': proprio, 'state': state}

    def _current_state(self) -> np.ndarray:
        agent_pos = np.array(self.agent.position, dtype=np.float64)
        agent_vel = np.array(self.agent.velocity, dtype=np.float64)
        obj_block = np.full(3 * len(self.object_ids), np.nan, dtype=np.float64)
        for i, oid in enumerate(self.object_ids):
            if oid not in self.enabled:
                continue
            body = self.bodies[oid]
            obj_block[3 * i:3 * i + 2] = np.array(body.position)
            obj_block[3 * i + 2] = float(body.angle % (2 * np.pi))
        return np.concatenate([agent_pos, obj_block, agent_vel])

    # ------------------------------------------------------------------
    # Success / reward
    # ------------------------------------------------------------------

    def _joint_eval(self, cur_state: np.ndarray) -> tuple[bool, float]:
        """Compute joint success and a continuous reward.

        Reward = -sum over enabled objects of (pos_dist + angle_dist), so
        the planner has a smooth signal even before any object succeeds.
        Angle term is omitted for objects without orientation.
        """
        successes = []
        total_cost = 0.0
        for i, oid in enumerate(self.object_ids):
            if oid not in self.enabled:
                continue
            spec = self.specs[oid]
            cur_xy = cur_state[2 + 3 * i:2 + 3 * i + 2]
            cur_a = cur_state[2 + 3 * i + 2]
            goal_xy = self.goal_pose[oid][:2]
            goal_a = self.goal_pose[oid][2]

            pos_diff = float(np.linalg.norm(cur_xy - goal_xy))
            pos_ok = pos_diff < self.success_pos_tol
            total_cost += pos_diff
            if spec.has_orientation:
                ang_diff = float(abs(cur_a - goal_a) % (2 * np.pi))
                ang_diff = min(ang_diff, 2 * np.pi - ang_diff)
                ang_ok = ang_diff < self.success_angle_tol
                total_cost += ang_diff * 20.0  # rough unit scaling rad→px
            else:
                ang_ok = True
            successes.append(pos_ok and ang_ok)

        return bool(np.all(successes)), -total_cost

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def _get_info(self) -> dict:
        n_steps = int(1 / self.dt * self.control_hz)
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info: dict = {
            'env_name': self.env_name,
            'pos_agent': np.array(self.agent.position),
            'vel_agent': np.array(self.agent.velocity),
            'n_contacts': n_contact_points_per_step,
            'goal_state': (
                self.goal_state
                if self.goal_state is not None
                else np.zeros(2 + 3 * len(self.object_ids) + 2)
            ),
            'goal_proprio': (
                np.concatenate([self.goal_state[0:2], self.goal_state[-2:]])
                if self.goal_state is not None
                else np.zeros(4)
            ),
            'goal': self._goal_image if self._goal_image is not None else np.zeros(
                (self.render_size, self.render_size, 3), dtype=np.uint8
            ),
            'goal_segmentation': (
                self._goal_seg
                if self._goal_seg is not None
                else np.zeros((self.render_size, self.render_size), dtype=np.uint8)
            ),
            'segmentation': self._render_segmentation(),
        }
        # Fixed per-object info keys; NaN for disabled.
        for oid in self.object_ids:
            if oid in self.enabled and oid in self.bodies:
                body = self.bodies[oid]
                info[f'pose.{oid}'] = np.array(
                    [body.position.x, body.position.y, float(body.angle % (2 * np.pi))],
                    dtype=np.float64,
                )
                info[f'goal_pose.{oid}'] = self.goal_pose[oid].copy()
                info[f'enabled.{oid}'] = True
            else:
                info[f'pose.{oid}'] = np.full(3, np.nan, dtype=np.float64)
                info[f'goal_pose.{oid}'] = np.full(3, np.nan, dtype=np.float64)
                info[f'enabled.{oid}'] = False
        return info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        return self._render_frame(self.render_mode)

    def _render_frame(self, mode: str) -> np.ndarray:
        if self.window is None and mode == 'human':
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and mode == 'human':
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(self.variation_space['background']['color'].value)
        self.screen = canvas
        draw_options = DrawOptions(canvas)
        # Disable pymunk's debug overlays — by default `debug_draw` also
        # renders constraints and collision-point normals (short line
        # segments at each contact). For object-centric encoding those
        # would show up in both the live obs and the goal frame as
        # "random lines" the segmentation map doesn't account for.
        # Restrict to shape geometry only.
        draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES

        # Goal-pose outlines (optional, drawn first so they sit behind the
        # live bodies). One outline per enabled object, using its identity's
        # default color faded.
        render_goal = bool(
            self.variation_space['rendering']['render_goal'].value
        )
        if render_goal:
            self._draw_goal_outlines(canvas)

        # Recolor live bodies (variation may override default per-identity color).
        agent_color = tuple(
            int(c) for c in self.variation_space['agent']['color'].value
        )
        for s in self.agent.shapes:
            s.color = pygame.Color(*agent_color)
        for oid, body in self.bodies.items():
            color = tuple(
                int(c) for c in self.variation_space['obj'][oid]['color'].value
            )
            for s in body.shapes:
                s.color = pygame.Color(*color)

        self.space.debug_draw(draw_options)

        if mode == 'human':
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

        img = np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )
        img = cv2.resize(img, (self.render_size, self.render_size))
        return img

    def _draw_goal_outlines(self, canvas: pygame.Surface) -> None:
        """Draw a translucent outline per enabled object at its goal pose."""
        for oid in self.enabled:
            spec = self.specs[oid]
            goal = self.goal_pose[oid]
            v = self.variation_space['obj'][oid]
            color = tuple(int(c) for c in v['color'].value)
            faded = tuple(min(255, int(c * 0.4 + 153)) for c in color)
            # Build a ghost body at the goal pose just for vertex transform.
            ghost = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
            ghost.position = (float(goal[0]), float(goal[1]))
            ghost.angle = float(goal[2]) if spec.has_orientation else 0.0
            # The live body's shapes carry the right local vertex set; reuse them.
            if oid not in self.bodies:
                continue
            for shape in self.bodies[oid].shapes:
                if isinstance(shape, pymunk.Circle):
                    center = pymunk.pygame_util.to_pygame(
                        ghost.local_to_world(shape.offset), canvas
                    )
                    pygame.draw.circle(
                        canvas, faded,
                        (int(center[0]), int(center[1])),
                        int(shape.radius),
                    )
                elif isinstance(shape, pymunk.Poly):
                    pts = [
                        pymunk.pygame_util.to_pygame(
                            ghost.local_to_world(v_), canvas
                        )
                        for v_ in shape.get_vertices()
                    ]
                    pygame.draw.polygon(canvas, faded, pts)

    def _render_segmentation(self) -> np.ndarray:
        """Per-object segmentation label map at `render_size` resolution.

        Drawn analytically from pymunk shape geometry — no anti-aliasing,
        no color-decoding. Agent gets LABEL_AGENT, each enabled object
        gets its canonical `OBJECT_LIBRARY[oid].label`, everything else is
        LABEL_BG.
        """
        canvas = pygame.Surface(
            (self.window_size, self.window_size), depth=8
        )
        # 8-bit palette surfaces fill with a palette index; LABEL_BG = 0.
        canvas.fill(LABEL_BG)

        # Draw agent first so objects overlap it (objects "on top of" agent
        # is the right convention since the agent passes under T-corners).
        self._rasterize_body(canvas, self.agent, LABEL_AGENT)
        for oid in self.enabled:
            body = self.bodies.get(oid)
            if body is None:
                continue
            self._rasterize_body(canvas, body, self.specs[oid].label)

        arr = np.array(pygame.surfarray.pixels2d(canvas), dtype=np.uint8).T
        return cv2.resize(
            arr, (self.render_size, self.render_size),
            interpolation=cv2.INTER_NEAREST,
        )

    @staticmethod
    def _rasterize_body(canvas: pygame.Surface, body: pymunk.Body, label: int) -> None:
        for shape in body.shapes:
            if isinstance(shape, pymunk.Circle):
                center = body.local_to_world(shape.offset)
                pygame.draw.circle(
                    canvas, label,
                    (int(round(center.x)), int(round(center.y))),
                    int(round(shape.radius)),
                )
            elif isinstance(shape, pymunk.Poly):
                pts = [
                    (int(round(p.x)), int(round(p.y)))
                    for p in (body.local_to_world(v) for v in shape.get_vertices())
                ]
                if len(pts) >= 3:
                    pygame.draw.polygon(canvas, label, pts)

    # ------------------------------------------------------------------
    # Goal-frame rendering (the idempotent snapshot/restore dance)
    # ------------------------------------------------------------------

    def _render_goal_frame(
        self, goal_state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render the (rgb, segmentation) pair at the supplied goal_state.

        Discipline (cf. PushT._set_goal_state docstring):
          - snapshot all bodies' pose + velocity
          - set each body's angle BEFORE position (cog depends on angle)
          - reindex_shapes_for_body on every moved body
          - render
          - restore in reverse order

        We bypass `_apply_state` here because that helper advances physics
        by one `dt` (which we deliberately want to avoid: rendering the
        goal must not perturb the live simulation state).
        """
        snap = self._snapshot_bodies()

        # Place at goal pose (no physics step).
        gs = np.asarray(goal_state, dtype=np.float64)
        self.agent.angle = 0.0
        self.agent.position = (float(gs[0]), float(gs[1]))
        for i, oid in enumerate(self.object_ids):
            if oid not in self.enabled:
                continue
            body = self.bodies[oid]
            ox, oy, oa = gs[2 + 3 * i:2 + 3 * i + 3]
            body.angle = float(oa)
            body.position = (float(ox), float(oy))
        # Refresh cached shape transforms; without this, debug_draw and the
        # seg rasterizer use stale poses.
        self.space.reindex_shapes_for_body(self.agent)
        for body in self.bodies.values():
            self.space.reindex_shapes_for_body(body)

        # Render under the active variation. Temporarily disable goal-
        # outline drawing — at the goal pose itself, drawing a ghost would
        # be redundant and would distort the seg comparison.
        prev_render_goal = self.variation_space['rendering']['render_goal'].value
        self.variation_space['rendering']['render_goal'].set_value(0)
        rgb = self._render_frame(self.render_mode)
        seg = self._render_segmentation()
        self.variation_space['rendering']['render_goal'].set_value(
            prev_render_goal
        )

        self._restore_bodies(snap)
        self.space.reindex_shapes_for_body(self.agent)
        for body in self.bodies.values():
            self.space.reindex_shapes_for_body(body)
        return rgb, seg

    def _snapshot_bodies(self) -> dict:
        s = {
            'agent': (
                tuple(self.agent.position),
                tuple(self.agent.velocity),
                float(self.agent.angle),
            ),
            'objects': {},
        }
        for oid, body in self.bodies.items():
            s['objects'][oid] = (
                tuple(body.position),
                float(body.angle),
                tuple(body.velocity),
                float(body.angular_velocity),
            )
        return s

    def _restore_bodies(self, snap: dict) -> None:
        ap, av, aa = snap['agent']
        self.agent.angle = aa
        self.agent.position = ap
        self.agent.velocity = av
        for oid, (bp, ba, bv, baw) in snap['objects'].items():
            body = self.bodies[oid]
            body.angle = ba
            body.position = bp
            body.velocity = bv
            body.angular_velocity = baw

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _handle_collision(self, arbiter, space, data) -> None:
        self.n_contact_points += len(arbiter.contact_point_set.points)
