"""Collection policies for PushTMulti.

Two policies live here:

* `MultiObjectWeakPolicy` — random actions biased toward object focus
  points; useful for high-coverage off-policy data.
* `MultiObjectGoalPolicy` — privileged-info expert that drives each
  enabled object to its sampled goal pose with Gaussian action noise,
  analogous to the scripted PushT expert used to collect the 18k-traj
  DINO-WM dataset. Produces mostly-successful trajectories.
"""

from __future__ import annotations

import numpy as np

from stable_worldmodel.policy import BasePolicy


class MultiObjectWeakPolicy(BasePolicy):
    """Random-action policy biased to stay near a rotating focus point.

    Args:
        dist_constraint: half-side of the focus-centered square (pixels)
            inside which the sampled action is clipped.
        switch_every: number of env steps between focus reselections.
        p_wedge: probability of choosing focus = midpoint between two
            enabled objects (instead of a single object).
        seed: RNG seed.
    """

    def __init__(
        self,
        dist_constraint: float = 100.0,
        switch_every: int = 10,
        p_wedge: float = 0.3,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert dist_constraint > 0
        assert 0.0 <= p_wedge <= 1.0
        self.dist_constraint = float(dist_constraint)
        self.switch_every = int(switch_every)
        self.p_wedge = float(p_wedge)
        self.set_seed(seed)
        self._steps: np.ndarray | None = None
        self._focus_oid: list[tuple[str, ...] | None] = []

    def set_seed(self, seed: int | None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def set_env(self, env) -> None:
        self.env = env
        spec = getattr(env, 'spec', None)
        if spec is None:
            envs = getattr(env, 'envs', None)
            if envs:
                spec = envs[0].spec
        assert spec is not None and 'swm/PushTMulti' in spec.id, (
            f'MultiObjectWeakPolicy requires swm/PushTMulti, got {spec}'
        )

    def _envs(self):
        base = self.env.unwrapped
        return [e.unwrapped for e in base.envs] if hasattr(base, 'envs') else [base]

    def _ensure_state(self, n: int) -> None:
        if self._steps is None or len(self._steps) != n:
            self._steps = np.zeros(n, dtype=np.int64)
            self._focus_oid = [None] * n

    def _pick_focus(self, env) -> np.ndarray:
        """Return the (x, y) focus point for one env."""
        enabled = sorted(env.enabled)
        if len(enabled) == 0:
            # No objects to push — fall back to agent's current position.
            return np.array(env.agent.position, dtype=np.float64)
        if len(enabled) >= 2 and self.rng.random() < self.p_wedge:
            a, b = self.rng.choice(enabled, size=2, replace=False)
            pa = np.array(env.bodies[a].position, dtype=np.float64)
            pb = np.array(env.bodies[b].position, dtype=np.float64)
            return 0.5 * (pa + pb)
        oid = enabled[int(self.rng.integers(0, len(enabled)))]
        return np.array(env.bodies[oid].position, dtype=np.float64)

    def get_action(self, info_dict, **kwargs):
        assert hasattr(self, 'env'), 'Environment not set for the policy'
        envs = self._envs()
        self._ensure_state(len(envs))

        act_shape = self.env.action_space.shape
        actions = np.zeros(act_shape, dtype=np.float32)

        for i, env in enumerate(envs):
            if self._steps[i] % self.switch_every == 0 or self._focus_oid[i] is None:
                focus = self._pick_focus(env)
                self._focus_oid[i] = focus  # stored for the next `switch_every` steps
            else:
                focus = self._focus_oid[i]

            self._steps[i] += 1

            # Sample a random action in env-action units and clip near focus.
            action = self.rng.uniform(-1, 1, size=env.action_space.shape)
            action = action * env.action_scale
            action = np.array(env.agent.position) + action
            action = np.clip(
                action, focus - self.dist_constraint, focus + self.dist_constraint
            )
            # Back to normalized action.
            action = (action - np.array(env.agent.position)) / env.action_scale
            action = np.clip(action, -1, 1)
            actions[i] = action

        return actions


class MultiObjectGoalPolicy(BasePolicy):
    """Greedy goal-driving expert with action noise + orientation control.

    Three-phase controller per focus object:

      STAGE   — agent is far from the "behind" point on the obj→goal
                axis; aim at that staging point with a lateral offset
                so the agent walks around the obj instead of through
                it.
      PUSH    — agent is already behind the obj on the goal side; aim
                slightly past the obj to nudge it toward the goal.
      ORIENT  — pos_err < `orient_pos_threshold` and the obj has
                orientation that's still off by more than
                `switch_ang_tol`. Push at a lever-arm point offset
                perpendicular to obj→goal so the resulting force
                generates torque in the sign(Δθ) direction. Side
                effects: a small lateral translation, corrected when
                the policy switches back to STAGE/PUSH on the next
                focus cycle.

    Focus picking ("which obj to drive next"):
      An obj is *unsatisfied* if pos_err > pos_tol OR (has_orientation
      AND ang_err > ang_tol). The unsatisfied obj with the highest
      combined score `pos_err + 30 · ang_err` is the next focus
      (weight matches the env / CEM joint cost).

    Workspace bound:
      The world-space agent target is clipped to `[margin, ws - margin]`
      so the policy never aims off-screen and drags an obj out of view.

    Action smoothness:
      `max_step` caps per-step displacement to ≈ standard-PushT step
      size; `action_smoothing` EMA-blends with the previous executed
      action to cut sign-flip rate. See std-PushT comparison in the
      branch's docs.

    Args:
        push_distance: staging distance behind the object (pixels).
        noise_std: Gaussian noise std added to the action.
        switch_every: hard cap on consecutive steps targeting the
            same focus.
        switch_pos_tol: pos-tolerance for considering a focus
            "satisfied" along position.
        switch_ang_tol: ang-tolerance (rad) for considering a focus
            "satisfied" along angle.
        orient_pos_threshold: position-error threshold below which
            the policy switches into ORIENT mode for the current
            focus.
        orient_lever_frac: fraction of the obj's bounding_radius used
            as the lever-arm radius. 1.0 is on the edge; 0.7 sits
            slightly inside so the agent has room to push.
        orient_stage_distance_frac: how far the agent stages from the
            lever-arm contact point, as a fraction of push_distance.
        workspace_margin: clip target to [margin, ws - margin].
        max_step: per-step displacement cap in [−1, 1] action units.
        action_smoothing: EMA coefficient on actions (α ∈ [0, 1]).
        seed: RNG seed.
    """

    def __init__(
        self,
        push_distance: float = 60.0,
        noise_std: float = 0.05,
        switch_every: int = 40,
        switch_pos_tol: float = 18.0,
        switch_ang_tol: float = np.pi / 9,
        orient_pos_threshold: float = 40.0,
        orient_lever_frac: float = 0.7,
        orient_stage_distance_frac: float = 0.7,
        workspace_margin: float = 30.0,
        max_step: float = 0.25,
        action_smoothing: float = 1.0,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert push_distance > 0
        assert 0 <= noise_std
        assert switch_every >= 1
        assert 0 < max_step <= 1.0
        assert 0.0 <= action_smoothing <= 1.0
        assert 0 < orient_lever_frac <= 1.0
        assert orient_stage_distance_frac > 0
        assert workspace_margin >= 0
        self.push_distance = float(push_distance)
        self.noise_std = float(noise_std)
        self.switch_every = int(switch_every)
        self.switch_pos_tol = float(switch_pos_tol)
        self.switch_ang_tol = float(switch_ang_tol)
        self.orient_pos_threshold = float(orient_pos_threshold)
        self.orient_lever_frac = float(orient_lever_frac)
        self.orient_stage_distance_frac = float(orient_stage_distance_frac)
        self.workspace_margin = float(workspace_margin)
        # Cap on per-step displacement in [-1,1] action units. Without
        # this, far targets produce `delta = (target - agent)/action_scale`
        # ≫ 1 and the clip saturates — std-PushT expert sits near
        # |a| ≈ 0.27, so default 0.3 matches that step-size.
        self.max_step = float(max_step)
        # Exponential moving-average smoothing on the executed action:
        # aₜ = α · raw + (1 − α) · aₜ₋₁. A purely reactive state-based
        # policy can still produce sign flips when the agent oscillates
        # around its target; EMA filters out the high-frequency component
        # without changing the policy's intent. action_smoothing is α
        # (in [0, 1]); 0 means no smoothing, lower → smoother. Default
        # 0.4 ≈ time-constant of ~2 steps, matching std-PushT expert
        # |Δa| ≈ 0.085.
        self.action_smoothing = float(action_smoothing)
        self.set_seed(seed)
        self._focus: list[str | None] = []
        self._focus_age: np.ndarray | None = None
        self._focus_side: list[int] = []  # which side to circumnavigate
        self._focus_mode: list[str] = []  # 'STAGE' | 'PUSH', hysteresis-cached
        self._prev_action: list[np.ndarray | None] = []

    def set_seed(self, seed: int | None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def set_env(self, env) -> None:
        self.env = env
        spec = getattr(env, 'spec', None)
        if spec is None:
            envs = getattr(env, 'envs', None)
            if envs:
                spec = envs[0].spec
        assert spec is not None and 'swm/PushTMulti' in spec.id, (
            f'MultiObjectGoalPolicy requires swm/PushTMulti, got {spec}'
        )

    def _envs(self):
        base = self.env.unwrapped
        return [e.unwrapped for e in base.envs] if hasattr(base, 'envs') else [base]

    def _ensure_state(self, n: int) -> None:
        if self._focus_age is None or len(self._focus_age) != n:
            self._focus = [None] * n
            self._focus_age = np.zeros(n, dtype=np.int64)
            self._focus_side = [1] * n
            self._focus_mode = ['STAGE'] * n
            self._prev_action = [None] * n

    @staticmethod
    def _signed_angle_diff(current: float, target: float) -> float:
        """Shortest signed difference (target - current) wrapped to (-π, π].

        Positive means the body needs CCW rotation to reach target.
        """
        d = (target - current + np.pi) % (2 * np.pi) - np.pi
        return float(d)

    def _pick_focus(self, env, current: str | None) -> str | None:
        """Pick the next focus object.

        Strategy:
          1. Build the set of *unsatisfied* objects (pos_err > pos_tol
             OR ang_err > ang_tol).
          2. If there's more than one unsatisfied obj and `current` is
             in the set, rotate away from `current` — pick the highest-
             scoring *other* unsatisfied obj. This prevents argmax from
             always returning the same biggest-score obj indefinitely.
          3. Otherwise pick the highest-scoring unsatisfied obj.

        Score: `pos_err + 30·|ang_err|` (matches env / CEM joint cost).
        Returns None if every enabled obj is within tolerance.
        """
        enabled = sorted(env.enabled)
        if not enabled:
            return None
        unsatisfied: list[tuple[str, float]] = []
        for oid in enabled:
            body = env.bodies.get(oid)
            if body is None:
                continue
            pos = np.array(body.position, dtype=np.float64)
            goal = env.goal_pose[oid]
            spec = env.specs[oid]
            pos_err = float(np.linalg.norm(pos - goal[:2]))
            if spec.has_orientation:
                ang_err = abs(self._signed_angle_diff(
                    float(body.angle), float(goal[2])
                ))
            else:
                ang_err = 0.0
            is_unsatisfied = (
                pos_err > self.switch_pos_tol
                or (spec.has_orientation and ang_err > self.switch_ang_tol)
            )
            if is_unsatisfied:
                score = pos_err + 30.0 * ang_err
                unsatisfied.append((oid, score))
        if not unsatisfied:
            return current if current in enabled else enabled[0]
        return max(unsatisfied, key=lambda t: t[1])[0]

    def get_action(self, info_dict, **kwargs):
        assert hasattr(self, 'env'), 'Environment not set for the policy'
        envs = self._envs()
        self._ensure_state(len(envs))

        act_shape = self.env.action_space.shape
        actions = np.zeros(act_shape, dtype=np.float32)

        for i, env in enumerate(envs):
            current = self._focus[i]
            need_switch = (
                current is None
                or current not in env.enabled
                or self._focus_age[i] >= self.switch_every
            )
            # Also switch when current focus has reached BOTH position and
            # angle tolerance (per the joint success criterion of the env).
            if not need_switch:
                body = env.bodies.get(current)
                if body is not None:
                    pos = np.array(body.position, dtype=np.float64)
                    goal = env.goal_pose[current]
                    spec = env.specs[current]
                    pos_ok = float(np.linalg.norm(pos - goal[:2])) <= self.switch_pos_tol
                    if spec.has_orientation:
                        ang_err = abs(self._signed_angle_diff(
                            float(body.angle), float(goal[2])
                        ))
                        ang_ok = ang_err <= self.switch_ang_tol
                    else:
                        ang_ok = True
                    if pos_ok and ang_ok:
                        need_switch = True

            if need_switch:
                new_focus = self._pick_focus(env, current)
                self._focus[i] = new_focus
                self._focus_age[i] = 0
                self._focus_mode[i] = 'STAGE'  # reset hysteresis on switch
                # Commit to a circumnavigation side based on the current
                # agent geometry. Without this commitment the side flips
                # back and forth between steps and the agent oscillates.
                if new_focus is not None and new_focus in env.bodies:
                    obj_pos = np.array(env.bodies[new_focus].position, dtype=np.float64)
                    goal_pos = env.goal_pose[new_focus][:2]
                    push_dir = goal_pos - obj_pos
                    nrm = float(np.linalg.norm(push_dir))
                    if nrm > 1e-6:
                        push_dir = push_dir / nrm
                        perp = np.array([-push_dir[1], push_dir[0]])
                        agent_xy = np.array(env.agent.position, dtype=np.float64)
                        lat = float(np.dot(agent_xy - obj_pos, perp))
                        self._focus_side[i] = 1 if lat >= 0 else -1
                    else:
                        self._focus_side[i] = 1

            focus = self._focus[i]
            agent_pos = np.array(env.agent.position, dtype=np.float64)

            if focus is None:
                # All satisfied — idle near the workspace center.
                target = np.array([env.window_size / 2] * 2, dtype=np.float64)
            else:
                body = env.bodies[focus]
                obj_pos = np.array(body.position, dtype=np.float64)
                obj_angle = float(body.angle)
                goal = env.goal_pose[focus]
                goal_pos = goal[:2]
                goal_angle = float(goal[2])
                spec = env.specs[focus]
                push_vec = goal_pos - obj_pos
                d_to_goal = float(np.linalg.norm(push_vec))

                # Geometry: stage outside the focus obj's bounding circle.
                # The default `push_distance` is a configurable minimum but
                # the *effective* staging distance must be ≥ obj radius +
                # agent radius + small margin, or the agent target lands
                # *inside* the obj's footprint and the agent gets stuck.
                agent_r = 0.375 * float(env.variation_space['agent']['scale'].value)
                clear_dist = spec.bounding_radius + agent_r + 10.0
                behind_dist = max(self.push_distance, clear_dist)

                # Decide mode: ORIENT runs only for has-orientation objects
                # whose position is already close but angle is still off.
                if spec.has_orientation:
                    signed_ang = self._signed_angle_diff(obj_angle, goal_angle)
                else:
                    signed_ang = 0.0
                use_orient = (
                    spec.has_orientation
                    and d_to_goal < self.orient_pos_threshold
                    and abs(signed_ang) > self.switch_ang_tol
                )

                if use_orient:
                    # ORIENT: agent stages just outside the obj's bounding
                    # circle along `u` (perpendicular to obj→goal), then
                    # walks tangentially in direction `F` (perpendicular
                    # to `u`, sign chosen so r×F induces the desired
                    # rotation). Walking through the contact slides along
                    # the obj's edge, applying tangential force = torque.
                    if d_to_goal < 1e-3:
                        # Object on goal — pick a u aligned with the body
                        # so the construction is well-defined.
                        u = np.array([np.cos(obj_angle), np.sin(obj_angle)])
                    else:
                        push_dir = push_vec / d_to_goal
                        u = np.array([-push_dir[1], push_dir[0]])
                    sign = 1.0 if signed_ang > 0 else -1.0
                    F = sign * np.array([-u[1], u[0]])
                    stage = obj_pos + (spec.bounding_radius + agent_r + 5.0) * u
                    # Tangential walk distance — small so the orient pass
                    # doesn't blow past the angle goal.
                    target = stage + F * 30.0
                else:
                    # STAGE/PUSH (translation control).
                    if d_to_goal < 1e-3:
                        push_dir = self.rng.normal(0, 1, 2)
                        push_dir /= max(np.linalg.norm(push_dir), 1e-9)
                    else:
                        push_dir = push_vec / d_to_goal

                    agent_offset = agent_pos - obj_pos
                    along_push = float(np.dot(agent_offset, push_dir))
                    perp_dir = np.array([-push_dir[1], push_dir[0]])
                    side = self._focus_side[i]
                    behind_pt = obj_pos - push_dir * behind_dist

                    # Hysteresis between STAGE and PUSH to prevent
                    # oscillation around the boundary. Enter PUSH when
                    # well behind the obj (along_push ≤ −0.7·d); only
                    # exit back to STAGE when the agent is in front
                    # (along_push ≥ −0.3·d). The previous mode is
                    # cached per-focus on the policy so the transition
                    # is sticky across steps.
                    prev_mode = self._focus_mode[i]
                    if prev_mode == 'PUSH':
                        in_push = along_push < -0.3 * behind_dist
                    else:
                        in_push = along_push < -0.7 * behind_dist
                    self._focus_mode[i] = 'PUSH' if in_push else 'STAGE'

                    if in_push:
                        # PUSH target sits at a moderate distance *ahead*
                        # of the obj along push_dir. Two competing
                        # constraints:
                        # - Far enough that the agent commits to a long
                        #   walk through the obj (otherwise STAGE/PUSH
                        #   oscillates after one push step).
                        # - Not so far that the obj overshoots and exits
                        #   the workspace. 1.5×behind_dist works well in
                        #   practice; capped by d_to_goal so the final
                        #   approach is gentle.
                        push_ahead = min(1.5 * behind_dist, d_to_goal)
                        target = obj_pos + push_dir * push_ahead
                    else:
                        d_to_behind = float(np.linalg.norm(behind_pt - agent_pos))
                        offset_factor = min(d_to_behind / (2 * behind_dist), 1.0)
                        lat_off = perp_dir * side * behind_dist * offset_factor
                        target = behind_pt + lat_off

            # Workspace bound: clip the agent target so the policy can't
            # aim off-screen and drag objects out of view. Margin is set
            # generously so the agent doesn't kiss the canvas edge.
            ws = env.window_size
            target = np.clip(
                target, self.workspace_margin, ws - self.workspace_margin
            )

            # Convert world-space target into the env's [-1, 1] action
            # (relative mode: action * action_scale = delta from current).
            # Scale the step so its magnitude doesn't exceed `max_step`
            # — without this cap, far targets produce delta ≫ 1 and the
            # final clip saturates to ±1, producing the harshest possible
            # pushes. With max_step=0.3 the agent moves ~30 px/step,
            # matching the std-PushT expert.
            delta = (target - agent_pos) / env.action_scale
            mag = float(np.linalg.norm(delta))
            if mag > self.max_step:
                delta = delta * (self.max_step / mag)
            delta = delta + self.rng.normal(0, self.noise_std, size=2)
            raw = np.clip(delta, -1.0, 1.0)
            # EMA smoothing against the previous executed action — see
            # `action_smoothing` in __init__. 1.0 means no smoothing.
            if self.action_smoothing < 1.0 and self._prev_action[i] is not None:
                alpha = self.action_smoothing
                raw = alpha * raw + (1 - alpha) * self._prev_action[i]
                raw = np.clip(raw, -1.0, 1.0)
            actions[i] = raw
            self._prev_action[i] = np.asarray(raw, dtype=np.float64).copy()
            self._focus_age[i] += 1

        return actions


class MultiObjectCEMPolicy(BasePolicy):
    """Oracle-CEM MPC expert.

    Treats the environment itself as the dynamics model: at every
    replan, the policy snapshots the env state, samples action
    sequences from a Gaussian, rolls each sequence forward through the
    *real* simulator, scores by joint distance to goal, takes the
    top-K elites, refits, and repeats for `n_iter` CEM iterations.
    The first action of the elite mean is executed; the remainder
    is cached and replayed for the next `replan_every - 1` steps.

    Because rollouts step the same env that we are collecting on, the
    expert sees physics identical to data-collection time — there's no
    sim-to-real gap between the planner and the runtime, only compute
    cost. Expected speed at K=64, H=8, n_iter=2, replan_every=3 is
    roughly 1–2 seconds per env-step on a single env, so this expert
    is meant for **non-vectorized collection** (`num_envs=1`).

    Args:
        horizon: planning horizon in env steps.
        num_samples: CEM candidates per iteration.
        n_iter: CEM refit iterations per replan.
        topk: number of elite candidates kept after each iteration.
        replan_every: env steps between replan calls (1 = full MPC).
        init_std: initial Gaussian std for action sampling.
        action_noise_std: Gaussian noise on the executed action (small,
            so the expert isn't perfectly deterministic).
        angle_weight: cost weight on angle error (px units; angle in
            radians is multiplied by this number).
        seed: RNG seed.
    """

    def __init__(
        self,
        horizon: int = 8,
        num_samples: int = 64,
        n_iter: int = 2,
        topk: int = 8,
        replan_every: int = 3,
        init_std: float = 0.3,
        action_noise_std: float = 0.05,
        angle_weight: float = 30.0,
        action_penalty: float = 1.5,
        action_clip: float = 0.3,
        smoothness_weight: float = 8.0,
        warm_start_from_previous: bool = True,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert num_samples >= topk * 2, 'num_samples should be ≥ 2*topk'
        assert 0.0 < action_clip <= 1.0
        self.horizon = int(horizon)
        self.num_samples = int(num_samples)
        self.n_iter = int(n_iter)
        self.topk = int(topk)
        self.replan_every = int(replan_every)
        self.init_std = float(init_std)
        self.action_noise_std = float(action_noise_std)
        self.angle_weight = float(angle_weight)
        # Action-magnitude penalty: cost += action_penalty * ||a||² summed
        # over the rollout. Without this, CEM frequently saturates the
        # action to ±1 (full-speed pushes) because terminal cost has no
        # opinion on effort — the resulting contacts look way harder
        # than the standard PushT expert (human teleop, naturally gentle).
        self.action_penalty = float(action_penalty)
        # Tighter action clip during CEM sampling — agent never commands
        # full-speed unless the cost gradient really demands it. Default
        # 0.3 matches the standard PushT expert's typical |a| ≈ 0.27.
        self.action_clip = float(action_clip)
        # Smoothness penalty: cost += smoothness_weight * Σ‖aₜ − aₜ₋₁‖²
        # over the rollout, *including* the boundary to the previously
        # executed action. Without this, CEM is free to flip the agent's
        # direction every step because terminal cost is direction-agnostic
        # over the horizon; standard PushT expert action sign-flip rate is
        # ~7%, ours was 27%.
        self.smoothness_weight = float(smoothness_weight)
        # When True, the CEM mean is initialised by shifting the previous
        # elite plan left by `replan_every` steps (pad zeros at the tail).
        # This is the standard receding-horizon warm-start and is the
        # second-largest source of smoothness — without it every replan
        # restarts from a heuristic mean and the plan jumps at boundaries.
        self.warm_start_from_previous = bool(warm_start_from_previous)
        self.set_seed(seed)
        self._plan: list[np.ndarray] = []  # cached per-env plan tail
        self._step: int = 0
        self._last_plan: np.ndarray | None = None  # for warm-start
        self._last_executed_action: np.ndarray | None = None  # smooth boundary

    def set_seed(self, seed: int | None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def set_env(self, env) -> None:
        self.env = env
        spec = getattr(env, 'spec', None)
        if spec is None:
            envs = getattr(env, 'envs', None)
            if envs:
                spec = envs[0].spec
        assert spec is not None and 'swm/PushTMulti' in spec.id, (
            f'MultiObjectCEMPolicy requires swm/PushTMulti, got {spec}'
        )
        envs = self._envs()
        if len(envs) != 1:
            raise RuntimeError(
                'MultiObjectCEMPolicy supports num_envs=1 only; got '
                f'{len(envs)}. CEM rollouts in the live env are not '
                'parallelizable.'
            )

    def _envs(self):
        base = self.env.unwrapped
        return [e.unwrapped for e in base.envs] if hasattr(base, 'envs') else [base]

    def _cost(self, env) -> float:
        """Joint cost = sum over enabled objects of (pos_dist + angle*weight)."""
        cost = 0.0
        for oid in env.enabled:
            body = env.bodies.get(oid)
            if body is None:
                continue
            pos = np.array(body.position, dtype=np.float64)
            angle = float(body.angle % (2 * np.pi))
            goal = env.goal_pose[oid]
            pos_d = float(np.linalg.norm(pos - goal[:2]))
            cost += pos_d
            spec = env.specs[oid]
            if spec.has_orientation:
                ang_d = abs((angle - goal[2]) % (2 * np.pi))
                ang_d = min(ang_d, 2 * np.pi - ang_d)
                cost += ang_d * self.angle_weight
        return cost

    def _rollout_cost(
        self, env, action_seq: np.ndarray, prev_action: np.ndarray
    ) -> float:
        """Apply `action_seq` (H, 2) to the env step-by-step.

        Cost = terminal joint distance + action effort + smoothness,
        where smoothness penalises both inner-rollout direction changes
        and a single-step boundary to the action just executed.
        """
        for h in range(action_seq.shape[0]):
            env.step(action_seq[h])
        terminal = self._cost(env)
        effort = self.action_penalty * float(np.sum(action_seq ** 2))
        # Inner-rollout smoothness.
        dseq = np.diff(action_seq, axis=0)
        smooth_inner = float(np.sum(dseq ** 2))
        # Boundary to the previously executed action.
        smooth_boundary = float(np.sum((action_seq[0] - prev_action) ** 2))
        smoothness = self.smoothness_weight * (smooth_inner + smooth_boundary)
        return terminal + effort + smoothness

    def _warm_start(self, env) -> np.ndarray:
        """Heuristic init: aim at the "behind" point of the focus object
        (the side opposite to its goal) for the first half of the
        horizon, then at the obj's center for the second half. So the
        warm-started rollout walks the agent around to the staging point
        and then pushes through.

        Without a focus-aware warm-start, random walks from a zero-mean
        Gaussian almost never touch the obj — every rollout returns the
        same constant cost and CEM has no signal.
        """
        H = self.horizon
        agent_pos = np.array(env.agent.position, dtype=np.float64)
        dists = {}
        for oid in env.enabled:
            body = env.bodies.get(oid)
            if body is None:
                continue
            p = np.array(body.position, dtype=np.float64)
            g = env.goal_pose[oid][:2]
            dists[oid] = float(np.linalg.norm(p - g))
        if not dists:
            return np.zeros((H, 2), dtype=np.float64)
        focus_oid = max(dists, key=dists.get)
        obj_pos = np.array(env.bodies[focus_oid].position, dtype=np.float64)
        goal_pos = np.asarray(env.goal_pose[focus_oid][:2], dtype=np.float64)
        push_dir = goal_pos - obj_pos
        nrm = float(np.linalg.norm(push_dir))
        if nrm < 1e-3:
            return np.zeros((H, 2), dtype=np.float64)
        push_dir = push_dir / nrm
        behind = obj_pos - push_dir * 60.0

        # Phase 1: head to behind. Phase 2: head past obj toward goal.
        def _unit_to(target, pos):
            d = target - pos
            n = float(np.linalg.norm(d))
            return d / n if n > 1e-3 else np.zeros(2)

        d1 = _unit_to(behind, agent_pos)
        d2 = push_dir  # already unit-length
        plan = np.zeros((H, 2), dtype=np.float64)
        half = max(1, H // 2)
        plan[:half] = d1
        plan[half:] = d2
        return np.clip(plan, -self.action_clip, self.action_clip)

    def _cem(self, env) -> np.ndarray:
        """Run CEM in the env. Returns the elite mean action sequence (H, 2).

        Snapshots and restores state internally so the live env pose is
        unchanged on return.
        """
        snapshot = env._snapshot_bodies()
        H, K = self.horizon, self.num_samples

        # Warm-start: shift the last elite plan left by `replan_every` and
        # pad the tail with zeros. Falls back to the focus-aware heuristic
        # on the very first replan or after an episode reset.
        if self.warm_start_from_previous and self._last_plan is not None:
            shift = self.replan_every
            prev = self._last_plan
            mean = np.zeros((H, 2), dtype=np.float64)
            copy_len = max(0, H - shift)
            if copy_len > 0:
                mean[:copy_len] = prev[shift:shift + copy_len]
        else:
            mean = self._warm_start(env)

        std = np.full((H, 2), self.init_std, dtype=np.float64)
        prev_action = (
            self._last_executed_action
            if self._last_executed_action is not None
            else np.zeros(2, dtype=np.float64)
        )

        for _ in range(self.n_iter):
            # Sample K candidate sequences and clip to action box.
            noise = self.rng.normal(size=(K, H, 2))
            candidates = np.clip(
                mean[None] + std[None] * noise,
                -self.action_clip, self.action_clip,
            )

            costs = np.empty(K, dtype=np.float64)
            for k in range(K):
                env._restore_bodies(snapshot)
                costs[k] = self._rollout_cost(env, candidates[k], prev_action)

            # Pick elites, refit Gaussian.
            elite_idx = np.argsort(costs)[: self.topk]
            elites = candidates[elite_idx]
            mean = elites.mean(axis=0)
            std = elites.std(axis=0) + 1e-3

        # Final restore so the live env pose is exactly the pre-CEM state.
        env._restore_bodies(snapshot)
        self._last_plan = mean.copy()
        return mean

    def get_action(self, info_dict, **kwargs):
        assert hasattr(self, 'env'), 'Environment not set for the policy'
        envs = self._envs()
        env = envs[0]

        if self._step % self.replan_every == 0 or not self._plan:
            plan = self._cem(env)
            self._plan = [plan[h] for h in range(plan.shape[0])]

        action = self._plan.pop(0)
        action = action + self.rng.normal(0, self.action_noise_std, size=2)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self._last_executed_action = action.astype(np.float64)
        self._step += 1

        return action[None, :]  # shape (n_envs=1, 2)
