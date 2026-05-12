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
    """Greedy goal-driving expert with action noise.

    Each step (or every `switch_every` steps), pick a *focus* object —
    the unsatisfied object farthest from its goal — and drive the agent
    to a "push-from-behind" target relative to the goal direction. Once
    the focus object is within `switch_pos_tol` of its goal it's
    considered locally satisfied and the policy switches.

    Action noise (Gaussian, std=`noise_std` in the [-1, 1] action space)
    breaks symmetry and makes trajectories non-deterministic, mirroring
    the noisy-expert protocol used to collect the 18k-trajectory PushT
    dataset for DINO-WM. The expert is *not* perfect (no angle control
    here — angle is left to the noise + dynamics), which is intentional:
    we want diverse near-goal trajectories, not optimal ones.

    Args:
        push_distance: how far behind the object the agent aims (in
            pymunk pixels). Roughly the object's bounding radius plus
            a margin.
        noise_std: Gaussian noise std added to the action.
        switch_every: hard cap on consecutive steps targeting the same
            focus, so the policy doesn't get stuck pushing in vain.
        switch_pos_tol: focus is "locally satisfied" when the object's
            position is within this many pixels of its goal.
        seed: RNG seed.
    """

    def __init__(
        self,
        push_distance: float = 60.0,
        noise_std: float = 0.15,
        switch_every: int = 40,
        switch_pos_tol: float = 18.0,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert push_distance > 0
        assert 0 <= noise_std
        assert switch_every >= 1
        self.push_distance = float(push_distance)
        self.noise_std = float(noise_std)
        self.switch_every = int(switch_every)
        self.switch_pos_tol = float(switch_pos_tol)
        self.set_seed(seed)
        self._focus: list[str | None] = []
        self._focus_age: np.ndarray | None = None
        self._focus_side: list[int] = []  # which side to circumnavigate

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

    def _pick_focus(self, env, current: str | None) -> str | None:
        """Pick the unsatisfied object farthest from its goal. Returns
        None if every enabled object is already within tolerance.
        """
        enabled = sorted(env.enabled)
        if not enabled:
            return None
        dists: dict[str, float] = {}
        for oid in enabled:
            body = env.bodies.get(oid)
            if body is None:
                continue
            pos = np.array(body.position, dtype=np.float64)
            goal = env.goal_pose[oid][:2]
            dists[oid] = float(np.linalg.norm(pos - goal))
        if not dists:
            return None
        unsatisfied = {o: d for o, d in dists.items() if d > self.switch_pos_tol}
        if unsatisfied:
            return max(unsatisfied, key=unsatisfied.get)
        # All locally satisfied — keep current focus or pick any.
        return current if current in dists else next(iter(dists))

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
            # Also switch if current focus is already within tolerance.
            if not need_switch:
                body = env.bodies.get(current)
                if body is not None:
                    pos = np.array(body.position, dtype=np.float64)
                    goal = env.goal_pose[current][:2]
                    if np.linalg.norm(pos - goal) <= self.switch_pos_tol:
                        need_switch = True

            if need_switch:
                new_focus = self._pick_focus(env, current)
                self._focus[i] = new_focus
                self._focus_age[i] = 0
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
                goal_pos = env.goal_pose[focus][:2]
                push_dir = goal_pos - obj_pos
                d_to_goal = float(np.linalg.norm(push_dir))
                if d_to_goal < 1e-3:
                    # Object is on the goal; small random nudge.
                    push_dir = self.rng.normal(0, 1, 2)
                    push_dir /= max(np.linalg.norm(push_dir), 1e-9)
                else:
                    push_dir = push_dir / d_to_goal

                # Two-phase controller:
                #   PUSH  — agent is behind the obj relative to the goal;
                #           aim past the obj so walking forward drives
                #           it through to the goal.
                #   STAGE — agent is in front of or beside the obj; aim
                #           at the "behind" point plus a perpendicular
                #           offset on the committed side. The offset
                #           shrinks linearly with distance to behind, so
                #           when the agent is far it walks around the
                #           obj, and when it's near, it lines up on the
                #           push axis.
                agent_offset = agent_pos - obj_pos
                along_push = float(np.dot(agent_offset, push_dir))
                perp_dir = np.array([-push_dir[1], push_dir[0]])
                behind_dist = self.push_distance
                side = self._focus_side[i]
                behind_pt = obj_pos - push_dir * behind_dist

                if along_push < -0.5 * behind_dist:
                    # PUSH target close to the obj, not far past it —
                    # otherwise the agent teleports past the obj each
                    # step and ends up on the goal-side, flipping back to
                    # APPROACH. Keeping the target close means the agent
                    # nudges the obj, then sits ~behind it for the next
                    # push step.
                    target = obj_pos + push_dir * 10.0
                else:
                    d_to_behind = float(np.linalg.norm(behind_pt - agent_pos))
                    # Lateral offset = behind_dist when far, 0 at behind.
                    offset_factor = min(d_to_behind / (2 * behind_dist), 1.0)
                    lat_off = perp_dir * side * behind_dist * offset_factor
                    target = behind_pt + lat_off

            # Convert world-space target into the env's [-1, 1] action
            # (relative mode: action * action_scale = delta from current).
            # Action magnitude scales with distance to target — large
            # when far, small when close — so the agent doesn't teleport
            # past the target every step (which causes oscillation).
            delta = (target - agent_pos) / env.action_scale
            delta = delta + self.rng.normal(0, self.noise_std, size=2)
            actions[i] = np.clip(delta, -1.0, 1.0)
            self._focus_age[i] += 1

        return actions
