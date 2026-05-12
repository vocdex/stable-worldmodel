"""Weak collection policy for PushTMulti.

Generalizes the single-object `WeakPolicy` in `pusht/expert_policy.py`:
every `switch_every` steps, pick a *focus* — usually one of the enabled
objects, but with probability `p_wedge` the midpoint between two
enabled objects. The wedge mode drives pairwise-contact sampling so that
the AB / BC training datasets actually contain object-object interactions
(see the RQ2 risk note in the plan).
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
