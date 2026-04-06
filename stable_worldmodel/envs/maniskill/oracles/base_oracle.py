import numpy as np


# Conservative workspace for post-task wandering (avoids IK limits)
_WANDER_LOW = np.array([-0.15, -0.15, 0.08])
_WANDER_HIGH = np.array([0.15, 0.15, 0.35])

# Play-mode workspace: near table surface for contact interactions
_PLAY_LOW = np.array([-0.15, -0.15, 0.02])
_PLAY_HIGH = np.array([0.15, 0.15, 0.10])


class PandaMarkovOracle:
    """Base class for closed-loop Markov oracles on ManiSkill Panda tasks.

    All oracles use pd_ee_delta_pos control mode (4D actions: dx, dy, dz, gripper).
    The Panda controller maps [-1, 1] to [-0.1, 0.1] m/step for EE delta,
    and [-1, 1] to [closed, open] for the gripper.

    After the task-specific phases complete, the oracle continuously visits
    random waypoints across the workspace until the episode ends.
    """

    GRIPPER_OPEN = 1.0
    GRIPPER_CLOSE = -1.0

    def __init__(self, min_norm=0.1, gain=2.0, max_steps=200):
        self._min_norm = min_norm
        self._gain = gain
        self._max_steps = max_steps
        self._done = False
        self._task_done = False
        self._step = 0
        self._debug = False
        self._waypoint = None

    @property
    def done(self):
        return self._done

    def reset(self, ob, info):
        self._done = False
        self._task_done = False
        self._step = 0
        self._waypoint = np.random.uniform(_WANDER_LOW, _WANDER_HIGH)
        self._play_gripper = self.GRIPPER_OPEN

    def select_action(self, ob, info):
        raise NotImplementedError

    def _post_task_action(self, tcp):
        """Slowly wander between waypoints after task completion."""
        if np.linalg.norm(self._waypoint - tcp) < 0.04:
            self._waypoint = np.random.uniform(_WANDER_LOW, _WANDER_HIGH)
        diff = self._waypoint - tcp
        action = np.zeros(4)
        action[:3] = diff * 1.5
        action[3] = self.GRIPPER_OPEN
        return np.clip(action, -1, 1)

    def _play_action(self, tcp):
        """Sweep near the table surface, pushing and bumping objects."""
        if np.linalg.norm(self._waypoint - tcp) < 0.03:
            self._waypoint = np.random.uniform(_PLAY_LOW, _PLAY_HIGH)
            # Randomly toggle gripper for grasp attempts
            self._play_gripper = np.random.choice(
                [self.GRIPPER_OPEN, self.GRIPPER_CLOSE],
            )
        diff = self._waypoint - tcp
        action = np.zeros(4)
        action[:3] = diff * 1.5
        action[3] = self._play_gripper
        return np.clip(action, -1, 1)

    def shape_diff(self, diff):
        """Shape the difference vector to have a minimum norm."""
        diff_norm = np.linalg.norm(diff)
        if diff_norm >= self._min_norm:
            return diff
        return diff / (diff_norm + 1e-6) * self._min_norm

    def print_phase(self, phase):
        if self._debug:
            print(f'Phase {phase:50}', end=' ')

    def ee_action(self, tcp, target, gripper):
        """Compute 4D pd_ee_delta_pos action toward target."""
        diff = target - tcp
        diff = self.shape_diff(diff)
        action = np.zeros(4)
        action[:3] = diff * self._gain
        action[3] = gripper
        return np.clip(action, -1, 1)
