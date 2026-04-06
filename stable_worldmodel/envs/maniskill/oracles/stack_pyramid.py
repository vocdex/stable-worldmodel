import numpy as np

from stable_worldmodel.envs.maniskill.oracles.base_oracle import PandaMarkovOracle


CUBE_HALF_SIZE = 0.02


class StackPyramidMarkovOracle(PandaMarkovOracle):
    """Closed-loop Markov oracle for the StackPyramid task.

    Two-phase task:
      1. align_ab: If cubeA and cubeB are far apart, pick cubeA and place next to cubeB.
      2. stack_c:  Pick cubeC and place on top of cubeA+cubeB midpoint.

    After stacking, executes post-task phases (release, retreat, wander).
    """

    ALIGN_DISTANCE = 0.07

    def __init__(self, max_steps=200, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_steps = max_steps

    def reset(self, ob, info):
        super().reset(ob, info)
        self._subtask = 'check'
        self._released_a = False

    def select_action(self, ob, info):
        tcp = info['extra/tcp_pose'][:3].copy()

        if self._task_done:
            action = self._post_task_action(tcp)
            self._step += 1
            if self._step >= self._max_steps:
                self._done = True
            return action

        cubeA = info['extra/cubeA_pose'][:3].copy()
        cubeB = info['extra/cubeB_pose'][:3].copy()
        cubeC = info['extra/cubeC_pose'][:3].copy()
        grasped_A = bool(info.get('eval/is_cubeA_grasped', False))
        grasped_C = bool(info.get('eval/is_cubeC_grasped', False))

        if self._subtask == 'check':
            ab_dist = np.linalg.norm(cubeA[:2] - cubeB[:2])
            self._subtask = 'align_ab' if ab_dist > self.ALIGN_DISTANCE else 'stack_c'

        if self._subtask == 'align_ab':
            action = self._align_ab(tcp, cubeA, cubeB, grasped_A)
        else:
            action = self._stack_c(tcp, cubeA, cubeB, cubeC, grasped_C)

        self._step += 1
        if self._step >= self._max_steps:
            self._done = True

        return action

    def _pick_place(self, tcp, obj, target, grasped, released_flag, label):
        """Generic pick-and-place subroutine used by both subtasks."""
        above_offset = np.array([0.0, 0.0, 0.10])
        above_threshold = 0.16
        xy_thresh = 0.04
        pos_thresh = 0.02

        xy_aligned = np.linalg.norm(obj[:2] - tcp[:2]) <= xy_thresh
        pos_aligned = np.linalg.norm(obj - tcp) <= pos_thresh
        target_xy_aligned = np.linalg.norm(target[:2] - tcp[:2]) <= xy_thresh
        target_aligned = np.linalg.norm(target - tcp) <= pos_thresh
        above = tcp[2] > above_threshold

        if released_flag:
            retreat = tcp.copy()
            retreat[2] = above_threshold
            action = self.ee_action(tcp, retreat, self.GRIPPER_OPEN)
            return action, tcp[2] > above_threshold - 0.03

        if not grasped:
            if not xy_aligned:
                self.print_phase(f'{label}: approach')
                action = self.ee_action(tcp, obj + above_offset, self.GRIPPER_OPEN)
            elif not pos_aligned:
                self.print_phase(f'{label}: lower')
                action = self.ee_action(tcp, obj, self.GRIPPER_OPEN)
            else:
                self.print_phase(f'{label}: grasp')
                action = self.ee_action(tcp, obj, self.GRIPPER_CLOSE)
        elif not target_xy_aligned and not above:
            self.print_phase(f'{label}: lift')
            lift_target = np.array([tcp[0], tcp[1], above_threshold * 2])
            action = self.ee_action(tcp, lift_target, self.GRIPPER_CLOSE)
        elif not target_xy_aligned:
            self.print_phase(f'{label}: transport')
            transport_target = np.array([
                target[0], target[1],
                max(tcp[2], target[2] + 0.06),
            ])
            action = self.ee_action(tcp, transport_target, self.GRIPPER_CLOSE)
        else:
            self.print_phase(f'{label}: lower to target')
            action = self.ee_action(tcp, target, self.GRIPPER_CLOSE)
            if target_aligned:
                action = self.ee_action(tcp, target, self.GRIPPER_OPEN)
                return action, True  # signal: just released

        return action, False

    def _align_ab(self, tcp, cubeA, cubeB, grasped):
        """Pick cubeA and place it adjacent to cubeB."""
        direction = cubeA[:2] - cubeB[:2]
        d = np.linalg.norm(direction)
        if d > 1e-6:
            direction = direction / d
        else:
            direction = np.array([1.0, 0.0])
        place_target = cubeB.copy()
        place_target[:2] += direction * 0.045

        action, done = self._pick_place(
            tcp, cubeA, place_target, grasped, self._released_a, 'align',
        )
        if done and not self._released_a:
            self._released_a = True
        elif done and self._released_a:
            self._subtask = 'stack_c'
        return action

    def _stack_c(self, tcp, cubeA, cubeB, cubeC, grasped):
        """Pick cubeC and place on top of cubeA+cubeB midpoint."""
        midpoint = (cubeA + cubeB) / 2.0
        stack_target = midpoint.copy()
        stack_target[2] += CUBE_HALF_SIZE * 2

        action, done = self._pick_place(
            tcp, cubeC, stack_target, grasped, False, 'stack',
        )
        if done:
            self._task_done = True
        return action
