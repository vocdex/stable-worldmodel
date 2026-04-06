import numpy as np

from stable_worldmodel.envs.maniskill.oracles.base_oracle import PandaMarkovOracle


CUBE_HALF_SIZE = 0.02


class StackCubeMarkovOracle(PandaMarkovOracle):
    """Closed-loop Markov oracle for the StackCube task.

    Randomly picks which cube to grasp (50% cubeA on cubeB, 50% cubeB on
    cubeA). Noise causes occasional drops; the condition tree automatically
    recovers. After success the arm wanders.
    """

    def __init__(self, max_steps=200, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_steps = max_steps

    def reset(self, ob, info):
        super().reset(ob, info)
        self._pick_A = np.random.random() < 0.5

    def select_action(self, ob, info):
        tcp = info['extra/tcp_pose'][:3].copy()

        if self._task_done:
            action = self._post_task_action(tcp)
            self._step += 1
            if self._step >= self._max_steps:
                self._done = True
            return action

        if self._pick_A:
            pick = info['extra/cubeA_pose'][:3].copy()
            base = info['extra/cubeB_pose'][:3].copy()
            grasped = bool(info.get('eval/is_cubeA_grasped', False))
            on_top = bool(info.get('eval/is_cubeA_on_cubeB', False))
        else:
            pick = info['extra/cubeB_pose'][:3].copy()
            base = info['extra/cubeA_pose'][:3].copy()
            grasped = bool(info.get('eval/is_cubeB_grasped', False))
            # No eval/is_cubeB_on_cubeA from env, compute manually
            offset = pick - base
            xy_close = np.linalg.norm(offset[:2]) < 0.02
            z_close = abs(offset[2] - CUBE_HALF_SIZE * 2) < 0.005
            on_top = xy_close and z_close

        stack_target = base.copy()
        stack_target[2] += CUBE_HALF_SIZE * 2

        above_offset = np.array([0.0, 0.0, 0.10])
        above_threshold = 0.16
        xy_thresh = 0.04
        pos_thresh = 0.02

        xy_aligned = np.linalg.norm(pick[:2] - tcp[:2]) <= xy_thresh
        pos_aligned = np.linalg.norm(pick - tcp) <= pos_thresh
        stack_xy_aligned = np.linalg.norm(stack_target[:2] - tcp[:2]) <= xy_thresh
        above = tcp[2] > above_threshold

        if on_top:
            self.print_phase('7: task complete')
            self._task_done = True
            action = np.array([0.0, 0.0, 0.0, self.GRIPPER_OPEN])
        elif not grasped:
            if not xy_aligned:
                self.print_phase('1: approach above pick cube')
                action = self.ee_action(tcp, pick + above_offset, self.GRIPPER_OPEN)
            elif not pos_aligned:
                self.print_phase('2: lower to pick cube')
                action = self.ee_action(tcp, pick, self.GRIPPER_OPEN)
            else:
                self.print_phase('3: grasp pick cube')
                action = self.ee_action(tcp, pick, self.GRIPPER_CLOSE)
        elif not stack_xy_aligned and not above:
            self.print_phase('4: lift')
            lift_target = np.array([tcp[0], tcp[1], above_threshold * 2])
            action = self.ee_action(tcp, lift_target, self.GRIPPER_CLOSE)
        elif not stack_xy_aligned:
            self.print_phase('5: transport above base cube')
            transport_target = np.array([
                stack_target[0], stack_target[1],
                max(tcp[2], stack_target[2] + 0.06),
            ])
            action = self.ee_action(tcp, transport_target, self.GRIPPER_CLOSE)
        else:
            self.print_phase('6: lower onto base cube')
            action = self.ee_action(tcp, stack_target, self.GRIPPER_CLOSE)

        self._step += 1
        if self._step >= self._max_steps:
            self._done = True

        return action
