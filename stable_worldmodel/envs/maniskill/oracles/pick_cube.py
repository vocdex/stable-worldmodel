import numpy as np

from stable_worldmodel.envs.maniskill.oracles.base_oracle import PandaMarkovOracle


class PickCubeMarkovOracle(PandaMarkovOracle):
    """Closed-loop Markov oracle for the PickCube task.

    Condition-based state machine: each step re-evaluates all conditions from
    the current state, so recovery from dropped objects is automatic.

    After placing the cube at the goal, holds it still so the env registers
    success, then releases and executes post-task wander phases.
    """

    HOLD_STEPS = 15  # hold at goal for eval/success to trigger

    def __init__(self, max_steps=200, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_steps = max_steps

    def reset(self, ob, info):
        super().reset(ob, info)
        self._hold_counter = 0
        self._goal_pos = None

    def select_action(self, ob, info):
        tcp = info['extra/tcp_pose'][:3].copy()

        if self._task_done:
            action = self._post_task_action(tcp)
            self._step += 1
            if self._step >= self._max_steps:
                self._done = True
            return action

        cube = info['extra/obj_pose'][:3].copy()
        goal = info['extra/goal_pos'][:3].copy()
        grasped = bool(info.get('eval/is_cube_grasped', False))

        above_offset = np.array([0.0, 0.0, 0.10])
        above_threshold = 0.16
        xy_thresh = 0.04
        pos_thresh = 0.02
        goal_thresh = 0.025

        xy_aligned = np.linalg.norm(cube[:2] - tcp[:2]) <= xy_thresh
        pos_aligned = np.linalg.norm(cube - tcp) <= pos_thresh
        goal_xy_aligned = np.linalg.norm(goal[:2] - tcp[:2]) <= xy_thresh
        goal_reached = np.linalg.norm(goal - cube) <= goal_thresh
        above = tcp[2] > above_threshold

        # Hold at goal with zero delta so robot settles (is_robot_static)
        if self._hold_counter > 0:
            self._hold_counter -= 1
            if self._hold_counter == 0:
                self.print_phase('7b: release at goal')
                self._task_done = True
                action = np.array([0.0, 0.0, 0.0, self.GRIPPER_OPEN])
            else:
                self.print_phase('7a: hold still at goal')
                action = np.array([0.0, 0.0, 0.0, self.GRIPPER_CLOSE])
        elif goal_reached and grasped:
            self.print_phase('7: goal reached — holding')
            self._hold_counter = self.HOLD_STEPS
            self._goal_pos = goal.copy()
            action = self.ee_action(tcp, goal, self.GRIPPER_CLOSE)
        elif not grasped:
            if not xy_aligned:
                self.print_phase('1: approach above cube')
                action = self.ee_action(tcp, cube + above_offset, self.GRIPPER_OPEN)
            elif not pos_aligned:
                self.print_phase('2: lower to cube')
                action = self.ee_action(tcp, cube, self.GRIPPER_OPEN)
            else:
                self.print_phase('3: grasp')
                action = self.ee_action(tcp, cube, self.GRIPPER_CLOSE)
        elif not goal_xy_aligned and not above:
            self.print_phase('4: lift')
            lift_target = np.array([tcp[0], tcp[1], above_threshold * 2])
            action = self.ee_action(tcp, lift_target, self.GRIPPER_CLOSE)
        elif not goal_xy_aligned:
            self.print_phase('5: transport')
            transport_target = np.array([
                goal[0], goal[1], max(tcp[2], goal[2] + 0.05),
            ])
            action = self.ee_action(tcp, transport_target, self.GRIPPER_CLOSE)
        else:
            self.print_phase('6: place')
            action = self.ee_action(tcp, goal, self.GRIPPER_CLOSE)

        self._step += 1
        if self._step >= self._max_steps:
            self._done = True

        return action
