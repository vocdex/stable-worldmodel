import numpy as np

from stable_worldmodel.envs.maniskill.oracles.base_oracle import PandaMarkovOracle

from stable_worldmodel.envs.maniskill.composable_tabletop_ms_env import (
    CUBE_HALF_SIZE,
    SPHERE_RADIUS,
    PEG_WIDTH,
    _rest_z,
)

TASKS = ('pick', 'stack', 'push', 'topple', 'rearrange')


class ComposableTabletopOracle(PandaMarkovOracle):
    """Generic oracle for the composable tabletop env.

    Supports multiple task types with any object shape. Objects are
    referenced by index (obj_0, obj_1). The task can be set explicitly
    or sampled randomly at each reset.

    Tasks:
      pick      — pick obj and hold at random goal
      stack     — pick one obj, place on another
      push      — push obj to random target xy (no grasp)
      topple    — swipe at a peg from the side to knock it over
      rearrange — swap positions of two objects
    """

    def __init__(self, task='random', num_objects=2, shapes=None,
                 max_steps=200, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._task_config = task
        self._num_objects = num_objects
        self._shapes = shapes or ['cube'] * num_objects
        self._max_steps = max_steps

    def reset(self, ob, info):
        super().reset(ob, info)
        self._hold_counter = 0
        self._subtask = None
        self._rearrange_phase = 0

        # Select task
        if self._task_config == 'random':
            available = ['pick', 'stack', 'push']
            if any(s == 'peg' for s in self._shapes):
                available.append('topple')
            if self._num_objects >= 2:
                available.append('rearrange')
            self._task = np.random.choice(available)
        else:
            self._task = self._task_config

        # Pick/base indices
        if self._task in ('stack', 'rearrange') and self._num_objects >= 2:
            self._pick_idx = int(np.random.random() < 0.5)
            self._base_idx = 1 - self._pick_idx
        else:
            self._pick_idx = np.random.randint(self._num_objects)
            self._base_idx = None

        # Random targets within comfortable workspace
        self._push_target = np.array([
            np.random.uniform(-0.06, 0.06),
            np.random.uniform(-0.10, 0.10),
        ])
        self._pick_goal = np.array([
            np.random.uniform(-0.06, 0.06),
            np.random.uniform(-0.08, 0.08),
            0.20,
        ])

        # For rearrange: remember initial positions
        if self._task == 'rearrange' and info is not None:
            self._swap_pos_a = info[f'extra/obj_{self._pick_idx}_pose'][:3].copy()
            self._swap_pos_b = info[f'extra/obj_{self._base_idx}_pose'][:3].copy()

    def select_action(self, ob, info):
        tcp = info['extra/tcp_pose'][:3].copy()
        tcp_quat = info['extra/tcp_pose'][3:7].copy()
        self._tcp_quat = tcp_quat  # store for helpers

        if self._task_done:
            action = self._post_task_action(tcp)
            self._step += 1
            if self._step >= self._max_steps:
                self._done = True
            return action

        if self._task == 'pick':
            action = self._pick_action(tcp, info)
        elif self._task == 'stack':
            action = self._stack_action(tcp, info)
        elif self._task == 'push':
            action = self._push_action(tcp, info)
        elif self._task == 'topple':
            action = self._topple_action(tcp, info)
        elif self._task == 'rearrange':
            action = self._rearrange_action(tcp, info)
        else:
            action = self._pick_action(tcp, info)

        self._step += 1
        if self._step >= self._max_steps:
            self._done = True

        return action

    # ----- helpers -----

    def _pick_and_place(self, tcp, obj_pos, target, grasped):
        """Full pick-and-place: approach → grasp → lift → transport → place.

        Uses the OGBench-style condition tree where the height check only
        gates horizontal transport, not final placement.
        Grasp position is slightly above object center to avoid IK limits.
        """
        q = self._tcp_quat
        above_offset = np.array([0.0, 0.0, 0.12])
        above_threshold = 0.20
        xy_thresh = 0.04
        pos_thresh = 0.02

        grasp_pos = obj_pos.copy()

        xy_aligned = np.linalg.norm(grasp_pos[:2] - tcp[:2]) <= xy_thresh
        pos_aligned = np.linalg.norm(grasp_pos - tcp) <= pos_thresh
        target_xy_aligned = np.linalg.norm(target[:2] - tcp[:2]) <= xy_thresh
        above = tcp[2] > above_threshold

        if not grasped:
            if not xy_aligned:
                return self.ee_action(tcp, grasp_pos + above_offset, self.GRIPPER_OPEN, q)
            elif not pos_aligned:
                return self.ee_action(tcp, grasp_pos, self.GRIPPER_OPEN, q)
            else:
                return self.ee_action(tcp, grasp_pos, self.GRIPPER_CLOSE, q)
        elif not target_xy_aligned and not above:
            lift = np.array([tcp[0], tcp[1], above_threshold * 2])
            return self.ee_action(tcp, lift, self.GRIPPER_CLOSE, q)
        elif not target_xy_aligned:
            transport = np.array([
                target[0], target[1],
                max(tcp[2], target[2] + 0.06),
            ])
            return self.ee_action(tcp, transport, self.GRIPPER_CLOSE, q)
        else:
            return self.ee_action(tcp, target, self.GRIPPER_CLOSE, q)

    # ----- task implementations -----

    def _pick_action(self, tcp, info):
        """Pick object and hold at random goal."""
        obj = info[f'extra/obj_{self._pick_idx}_pose'][:3].copy()
        grasped = bool(info.get(f'eval/is_obj_{self._pick_idx}_grasped', False))
        goal = self._pick_goal

        if self._hold_counter > 0:
            self._hold_counter -= 1
            if self._hold_counter == 0:
                self._task_done = True
                return np.array([0.0, 0.0, 0.0, self.GRIPPER_OPEN])
            return np.array([0.0, 0.0, 0.0, self.GRIPPER_CLOSE])

        goal_reached = np.linalg.norm(goal - obj) <= 0.025
        if goal_reached and grasped:
            self._hold_counter = 15
            return self.ee_action(tcp, goal, self.GRIPPER_CLOSE, self._tcp_quat)

        return self._pick_and_place(tcp, obj, goal, grasped)

    def _stack_action(self, tcp, info):
        """Pick one object and stack on another."""
        pick = info[f'extra/obj_{self._pick_idx}_pose'][:3].copy()
        base = info[f'extra/obj_{self._base_idx}_pose'][:3].copy()
        grasped = bool(info.get(f'eval/is_obj_{self._pick_idx}_grasped', False))

        base_top = _rest_z(self._shapes[self._base_idx])
        pick_bottom = _rest_z(self._shapes[self._pick_idx])
        stack_target = base.copy()
        stack_target[2] += base_top + pick_bottom

        stack_xy_aligned = np.linalg.norm(stack_target[:2] - tcp[:2]) <= 0.04
        close_to_stack = stack_xy_aligned and grasped and tcp[2] < stack_target[2] + 0.03

        if close_to_stack:
            self._task_done = True
            return np.array([0.0, 0.0, 0.0, self.GRIPPER_OPEN])

        return self._pick_and_place(tcp, pick, stack_target, grasped)

    def _push_action(self, tcp, info):
        """Push object to target xy without grasping."""
        obj = info[f'extra/obj_{self._pick_idx}_pose'][:3].copy()
        target_xy = self._push_target
        push_height = _rest_z(self._shapes[self._pick_idx])

        obj_to_target = target_xy - obj[:2]
        dist_to_target = np.linalg.norm(obj_to_target)

        if dist_to_target < 0.03:
            self._task_done = True
            retreat = tcp.copy()
            retreat[2] = 0.20
            return self.ee_action(tcp, retreat, self.GRIPPER_CLOSE, self._tcp_quat)

        # Position behind the object (opposite to push direction)
        push_dir = obj_to_target / (dist_to_target + 1e-6)
        behind = obj[:2] - push_dir * 0.05
        behind_pos = np.array([behind[0], behind[1], push_height])

        # Approach from behind
        behind_xy_aligned = np.linalg.norm(behind[:2] - tcp[:2]) <= 0.03
        at_push_height = abs(tcp[2] - push_height) < 0.03

        if not behind_xy_aligned or not at_push_height:
            # Move to position behind the object
            approach = np.array([behind[0], behind[1], max(tcp[2], push_height + 0.08)])
            if np.linalg.norm(approach[:2] - tcp[:2]) > 0.03:
                return self.ee_action(tcp, approach, self.GRIPPER_CLOSE, self._tcp_quat)
            return self.ee_action(tcp, behind_pos, self.GRIPPER_CLOSE, self._tcp_quat)
        else:
            # Push through the object toward target
            push_target = np.array([target_xy[0], target_xy[1], push_height])
            return self.ee_action(tcp, push_target, self.GRIPPER_CLOSE, self._tcp_quat)

    def _topple_action(self, tcp, info):
        """Swipe at a peg from the side to knock it over."""
        obj = info[f'extra/obj_{self._pick_idx}_pose'][:3].copy()
        swipe_height = _rest_z(self._shapes[self._pick_idx])

        if self._subtask is None:
            self._subtask = 'approach'
            # Swipe direction: random perpendicular to peg long axis (x)
            self._swipe_dir = np.array([0.0, np.random.choice([-1, 1]), 0.0])

        offset = obj.copy()
        offset[:2] -= self._swipe_dir[:2] * 0.08  # start 8cm to the side
        offset[2] = swipe_height

        through = obj.copy()
        through[:2] += self._swipe_dir[:2] * 0.08  # end 8cm on the other side
        through[2] = swipe_height

        if self._subtask == 'approach':
            # Move above then down to swipe start
            above_start = offset.copy()
            above_start[2] = 0.15
            if tcp[2] > 0.10 and np.linalg.norm(tcp[:2] - offset[:2]) > 0.03:
                return self.ee_action(tcp, above_start, self.GRIPPER_CLOSE, self._tcp_quat)
            if abs(tcp[2] - swipe_height) > 0.02:
                return self.ee_action(tcp, offset, self.GRIPPER_CLOSE, self._tcp_quat)
            self._subtask = 'swipe'

        if self._subtask == 'swipe':
            if np.linalg.norm(tcp[:2] - through[:2]) < 0.03:
                self._subtask = 'retreat'
            return self.ee_action(tcp, through, self.GRIPPER_CLOSE, self._tcp_quat)

        # retreat
        retreat = tcp.copy()
        retreat[2] = 0.20
        if tcp[2] > 0.18:
            self._task_done = True
        return self.ee_action(tcp, retreat, self.GRIPPER_CLOSE, self._tcp_quat)

    def _rearrange_action(self, tcp, info):
        """Swap positions of two objects (pick A → B's spot, pick B → A's spot)."""
        if self._rearrange_phase == 0:
            # Phase 0: pick obj_A, place at obj_B's original position
            pick = info[f'extra/obj_{self._pick_idx}_pose'][:3].copy()
            grasped = bool(info.get(f'eval/is_obj_{self._pick_idx}_grasped', False))
            target = self._swap_pos_b.copy()

            target_xy_aligned = np.linalg.norm(target[:2] - tcp[:2]) <= 0.04
            close = target_xy_aligned and grasped and tcp[2] < target[2] + 0.03

            if close:
                self._rearrange_phase = 1
                return np.array([0.0, 0.0, 0.0, self.GRIPPER_OPEN])

            return self._pick_and_place(tcp, pick, target, grasped)

        else:
            # Phase 1: pick obj_B, place at obj_A's original position
            pick = info[f'extra/obj_{self._base_idx}_pose'][:3].copy()
            grasped = bool(info.get(f'eval/is_obj_{self._base_idx}_grasped', False))
            target = self._swap_pos_a.copy()

            target_xy_aligned = np.linalg.norm(target[:2] - tcp[:2]) <= 0.04
            close = target_xy_aligned and grasped and tcp[2] < target[2] + 0.03

            if close:
                self._task_done = True
                return np.array([0.0, 0.0, 0.0, self.GRIPPER_OPEN])

            return self._pick_and_place(tcp, pick, target, grasped)
