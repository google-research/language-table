# coding=utf-8
# Copyright 2023 The Language Tale Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Oracle uses RRT to avoid obstacles and slows to precise actions near goal."""

import collections

from absl import logging

from language_table.environments import constants
from language_table.environments.oracles import oriented_push_oracle
from language_table.environments.oracles import rrt_star
from language_table.environments.rewards import block2block_relative_location

import numpy as np

# Put a sphere with this diameter around each block in the environment.
# This means we don't have to reason about geometry when detecting collisions.
BLOCK_DIAMETER = 0.015
# Wait to get this close to subgoal before advancing to next subgoal.
ADVANCE_TO_NEXT_SUBGOAL_THRESHOLD = 0.025

PREBLOCK_OFFSET = 0.05
EE_BACKOFF_OFFSETS = [0.06, 0.07, 0.08]

# Think of this as the radius of the sphere around the pushing block that
# determines collisions.
RRT_COLLISION_THRESHOLD = 0.015  # aka `delta`.
# Size of each rrt step.
RRT_STEP_LENGTH = 0.05
# How often rrt samples the end goal instead of a random goal.
RRT_GOAL_SAMPLE_RATE = 0.1
# Search radius when expanding node tree.
RRT_SEARCH_RADIUS = 0.5
# Max iters of RRT to run.
RRT_MAX_ITERS = 1024

# If we fail to get a plan initially, just try direct movement, then replan
# every K steps till we get a workable plan.
REPLAN_IF_FAILURE = True
RETRY_FOR_NEW_PLAN_EVERY = 10

ADVANCE_TO_NEXT_EE_SUBGOAL_THRESHOLD = 0.01
EPS = 1e-5
BEYOND_TABLE_THRESHOLD = 2.0

# Configs for end effector RRT planning.
EE_RRT_STEP_LENGTH = 0.025
EE_RRT_DELTA = 0.01
EE_RRT_OBSTACLE_RADIUS = 0.02
EE_RRT_ITER_MAX = 2048
RETRY_FOR_NEW_EE_PLAN_EVERY = 1

# Allow both ee planning and block planning to consider moving this far outside
# the official boundary.
EXTRA_BOUNDARY_BUFFER = 0.04

# This effectively expands the planning "boundary" just on the sides
# non-adjacent to the robot.
X_RANGE_RRT = (
    constants.X_MIN,  # keep x_min boundary.
    constants.X_MAX + EXTRA_BOUNDARY_BUFFER)
Y_RANGE_RRT = (constants.Y_MIN - EXTRA_BOUNDARY_BUFFER,
               constants.Y_MAX + EXTRA_BOUNDARY_BUFFER)


def filter_subgoals(path, min_distance):
  """Make sure subgoals are all at least min_distance apart."""
  # List of subgoals to keep. Rightmost is earliest.
  keep = collections.deque([])
  # Always keep the first subgoal.
  keep.append(path.pop())
  # Traverse rest of path.
  for _ in range(len(path)):
    # If current subgoal is min dist away from latest in keep, add it.
    candidate = path.pop()
    dist = np.linalg.norm(np.array(candidate) - np.array(keep[0]))
    if dist >= min_distance:
      keep.appendleft(candidate)
  return keep


class ObstacleOrientedPushOracleBoard2dRRT(
    oriented_push_oracle.OrientedPushOracle):
  """Push oracle for language table that avoids obstacles."""

  def __init__(self,
               env,
               use_ee_planner,
               action_noise_std=0.0,
               slowdown_freespace=False,
               backoff_subgoal_rrt=True,
               replan_ee_rrt=True,
               backoff_ee_rrt=True,
               filter_ee_obstacle_poses=True,
               debug_logging=False,
               block_diameter=BLOCK_DIAMETER,
               rrt_collision_threshold=RRT_COLLISION_THRESHOLD):
    super(ObstacleOrientedPushOracleBoard2dRRT, self).__init__(env)
    self._block_diameter = block_diameter
    self._rrt_collision_threshold = rrt_collision_threshold
    self._env = env
    self._use_ee_planner = use_ee_planner
    self._np_random_state = np.random.RandomState(0)
    self.phase = "move_to_pre_block_avoid"

    self._pid_spheres = self._setup_env(env)
    self._action_noise_std = action_noise_std
    self._current_rrt_target = None
    self._replan = REPLAN_IF_FAILURE
    self._replan_counter = 0
    self._ee_replan_counter = 0
    self._need_ee_replan = None

    self._current_ee_target = None
    self._ee_plan = None
    self._ee_plan_success = None
    # Cache instruction and replan if we see a new one.
    self._prev_instruction = None
    self._logging = debug_logging
    self._slowdown_freespace = slowdown_freespace
    # If you fail to obtain an rrt plan, it could be because we are pushing
    # one block very close to another. If this is flag is enabled, try
    # backing off the original goal and plan to that offset instead.
    self._backoff_subgoal_rrt = backoff_subgoal_rrt
    # If we fail to obtain an ee rrt plan, it could be because the end effector
    # just completed a task and is too close to the block it just pushed.
    # If this flag is enabled, try falling back on "potential" avoidance for a
    # few steps (just to move away from the nearby block), then replan.
    self._replan_ee_rrt = replan_ee_rrt
    self._backoff_ee_rrt = backoff_ee_rrt
    self._filter_ee_obstacle_poses = filter_ee_obstacle_poses

  def _setup_env(self, env):
    """Add debug objects and set camera pose of visualizer."""
    p = self._env.pybullet_client

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

    # Render sphere at the target position.
    # Rendering is toggled via alpha chan, which is not ideal a) it's slow and
    # b) tiny-renderer doesn't support alpha blending. OK for now.
    pid_spheres = []
    # Disable mouse picking and use our own 2D interface for control.
    # p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    return pid_spheres

  def _get_obstacle_poses(self, raw_state):
    obstacle_poses = [
        raw_state[i][:2]
        for i in raw_state.keys()
        if i.startswith("block_") and "translation" in i
    ]
    # Filter to just those on table.
    obstacle_poses = [i for i in obstacle_poses
                      if np.max(i) < BEYOND_TABLE_THRESHOLD]
    return obstacle_poses

  @property
  def plan_success(self):
    return self.plan_success

  def get_plan(self, raw_state):
    """Create a list of subgoals to reach that have no collisions."""
    # Get position of end effector.
    xy_ee = raw_state["effector_target_translation"][:2]

    # Get position of task target. This is where we push to.
    xy_target = xy_ee + raw_state[
        "effector_target_to_task_target_translation"][:2]

    # Get position of task block. This is the one we push.
    xy_block = xy_ee + raw_state[
        "effector_target_to_start_block_translation"][:2]
    # Get all the obstacles in the scene.
    obstacle_poses = self._get_obstacle_poses(raw_state)

    # Exclude the block we are pushing from obstacles.
    obstacle_poses = [i for i in obstacle_poses
                      if np.linalg.norm(xy_block - i) > EPS]

    # If the xy_target is a block center of mass, we are doing 'block2block',
    # so remove it from the list of obstacles.
    obstacle_poses = [i for i in obstacle_poses
                      if np.linalg.norm(xy_target - i) > EPS]

    def _get_plan(cand_xy_target,
                  obstacle_w=self._block_diameter,
                  delta=self._rrt_collision_threshold):
      obstacle_widths = [obstacle_w for _ in obstacle_poses]
      # Create an open-loop collision free plan with RRT-* that gives us a list
      # of x,y subgoals to hit.
      path, success = rrt_star.get_shortest_path_no_collisions(
          xy_start=xy_block,
          xy_goal=cand_xy_target,
          x_range=X_RANGE_RRT,
          y_range=Y_RANGE_RRT,
          obstacle_xy=obstacle_poses,
          obstacle_widths=obstacle_widths,
          delta=delta,
          step_length=RRT_STEP_LENGTH,
          goal_sample_rate=RRT_GOAL_SAMPLE_RATE,
          search_radius=RRT_SEARCH_RADIUS,
          iter_max=RRT_MAX_ITERS,
          plot=False)
      path = collections.deque(path)
      return path, success

    # Try to get a plan.
    path, success = _get_plan(xy_target)

    if not success and self._backoff_subgoal_rrt:
      # Check if the xy_goal is very close to one block. This is typically the
      # case for BOARD2D_BLOCK2BLOCK_RELATIVE_LOCATION.
      is_block2block_relative = [
          i for i in obstacle_poses
          if block2block_relative_location.is_block2block_relative_pair(
              i, xy_target)
      ]
      if is_block2block_relative:
        # Get the target block.
        target_block_xy = is_block2block_relative[0]
        # Consider multiple plans, backing off from original target on line away
        # from target_block.
        dir_from_block_to_offset = xy_target - target_block_xy
        for scale in [1.1, 1.2, 1.3, 1.4, 1.5]:
          new_xy_target = (dir_from_block_to_offset*scale) + target_block_xy
          new_path, success = _get_plan(new_xy_target)
          if success:
            # We're now pushing to some (plannable) offset.
            # Add the original target back.
            new_path.appendleft(xy_target)
            path = new_path
            break

    # Make sure final goal is the xy_target.
    assert np.linalg.norm(path[0] - xy_target) < 1e-8

    if success:
      # We found a good plan. No need to replan.
      self._need_replan = False
    elif REPLAN_IF_FAILURE:
      # We couldn't find a plan. Replan after a countdown of just trying to
      # push directly to target.
      self._need_replan = True

    if not success:
      logging.info("RRT failed, likely because the environment was initialized"
                   " with the goal inside an obstacle.")
      self._already_reported_rrt_failure = True

    # The rightmost element is always the xy_start. Pop it.
    if len(path) > 1:
      path.pop()

    # Filter the subgoals such that they are at least
    # ADVANCE_TO_NEXT_SUBGOAL_THRESHOLD apart.
    len_p_before = len(path)
    path = filter_subgoals(path, ADVANCE_TO_NEXT_SUBGOAL_THRESHOLD)
    len_p_after = len(path)
    if self._logging:
      logging.info("Filtered block subgoal path from %d to %d",
                   len_p_before, len_p_after)

    # Set the current subgoal to the next subgoal in line.
    self._current_rrt_target = path.pop()
    self._plan = path
    return success

  def _maybe_advance_to_next_subgoal(self, info, raw_state):
    """Check if block close enough to subgoal, advance to next if so."""
    if (info.distance_to_target <= ADVANCE_TO_NEXT_SUBGOAL_THRESHOLD
        and self._plan):
      # Advance to next target if within threshold and there are subgoals left.
      self._current_rrt_target = self._plan.pop()
      # Update action info with the new target.
      info = self._get_action_info(raw_state)
    return info

  def _maybe_advance_to_next_ee_subgoal(self, info, raw_state):
    """Check if block close enough to subgoal, advance to next if so."""
    # Go 5 cm away from the block, on the line between the block and target.
    xy_delta_to_subgoal = self._current_ee_target - info.xy_ee
    diff_to_subgoal = np.linalg.norm(xy_delta_to_subgoal)
    if diff_to_subgoal < ADVANCE_TO_NEXT_EE_SUBGOAL_THRESHOLD and self._ee_plan:
      # Advance to next target if within threshold and there are subgoals left.
      self._current_ee_target = self._ee_plan.pop()
      # Update action info with the new target.
      info = self._get_action_info(raw_state)
    if not self._ee_plan:
      # Sometimes the pre-block changes on the fly if we make slight contact.
      # Make sure we're always following the latest closed-loop pre_block,
      # not just the last subgoal in an open-loop plan.
      self._current_ee_target = info.xy_pre_block
    return info

  def _maybe_slowdown(self, dist, xy_delta):
    """Slow down if nearby target to more precise actions."""
    dist_thresh = [0.02, 0.04, 0.06, 0.08, 0.1]
    slowdowns = [0.2, 0.3, 0.4, 0.5, 0.6]
    for (thresh, slowdown) in zip(dist_thresh, slowdowns):
      if dist < thresh:
        xy_delta *= slowdown
        break
    return xy_delta

  def reset(self):
    self.phase = "move_to_pre_block_avoid"
    self._current_rrt_target = None
    self._current_ee_target = None
    self._ee_plan = None
    self._replan_counter = 0
    self._ee_replan_counter = 0

  def _get_move_to_preblock_avoid(self, info, raw_state):
    if self._current_ee_target is None and self._use_ee_planner:
      # Get free-space plan if it doesn't exist.
      self._get_ee_plan(raw_state, info)

    # Optionally replan ee-rrt.
    self._ee_replan_counter += 1
    if self._replan_ee_rrt and self._need_ee_replan and (
        self._ee_replan_counter % RETRY_FOR_NEW_EE_PLAN_EVERY == 0):
      self._get_ee_plan(raw_state, info)

    # If we're close enough to latest subgoal, advance to next.
    if self._use_ee_planner:
      info = self._maybe_advance_to_next_ee_subgoal(info, raw_state)
    if self._use_ee_planner and self._ee_plan_success:
      xy_delta, max_step_velocity = self._get_move_to_preblock_avoid_rrt(info)
    else:
      logging.info("FAILED TO PLAN, FALLING BACK ON POTENTIAL AVOIDANCE")
      xy_delta, max_step_velocity = (
          self._get_move_to_preblock_avoid_potential(info))
    return info, xy_delta, max_step_velocity

  def _get_action_for_block_target(self,
                                   raw_state,
                                   block="block",
                                   target="target"):

    if "instruction" in raw_state:
      curr_instruction = raw_state["instruction"]
      if self._prev_instruction is None:
        self._prev_instruction = curr_instruction
      if np.linalg.norm(self._prev_instruction - curr_instruction) > 0.:
        self.reset()
      self._prev_instruction = curr_instruction

    if self._current_rrt_target is None:
      # Generate a non-colliding plan.
      self.get_plan(raw_state)

    self._replan_counter += 1
    if REPLAN_IF_FAILURE and self._need_replan and (
        self._replan_counter % RETRY_FOR_NEW_PLAN_EVERY == 0):
      # Optionally replan.
      self.get_plan(raw_state)

    # Maybe advance to the next subgoal.
    info = self._get_action_info(raw_state)
    info = self._maybe_advance_to_next_subgoal(info, raw_state)

    # Specifying this as velocity makes it independent of control frequency.
    max_step_velocity = 0.35

    if self.phase == "move_to_pre_block_avoid":
      info, xy_delta, max_step_velocity = self._get_move_to_preblock_avoid(
          info, raw_state)

    if self.phase == "move_to_pre_block":
      xy_delta, max_step_velocity = self._get_move_to_preblock(
          info.xy_pre_block, info.xy_ee)

    if self.phase == "move_to_block":
      xy_delta = self._get_move_to_block(info)

    if self.phase == "push_block":
      xy_delta = self._get_push_block(
          info.theta_error, info.theta_threshold_to_orient,
          info.xy_delta_to_touchingblock)

    orient_circle_diameter = 0.025

    if self.phase == "orient_block_left" or self.phase == "orient_block_right":
      max_step_velocity = 0.15

    if self.phase == "orient_block_left":
      xy_delta = self._get_orient_block_left(
          info.xy_dir_block_to_ee,
          orient_circle_diameter,
          info.xy_block,
          info.xy_ee,
          info.theta_error,
          info.theta_threshold_flat_enough)

    if self.phase == "orient_block_right":
      xy_delta = self._get_orient_block_right(
          info.xy_dir_block_to_ee,
          orient_circle_diameter,
          info.xy_block,
          info.xy_ee,
          info.theta_error,
          info.theta_threshold_flat_enough)

    if self._action_noise_std != 0.0:
      xy_delta += (self._np_random_state.randn(2) *
                   self._action_noise_std)

    max_step_distance = max_step_velocity * (1 /
                                             self._env.get_control_frequency())

    # Apply slowdown unless we are moving in freespace.
    moving_in_freespace = self.phase == "move_to_pre_block_avoid"
    apply_slowdown = True if not moving_in_freespace else False
    if self._slowdown_freespace:
      # Also apply it to freespace movement.
      apply_slowdown = True
    if apply_slowdown:
      before_max_step_distance = max_step_distance
      max_step_distance = self._maybe_slowdown(
          info.distance_to_target, max_step_distance)
      if self._logging:
        logging.info("subgoal %d, info.distance_to_target: %.3f, "
                     "max_step_before: %.3f, max_step_after: %.3f",
                     len(self._plan), info.distance_to_target,
                     before_max_step_distance, max_step_distance)
    length = np.linalg.norm(xy_delta)
    if length > max_step_distance:
      xy_direction = xy_delta / length
      xy_delta = xy_direction * max_step_distance
    return xy_delta

  def _get_filtered_ee_obstacle_poses(
      self, obstacle_poses, xy_target, pushing_block):
    # Ignore obstacles that are already "in contact" with xy_target,
    # as long as they are *not* the block we will be pushing. This would lead to
    # undesirable collisions.
    obstacle_poses_filtered = []
    len_before = len(obstacle_poses)
    for i in obstacle_poses:
      in_collision = np.linalg.norm(i - xy_target) < 0.05
      is_pushing_block = np.linalg.norm(i - pushing_block) < 1e-6
      if in_collision and not is_pushing_block:
        # Skip this block because it is in collision, but not the pushing block.
        continue
      obstacle_poses_filtered.append(i)
    len_after = len(obstacle_poses_filtered)
    logging.info("Filtered ee subgoals from %d to %d", len_before, len_after)
    obstacle_poses = obstacle_poses_filtered
    return obstacle_poses

  def _get_ee_plan(self, raw_state, info):
    """Create a list of subgoals for ee to reach that have no collisions."""
    # Get position of end effector.
    xy_ee = raw_state["effector_target_translation"][:2]

    # Try different "preblock" targets, until you find a good one not
    # initialized in collision.
    xy_target = None
    offsets = [PREBLOCK_OFFSET]
    if self._backoff_ee_rrt:
      offsets += EE_BACKOFF_OFFSETS
    candidate_xy_targets = [
        info.xy_block + -info.xy_dir_block_to_target * i for i in offsets]
    for xy_target in candidate_xy_targets:
      obstacle_poses = self._get_obstacle_poses(raw_state)
      if self._filter_ee_obstacle_poses:
        obstacle_poses = self._get_filtered_ee_obstacle_poses(
            obstacle_poses, xy_target, info.xy_block)

      # Create an open-loop collision free plan with RRT-* that gives us a list
      # of x,y subgoals to hit.
      obstacle_widths = [EE_RRT_OBSTACLE_RADIUS for _ in obstacle_poses]
      path, success = rrt_star.get_shortest_path_no_collisions(
          xy_start=xy_ee,
          xy_goal=xy_target,
          x_range=X_RANGE_RRT,
          y_range=Y_RANGE_RRT,
          obstacle_xy=obstacle_poses,
          obstacle_widths=obstacle_widths,
          delta=EE_RRT_DELTA,
          step_length=EE_RRT_STEP_LENGTH,
          goal_sample_rate=RRT_GOAL_SAMPLE_RATE,
          search_radius=RRT_SEARCH_RADIUS,
          iter_max=EE_RRT_ITER_MAX,
          plot=False)
      if success:
        break

    if success:
      # We found a good plan. No need to replan.
      self._need_ee_replan = False
    elif self._replan_ee_rrt:
      # We couldn't find a plan. Replan after a countdown of just trying to
      # push directly to target.
      self._need_ee_replan = True

    # Filter the subgoals such that they are at least
    # ADVANCE_TO_NEXT_SUBGOAL_THRESHOLD apart.
    len_p_before = len(path)
    path = filter_subgoals(path, ADVANCE_TO_NEXT_EE_SUBGOAL_THRESHOLD)
    len_p_after = len(path)
    if self._logging:
      logging.info("Filtered ee plan from %d to %d", len_p_before, len_p_after)

    # We planned a path to *pre block rrt*, i.e. preblock + small offset to
    # avoid init collisions. We need to add the actual xy_pre_block as the
    # final node.
    final_xy_target = info.xy_pre_block.tolist()
    if not np.linalg.norm(np.array(path[0]) - np.array(final_xy_target)) < EPS:
      path.appendleft(final_xy_target)
    if not success:
      logging.info("RRT failed, likely because the environment was initialized"
                   " with the goal inside an obstacle.")
    # The rightmost element is always the xy_start. Pop it.
    if len(path) > 1:
      path.pop()
    # Set the current subgoal to the next subgoal in line.
    self._current_ee_target = path.pop()
    self._ee_plan = path
    self._ee_plan_success = success

  def _get_move_to_preblock_avoid_rrt(self, info):
    max_step_velocity = 0.3
    # Go 5 cm away from the block, on the line between the block and target.
    xy_delta_to_preblock = self._current_ee_target - info.xy_ee
    diff = np.linalg.norm(xy_delta_to_preblock)
    if diff < 0.001:
      self.phase = "move_to_block"
    xy_delta = xy_delta_to_preblock
    return xy_delta, max_step_velocity

  def _get_move_to_preblock_avoid_potential(self, info):
    max_step_velocity = 0.3
    # Go 5 cm away from the block, on the line between the block and target.
    xy_delta_to_preblock = info.xy_pre_block - info.xy_ee
    xy_delta = np.array([0., 0.])

    def calc_repulsive_potential(x, y, gx, gy):
      dist = np.linalg.norm(np.array([x, y]) - np.array([gx, gy]))
      theta = np.arctan2(gy - y, gx - x)

      r = 0.029
      s = 0.03
      if dist < r:
        dx = -np.sign(np.cos(theta)) * 999999999
        dy = -np.sign(np.sin(theta)) * 999999999
      elif r <= dist <= s + r:
        beta_close = 500
        dx = -beta_close*(s + r - dist)* np.cos(theta)
        dy = -beta_close*(s + r - dist)* np.sin(theta)
      elif dist > s + r:
        dx = 0
        dy = 0
      return np.array([dx, dy])

    pf_delta = np.array([0., 0.])

    for pose_xy in info.obstacle_poses:
      potential = calc_repulsive_potential(info.xy_ee[0], info.xy_ee[1],
                                           pose_xy[0], pose_xy[1])
      pf_delta += potential

    # Attractive force on goal.
    def calc_attractive_potential(x, y, gx, gy):
      alpha = 300
      dist = np.linalg.norm(np.array([x, y]) - np.array([gx, gy]))
      theta = np.arctan2(gy - y, gx - x)

      r = 0.03
      s = 0.03
      dx, dy = 0.0, 0.0
      if dist > r*2:
        dx = alpha * s * np.cos(theta)
        dy = alpha * s* np.sin(theta)
      elif r <= dist <= r*2:
        alpha_close = 550
        dx = alpha_close * s * np.cos(theta)
        dy = alpha_close * s * np.sin(theta)
      elif dist < r:
        alpha_close = 1000
        dx = alpha_close * r * np.cos(theta)
        dy = alpha_close * r * np.sin(theta)
      else:
        dx = alpha*(dist - r) * np.cos(theta)
        dy = alpha*(dist - r) * np.sin(theta)

      return np.array([dx, dy])

    for pose_xy in [info.xy_pre_block]:
      potential = calc_attractive_potential(info.xy_ee[0], info.xy_ee[1],
                                            pose_xy[0], pose_xy[1])
      pf_delta += potential

    xy_delta += pf_delta

    diff = np.linalg.norm(xy_delta_to_preblock)
    if diff < 0.015:
      xy_delta = xy_delta_to_preblock
    if diff < 0.01:
      self.phase = "move_to_block"
      xy_delta = xy_delta_to_preblock

    return xy_delta, max_step_velocity

  def _get_move_to_block(self, info):
    diff = np.linalg.norm(info.xy_delta_to_nexttoblock)
    if diff < 0.01:
      self.phase = "push_block"
    # If need to re-oorient, then re-orient.
    if info.theta_error > info.theta_threshold_to_orient:
      self.phase = "orient_block_left"
    if info.theta_error < -info.theta_threshold_to_orient:
      self.phase = "orient_block_right"
    # Otherwise, push into the block.
    xy_delta = info.xy_delta_to_nexttoblock

    return xy_delta

  def _get_xy_target_block(self, raw_state):
    # Get position of end effector.
    xy_ee = raw_state["effector_target_translation"][:2]
    xy_block = xy_ee + raw_state[
        "effector_target_to_start_block_translation"][:2]
    return xy_block

  def _is_crescent(self, xy_block, raw_state):
    crescent_xys = [
        raw_state[i][:2]
        for i in raw_state.keys()
        if "moon" in i and "mask" not in i
    ]
    for crescent in crescent_xys:
      if np.linalg.norm(xy_block - crescent) < 0.0001:
        return True
    return False

  def _get_action_info(self, raw_state, unused_block=None, unused_target=None):
    # Get position of end effector.
    xy_ee = raw_state["effector_target_translation"][:2]

    # Get the final task target (as specified by the reward).
    xy_task_target = xy_ee + raw_state[
        "effector_target_to_task_target_translation"][:2]

    # Get position of next subgoal on a non-colliding path.
    # This is where we push to.
    xy_rrt_target = self._current_rrt_target

    # The final xy_rrt_target should be equal to the xy_task_target, but
    # this is not always the case (e.g. if it's a block and the block moves).
    # So we always just use the latest xy_task_target if we're all done
    # following intermediate rrt subgoals.
    if self._plan:
      # There are still subgoals to follow.
      xy_target = xy_rrt_target
    else:
      # No more subgoals to follow. Use the reward-provided xy_target
      # (which may change dynamically if, say, a target block moves).
      xy_target = xy_task_target

    # Get position of task block. This is the one we push.
    xy_block = xy_ee + raw_state[
        "effector_target_to_start_block_translation"][:2]
    theta_block = raw_state["start_block_orientation"]

    xy_block_to_target = xy_target - xy_block
    xy_dir_block_to_target = (xy_block_to_target) / (
        np.linalg.norm(xy_block_to_target) + np.finfo(np.float32).eps)
    theta_to_target = self.get_theta_from_vector(xy_dir_block_to_target)

    is_crescent = self._is_crescent(xy_block, raw_state)
    if not is_crescent:
      theta_error = theta_to_target - theta_block
      theta_threshold_to_orient = 0.2
      theta_threshold_flat_enough = 0.03
      # Block has 4-way symmetry.
      while theta_error > np.pi / 4:
        theta_error -= np.pi / 2.
      while theta_error < -np.pi / 4:
        theta_error += np.pi / 2.
    else:
      # Add 90deg to theta_block since we want to push the indent, or opposite
      # the indent, on the red block.
      theta_error = theta_to_target - theta_block + np.pi / 2
      theta_threshold_to_orient = 0.25
      theta_threshold_flat_enough = 0.02
      # Red block has 2 way symmetry.
      while theta_error < -np.pi / 2:
        theta_error += np.pi
      while theta_error > np.pi / 2:
        theta_error -= np.pi

    xy_pre_block = xy_block + -xy_dir_block_to_target * PREBLOCK_OFFSET
    xy_nexttoblock = xy_block + -xy_dir_block_to_target * 0.03
    xy_touchingblock = xy_block + -xy_dir_block_to_target * 0.01
    xy_delta_to_nexttoblock = xy_nexttoblock - xy_ee
    xy_delta_to_touchingblock = xy_touchingblock - xy_ee

    xy_block_to_ee = xy_ee - xy_block
    xy_dir_block_to_ee = xy_block_to_ee / np.linalg.norm(xy_block_to_ee)

    obstacle_poses = self._get_obstacle_poses(raw_state)
    return oriented_push_oracle.PushingInfo(
        xy_block=xy_block,
        xy_ee=xy_ee,
        xy_pre_block=xy_pre_block,
        xy_dir_block_to_target=xy_dir_block_to_target,
        xy_delta_to_nexttoblock=xy_delta_to_nexttoblock,
        xy_delta_to_touchingblock=xy_delta_to_touchingblock,
        xy_dir_block_to_ee=xy_dir_block_to_ee,
        theta_threshold_to_orient=theta_threshold_to_orient,
        theta_threshold_flat_enough=theta_threshold_flat_enough,
        theta_error=theta_error,
        obstacle_poses=obstacle_poses,
        distance_to_target=np.linalg.norm(xy_block_to_target))
