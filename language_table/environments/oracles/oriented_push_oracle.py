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

"""Oracle for pushing task which orients the block then pushes it."""
import dataclasses

from typing import Any

import numpy as np
from tf_agents.policies import py_policy
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts


@dataclasses.dataclass
class PushingInfo:
  """Holds onto info necessary for pushing state machine."""
  xy_block: Any = None
  xy_ee: Any = None
  xy_pre_block: Any = None
  xy_dir_block_to_target: Any = None
  xy_delta_to_nexttoblock: Any = None
  xy_delta_to_touchingblock: Any = None
  xy_dir_block_to_ee: Any = None
  theta_threshold_to_orient: Any = None
  theta_threshold_flat_enough: Any = None
  theta_error: Any = None
  obstacle_poses: Any = None
  distance_to_target: Any = None


class OrientedPushOracle(py_policy.PyPolicy):
  """Oracle for pushing task which orients the block then pushes it."""

  def __init__(self, env, action_noise_std=0.0):
    super(OrientedPushOracle, self).__init__(env.time_step_spec(),
                                             env.action_spec())
    self._env = env
    self._np_random_state = np.random.RandomState(0)
    self.phase = "move_to_pre_block"
    self._action_noise_std = action_noise_std

  def reset(self):
    self.phase = "move_to_pre_block"

  def get_theta_from_vector(self, vector):
    return np.arctan2(vector[1], vector[0])

  def theta_to_rotation2d(self, theta):
    r = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return r

  def rotate(self, theta, xy_dir_block_to_ee):
    rot_2d = self.theta_to_rotation2d(theta)
    return rot_2d @ xy_dir_block_to_ee

  def _get_action_info(self, raw_state, block, target):
    xy_block = raw_state["%s_translation" % block][:2]
    theta_block = raw_state["%s_orientation" % block]
    xy_target = raw_state["%s_translation" % target][:2]
    xy_ee = raw_state["effector_target_translation"][:2]

    xy_block_to_target = xy_target - xy_block
    xy_dir_block_to_target = (
        xy_block_to_target) / np.linalg.norm(xy_block_to_target)
    theta_to_target = self.get_theta_from_vector(xy_dir_block_to_target)

    theta_error = theta_to_target - theta_block
    # Block has 4-way symmetry.
    while theta_error > np.pi/4:
      theta_error -= np.pi/2.
    while theta_error < -np.pi/4:
      theta_error += np.pi/2.

    xy_pre_block = xy_block + -xy_dir_block_to_target * 0.05
    xy_nexttoblock = xy_block + -xy_dir_block_to_target * 0.03
    xy_touchingblock = xy_block + -xy_dir_block_to_target * 0.01
    xy_delta_to_nexttoblock = xy_nexttoblock - xy_ee
    xy_delta_to_touchingblock = xy_touchingblock - xy_ee

    xy_block_to_ee = xy_ee - xy_block
    xy_dir_block_to_ee = xy_block_to_ee / np.linalg.norm(xy_block_to_ee)

    theta_threshold_to_orient = 0.2
    theta_threshold_flat_enough = 0.03
    return PushingInfo(
        xy_block=xy_block,
        xy_ee=xy_ee,
        xy_pre_block=xy_pre_block,
        xy_delta_to_nexttoblock=xy_delta_to_nexttoblock,
        xy_delta_to_touchingblock=xy_delta_to_touchingblock,
        xy_dir_block_to_ee=xy_dir_block_to_ee,
        theta_threshold_to_orient=theta_threshold_to_orient,
        theta_threshold_flat_enough=theta_threshold_flat_enough,
        theta_error=theta_error)

  def _get_move_to_preblock(self, xy_pre_block, xy_ee):
    max_step_velocity = 0.3
    # Go 5 cm away from the block, on the line between the block and target.
    xy_delta_to_preblock = xy_pre_block - xy_ee
    diff = np.linalg.norm(xy_delta_to_preblock)
    if diff < 0.001:
      self.phase = "move_to_block"
    xy_delta = xy_delta_to_preblock
    return xy_delta, max_step_velocity

  def _get_move_to_block(
      self, xy_delta_to_nexttoblock, theta_threshold_to_orient, theta_error):
    diff = np.linalg.norm(xy_delta_to_nexttoblock)
    if diff < 0.001:
      self.phase = "push_block"
    # If need to re-oorient, then re-orient.
    if theta_error > theta_threshold_to_orient:
      self.phase = "orient_block_left"
    if theta_error < -theta_threshold_to_orient:
      self.phase = "orient_block_right"
    # Otherwise, push into the block.
    xy_delta = xy_delta_to_nexttoblock
    return xy_delta

  def _get_push_block(
      self, theta_error, theta_threshold_to_orient, xy_delta_to_touchingblock):
    # If need to reorient, go back to move_to_pre_block, move_to_block first.
    if theta_error > theta_threshold_to_orient:
      self.phase = "move_to_pre_block"
    if theta_error < -theta_threshold_to_orient:
      self.phase = "move_to_pre_block"
    xy_delta = xy_delta_to_touchingblock
    return xy_delta

  def _get_orient_block_left(self,
                             xy_dir_block_to_ee,
                             orient_circle_diameter,
                             xy_block,
                             xy_ee,
                             theta_error,
                             theta_threshold_flat_enough):
    xy_dir_block_to_ee = self.rotate(0.2, xy_dir_block_to_ee)
    xy_block_to_ee = xy_dir_block_to_ee * orient_circle_diameter
    xy_push_left_spot = xy_block + xy_block_to_ee
    xy_delta = xy_push_left_spot - xy_ee
    if theta_error < theta_threshold_flat_enough:
      self.phase = "move_to_pre_block"
    return xy_delta

  def _get_orient_block_right(self,
                              xy_dir_block_to_ee,
                              orient_circle_diameter,
                              xy_block,
                              xy_ee,
                              theta_error,
                              theta_threshold_flat_enough):
    xy_dir_block_to_ee = self.rotate(-0.2, xy_dir_block_to_ee)
    xy_block_to_ee = xy_dir_block_to_ee * orient_circle_diameter
    xy_push_left_spot = xy_block + xy_block_to_ee
    xy_delta = xy_push_left_spot - xy_ee
    if theta_error > -theta_threshold_flat_enough:
      self.phase = "move_to_pre_block"
    return xy_delta

  def _get_action_for_block_target(self,
                                   raw_state,
                                   block="block",
                                   target="target"):
    # Specifying this as velocity makes it independent of control frequency.
    max_step_velocity = 0.35
    info = self._get_action_info(raw_state, block, target)

    if self.phase == "move_to_pre_block":
      xy_delta, max_step_velocity = self._get_move_to_preblock(
          info.xy_pre_block, info.xy_ee)

    if self.phase == "move_to_block":
      xy_delta = self._get_move_to_block(
          info.xy_delta_to_nexttoblock, info.theta_threshold_to_orient,
          info.theta_error)

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
    length = np.linalg.norm(xy_delta)
    if length > max_step_distance:
      xy_direction = xy_delta / length
      xy_delta = xy_direction * max_step_distance
    return xy_delta

  def _action(self,
              time_step,
              policy_state,
              seed = None):
    if time_step.is_first():
      self.reset()
    raw_state = self._env.compute_state()
    xy_delta = self._get_action_for_block_target(
        raw_state, block="block", target="target")
    return policy_step.PolicyStep(action=np.asarray(xy_delta, dtype=np.float32))
