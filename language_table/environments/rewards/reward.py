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

"""LanguageTable base reward class."""
from typing import Any, List, Tuple
from language_table.environments import blocks as blocks_module
from language_table.environments import constants
from language_table.environments.rewards import synonyms
import numpy as np


class LanguageTableReward(object):
  """Base class for all 2d board rewards."""

  def __init__(self, goal_reward, rng, delay_reward_steps,
               block_mode):
    self._block_mode = block_mode
    self._goal_reward = goal_reward
    self._rng = rng
    # TODO(tding): Handle this in all rewards
    self._delay_reward_steps = delay_reward_steps
    self._in_reward_zone_steps = None

    self._target_translation = None

  def seed(self, rng):
    self._rng = rng

  def get_goal_region(self):
    """Returns the (target translation, radius) tuple."""
    return None, None

  def _get_block_synonym(self, block, blocks_on_table):
    return self._rng.choice(synonyms.get_block_synonyms(block, blocks_on_table))

  def _get_pose_for_block(self, block, state):
    state_translation_key = 'block_%s_translation' % block
    state_rotation_key = 'block_%s_orientation' % block
    return state[state_translation_key], state[state_rotation_key]

  def _get_translation_for_block(self, block, state):
    return np.array(self._get_pose_for_block(block, state)[0])

  def _sample_object(self, blocks_on_table):
    """Choose one of the blocks randomly."""
    block = self._rng.choice(blocks_on_table)
    return block

  def _sample_objects(self, blocks_on_table):
    """Randomly sample two objects."""
    start_block, target_block = self._rng.choice(
        blocks_on_table, 2, replace=False)
    return start_block, target_block


def target_inside_bounds(
    target, buffer=constants.WORKSPACE_BOUNDS_BUFFER):
  target_x, target_y = target
  return (target_x > constants.X_MIN + buffer and
          target_x < constants.X_MAX - buffer and
          target_y > constants.Y_MIN + buffer and
          target_y < constants.Y_MAX - buffer)
