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

"""Defines block2absolutelocation reset and reward."""
import enum

from typing import Any, List
from absl import logging
from language_table.environments import blocks as blocks_module
from language_table.environments.rewards import reward as base_reward
from language_table.environments.rewards import synonyms
from language_table.environments.rewards import task_info
import numpy as np


# There's a small offset in the Y direction to subtract.
# The red dots represent the bounds of the arm, which are not exactly in the
# center of the boards.
# This should only matter for this reward, which deals with absolute locations.
X_BUFFER = 0.025

X_MIN_REAL = 0.15
X_MAX_REAL = 0.6
Y_MIN_REAL = -0.3048
Y_MAX_REAL = 0.3048
X_MIN = X_MIN_REAL - X_BUFFER
X_MAX = X_MAX_REAL - X_BUFFER
Y_MIN = Y_MIN_REAL
Y_MAX = Y_MAX_REAL
CENTER_X = (X_MAX - X_MIN) / 2. + X_MIN
CENTER_Y = (Y_MAX - Y_MIN)/2. + Y_MIN

BLOCK2ABSOLUTELOCATION_TARGET_DISTANCE = 0.115
BLOCK2ABSOLUTELOCATION_CENTER_TARGET_DISTANCE = 0.1


class Locations(enum.Enum):
  TOP = 'top'
  TOP_LEFT = 'top_left'
  TOP_RIGHT = 'top_right'
  CENTER = 'center'
  CENTER_LEFT = 'center_left'
  CENTER_RIGHT = 'center_right'
  BOTTOM = 'bottom'
  BOTTOM_LEFT = 'bottom_left'
  BOTTOM_RIGHT = 'bottom_right'


ABSOLUTE_LOCATIONS = {
    'top': [X_MIN, CENTER_Y],
    'top_left': [X_MIN, Y_MIN],
    'top_right': [X_MIN, Y_MAX],
    'center': [CENTER_X, CENTER_Y],
    'center_left': [CENTER_X, Y_MIN],
    'center_right': [CENTER_X, Y_MAX],
    'bottom': [X_MAX, CENTER_Y],
    'bottom_left': [X_MAX, Y_MIN],
    'bottom_right': [X_MAX, Y_MAX],
}

LOCATION_SYNONYMS = {
    'top': ['top side', 'top', 'towards your base'],
    'top_left': ['top left of the board', 'top left',
                 'upper left corner', 'top left corner'],
    'top_right': ['top right of the board', 'top right',
                  'upper right corner', 'top right corner'],
    'center': ['middle of the board', 'center of the board',
               'center', 'middle'],
    'center_left': ['left side of the board', 'center left', 'left side'],
    'center_right': ['right side of the board', 'center right', 'right side'],
    'bottom': ['bottom side', 'bottom'],
    'bottom_left': ['bottom left of the board', 'bottom left',
                    'lower left corner', 'bottom left corner'],
    'bottom_right': ['bottom right of the board', 'bottom right',
                     'lower right corner', 'bottom right corner'],
}

BLOCK2ABSOLUTELOCATION_VERBS = [
    'move the',
    'push the',
    'slide the',
]


def generate_all_instructions(block_mode):
  """Generate all instructions for block2relativeposition."""
  all_instructions = []
  all_block_text_descriptions = blocks_module.get_blocks_text_descriptions(
      block_mode)
  for block_text in all_block_text_descriptions:
    for location in ABSOLUTE_LOCATIONS:
      for location_syn in LOCATION_SYNONYMS[location]:
        for verb in BLOCK2ABSOLUTELOCATION_VERBS:
          # Add instruction.
          inst = (f'{verb} {block_text} to the {location_syn}')
          all_instructions.append(inst)
  return all_instructions


class BlockToAbsoluteLocationReward(base_reward.LanguageTableReward):
  """Calculates reward/instructions for 'push block to absolute location'."""

  def __init__(self, goal_reward, rng, delay_reward_steps,
               block_mode):
    super(BlockToAbsoluteLocationReward, self).__init__(
        goal_reward=goal_reward,
        rng=rng,
        delay_reward_steps=delay_reward_steps,
        block_mode=block_mode)
    self._block = None
    self._instruction = None
    self._location = None
    self._target_translation = None

  def _sample_instruction(
      self, block, blocks_on_table, location):
    """Randomly sample a task involving two objects."""
    verb = self._rng.choice(synonyms.PUSH_VERBS)
    # Get some synonym for block.
    block_text = self._rng.choice(
        synonyms.get_block_synonyms(block, blocks_on_table))
    # Get some synonym for location.
    location_syn = self._rng.choice(LOCATION_SYNONYMS[location])
    return f'{verb} {block_text} to the {location_syn}'

  def reset(self, state, blocks_on_table):
    """Chooses new target block and location."""
    # Choose a random block.
    block = self._sample_object(blocks_on_table)

    # Choose a location randomly.
    location = self._rng.choice(list(sorted(ABSOLUTE_LOCATIONS.keys())))

    info = self.reset_to(state, block, location, blocks_on_table)
    # If the state of the board already triggers the reward, try to reset
    # again with a new configuration.
    if self._in_goal_region(state, self._block, self._target_translation):
      # Try again with a new board configuration.
      return task_info.FAILURE
    return info

  def reset_to(
      self, state, block, location, blocks_on_table):
    """Reset to a particular task definition."""
    self._block = block
    # Sample an instruction.
    self._instruction = self._sample_instruction(
        self._block, blocks_on_table, location)
    # Get the corresponding target_translation.
    target_translation = ABSOLUTE_LOCATIONS[location]
    # Cache the target location corresponding to the instruction.
    self._target_translation = np.copy(target_translation)
    self._location = location
    info = self.get_current_task_info(state)
    self._in_reward_zone_steps = 0
    return info

  @property
  def target_translation(self):
    return self._target_translation

  def get_goal_region(self):
    if self._location == Locations.CENTER.value:
      return self._target_translation, BLOCK2ABSOLUTELOCATION_CENTER_TARGET_DISTANCE
    return self._target_translation, BLOCK2ABSOLUTELOCATION_TARGET_DISTANCE

  def reward(self, state):
    """Calculates reward given state."""
    reward, done = self.reward_for(state, self._block, self._target_translation)
    return reward, done

  def reward_for(
      self, state, pushing_block, target_translation):
    """Returns 1. if pushing_block is in location."""
    reward = 0.0
    done = False

    in_goal_region = self._in_goal_region(state, pushing_block,
                                          target_translation)

    if in_goal_region:
      if self._in_reward_zone_steps >= self._delay_reward_steps:
        reward = self._goal_reward
        done = True
      else:
        logging.info('In reward zone for %d steps', self._in_reward_zone_steps)
        self._in_reward_zone_steps += 1
    return reward, done

  def _in_goal_region(self, state, pushing_block,
                      target_translation):
    # Get current location of the target block.
    current_translation, _ = self._get_pose_for_block(pushing_block, state)
    # Compute distance between current translation and target.
    dist = np.linalg.norm(
        np.array(current_translation) - np.array(target_translation))

    if self._location == Locations.CENTER.value:
      target_dist = BLOCK2ABSOLUTELOCATION_CENTER_TARGET_DISTANCE
    else:
      target_dist = BLOCK2ABSOLUTELOCATION_TARGET_DISTANCE

    if dist < target_dist:
      return True
    return False

  def reward_for_info(self, state, info):
    return self.reward_for(state, info.block, info.target_translation)

  def debug_info(self, state):
    """Returns 1. if pushing_block is in location."""
    # Get current location of the target block.
    current_translation, _ = self._get_pose_for_block(
        self._block, state)
    # Compute distance between current translation and target.
    dist = np.linalg.norm(
        np.array(current_translation) - np.array(self._target_translation))
    return dist

  def get_current_task_info(self, state):
    return task_info.Block2LocationTaskInfo(
        instruction=self._instruction,
        block=self._block,
        location=self._location,
        target_translation=self._target_translation)
