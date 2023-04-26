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

"""Defines block2block relative location reset and reward."""
import enum
import itertools
from typing import Any, List

from absl import logging
from language_table.environments import blocks as blocks_module
from language_table.environments.rewards import reward as base_reward
from language_table.environments.rewards import synonyms
from language_table.environments.rewards import task_info
import numpy as np

# Magnitudes of relative location in either direction from target block.
MAGNITUDE_X = 0.08
MAGNITUDE_Y = 0.08
MAGNITUDE_X_DIAG = 0.04
MAGNITUDE_Y_DIAG = 0.04

BLOCK_DIAMETER = 0.0075
DELTA = 0.0075
DRAGGED_THRESHOLD = 0.05

# Cardinal directions you can push the block. The top left of board
# is (height=0, width=0), so UP == -1, LEFT == -1, etc.
UP = -1.
DOWN = 1.
LEFT = -1.
RIGHT = 1.

# Block must be at least this close to target for reward == 1.
BLOCK2BLOCK_REL_LOCATION_TARGET_DISTANCE = 0.04


class Locations(enum.Enum):
  UP = 'up'
  DOWN = 'down'
  LEFT = 'left'
  RIGHT = 'right'
  DIAGONAL_UP_LEFT = 'diagonal_up_left'
  DIAGONAL_UP_RIGHT = 'diagonal_up_right'
  DIAGONAL_DOWN_LEFT = 'diagonal_down_left'
  DIAGONAL_DOWN_RIGHT = 'diagonal_down_right'


# Cardinal directions you can push the block.
DIRECTIONS = {
    'up': [UP, 0.],
    'down': [DOWN, 0.],
    'left': [0., LEFT],
    'right': [0., RIGHT],
    'diagonal_up_left': [UP, LEFT],
    'diagonal_up_right': [UP, RIGHT],
    'diagonal_down_left': [DOWN, LEFT],
    'diagonal_down_right': [DOWN, RIGHT],
}

DIRECTION_IDS = [
    'up',
    'down',
    'left',
    'right',
    'diagonal_up_left',
    'diagonal_up_right',
    'diagonal_down_left',
    'diagonal_down_right',
]


def get_unique_task_strings():
  """Returns a dict mapping from task string to a numerical ID."""
  all_task_strings = []
  for start_block in blocks_module.ALL_BLOCKS:
    for target_block in blocks_module.ALL_BLOCKS:
      for direction in DIRECTIONS:
        all_task_strings.append(f'{start_block}-{target_block}-{direction}')

  all_task_strings = sorted(all_task_strings)

  task_string_to_task_id = {}
  for n, task_string in enumerate(all_task_strings):
    task_string_to_task_id[task_string] = n

  return task_string_to_task_id


UNIQUE_TASK_STRINGS = get_unique_task_strings()
NUM_UNIQUE_TASKS = len(UNIQUE_TASK_STRINGS)


def is_block2block_relative_pair(xy_block, xy_target):
  """For given block and candidate location, is this a block2blockrel pair?."""
  for d in DIRECTIONS:
    # Define target_vector = direction * magnitude.
    mag_x = MAGNITUDE_X_DIAG if 'diagonal' in d else MAGNITUDE_X
    mag_y = MAGNITUDE_Y_DIAG if 'diagonal' in d else MAGNITUDE_Y
    target_vector = np.array(DIRECTIONS[d]) * np.array([mag_x, mag_y])
    # Define target_translation (where to push to) as target block translation
    # offset by target_vector.
    target_translation = np.array(xy_block) + target_vector
    if np.linalg.norm(target_translation - xy_target) < 1e-6:
      return True
  return False


BLOCK2BLOCKRELATIVELOCATION_VERBS = [
    'move the',
    'push the',
    'put the',
    'bring the',
    'slide the',
]


DIRECTION_SYNONYMS = {
    'up': ['above the', 'to the top side of the', 'to the top of the'],
    'down': ['below the', 'to the bottom side of the', 'to the bottom of the'],
    'left': ['just left of the', 'to the left of the',
             'left of the', 'to the left side of the'],
    'right': ['just right of the', 'to the right of the',
              'right of the', 'to the right side of the'],
    'diagonal_up_left': ['to the top left side of the',
                         'to the top left of the',
                         'diagonally up and to the left of the'],
    'diagonal_up_right': ['to the top right side of the',
                          'to the top right of the',
                          'diagonally up and to the right of the'],
    'diagonal_down_left': ['to the bottom left side of the',
                           'to the bottom left of the',
                           'diagonally down and to the left of the'],
    'diagonal_down_right': ['to the bottom right side of the',
                            'to the bottom right of the',
                            'diagonally down and to the right of the'],
}


def generate_all_instructions(block_mode):
  """Generate all instructions for block2relativeposition."""
  all_instructions = []
  all_block_text_descriptions = blocks_module.get_blocks_text_descriptions(
      block_mode)
  for block_syn, target_block_syn in itertools.permutations(
      all_block_text_descriptions, 2):
    for verb in BLOCK2BLOCKRELATIVELOCATION_VERBS:
      for direction in DIRECTIONS:
        for direction_syn in DIRECTION_SYNONYMS[direction]:
          # # Add instruction.
          inst = (
              f'{verb} {block_syn} {direction_syn} {target_block_syn}')
          all_instructions.append(inst)
  return all_instructions


class BlockToBlockRelativeLocationReward(base_reward.LanguageTableReward):
  """Calculates reward/instructions for 'push block to relative location'."""

  def __init__(self, goal_reward, rng, delay_reward_steps,
               block_mode):
    super(BlockToBlockRelativeLocationReward,
          self).__init__(goal_reward, rng, delay_reward_steps, block_mode)
    self._target_block = None
    self._block = None
    self._direction = None
    self._instruction = None
    self._target_translation = None

  def get_current_task_info(self, state):
    """Update the task info to handle if the target block has moved."""
    if self._target_block is None:
      raise ValueError('must call .reset first')
    # Get updated target translation.
    target_translation = self.target_translation_for(
        state, self._target_block, self._direction)
    self._target_translation = target_translation
    return task_info.Block2BlockRelativeLocationTaskInfo(
        # Reuse same instruction.
        instruction=self._instruction,
        # Reuse same block.
        block=self._block,
        # Update to fresh target_translation.
        target_translation=self._target_translation,
        target_block=self._target_block,
        direction=self._direction)

  def _sample_instruction(
      self, block, target_block, direction, blocks_on_table):
    """Randomly sample a task involving two objects."""
    verb = self._rng.choice(synonyms.PUSH_VERBS)
    # Sample synonyms for start and target blocks.
    block_syn = self._rng.choice(
        synonyms.get_block_synonyms(block, blocks_on_table))
    target_block_syn = self._rng.choice(
        synonyms.get_block_synonyms(target_block, blocks_on_table))
    direction_syn = self._rng.choice(DIRECTION_SYNONYMS[direction])
    return f'{verb} {block_syn} {direction_syn} {target_block_syn}'

  def reset(self, state, blocks_on_table):
    """Chooses new block, target block, direction; returns instruction.

    Args:
      state: state of the board.
      blocks_on_table: list of blocks on board.
    Returns:
      instruction: text instruction.
      block_idx: the index of the chosen block.
      target_location: the 2d point on the board that defines the target region.
    """
    cnt = 0
    max_tries = 100
    while True:
      # Choose a random block.
      block, target_block = self._sample_objects(blocks_on_table)
      # Choose a direction.
      direction = self._rng.choice(list(DIRECTIONS.keys()))

      # Get target translation.
      target_translation = self.target_translation_for(
          state, target_block, direction)

      # Only keep if target_translation is inside workspace bounds.
      if base_reward.target_inside_bounds(target_translation):
        break
      cnt += 1

      if cnt > max_tries:
        # Try again with a new board configuration.
        return task_info.FAILURE
    info = self.reset_to(state, block, target_block, direction, blocks_on_table)

    self._in_reward_zone_steps = 0
    if self.reward_for(
        state,
        self._block,
        self._target_block,
        self._direction,
        delay_reward_steps=0)[1]:
      # Try again with a new board configuration.
      return task_info.FAILURE
    return info

  @property
  def target_translation(self):
    return self._target_translation

  def reset_to(self, state, block, target_block, direction,
               blocks_on_table):
    self._block = block
    self._target_block = target_block
    # Cache the position of the target block at reset. We use this to determine
    # if the agent has dragged the target block too much.
    self._target_block_reset_translation = np.copy(
        self._get_pose_for_block(target_block, state)[0])
    self._direction = direction
    self._target_translation = self.target_translation_for(
        state, target_block, direction)
    self._instruction = self._sample_instruction(
        self._block, self._target_block, self._direction, blocks_on_table)
    return self.get_current_task_info(state)

  def instruction_for(
      self, pushing_block, target_block, direction, blocks_on_table):
    return self._sample_instruction(
        pushing_block, target_block, direction, blocks_on_table)

  def target_translation_for(self, state, target_block, direction, scale=1.):
    """Get invisible target location, given target block, relative direction."""
    # Get the current target block translation.
    target_block_translation, _ = self._get_pose_for_block(target_block, state)
    # Define target_vector = direction * magnitude.
    mag_x = MAGNITUDE_X_DIAG if 'diagonal' in direction else MAGNITUDE_X
    mag_y = MAGNITUDE_Y_DIAG if 'diagonal' in direction else MAGNITUDE_Y
    target_vector = np.array(DIRECTIONS[direction]) * np.array(
        [mag_x*scale, mag_y*scale])
    # Define target_translation (where to push to) as target block translation
    # offset by target_vector.
    target_translation = np.array(target_block_translation) + target_vector
    return target_translation

  def reward(self, state):
    """Calculates reward given state."""
    reward, done = self.reward_for(state, self._block, self._target_block,
                                   self._direction, self._delay_reward_steps)
    return reward, done

  def get_distance_for(self, state, pushing_block, target_block, direction):
    """Gets distance between pushing and target block. Useful for debugging."""
    # Get current location of pushing block.
    pushing_block_translation, _ = self._get_pose_for_block(
        pushing_block, state)
    # Get current relative location of target.
    target_translation = self.target_translation_for(
        state, target_block, direction)
    # Compute distance between current translation and target.
    dist = np.linalg.norm(
        np.array(pushing_block_translation) -
        np.array(target_translation))
    return dist

  def debug_info(self, state):
    """Gets distance between pushing and target block. Useful for debugging."""
    # Get current location of pushing block.
    pushing_block_translation, _ = self._get_pose_for_block(
        self._block, state)
    # Get current relative location of target.
    target_block_translation = self.target_translation_for(
        state, self._target_block, self._direction)
    # Compute distance between current translation and target.
    dist = np.linalg.norm(
        np.array(pushing_block_translation) -
        np.array(target_block_translation))
    return dist

  def reward_for(self, state, pushing_block, target_block, direction,
                 delay_reward_steps):
    """Gets reward for a pushing block and a relative location."""
    pushing_block_translation, _ = self._get_pose_for_block(
        pushing_block, state)
    target_block_translation, _ = self._get_pose_for_block(
        target_block, state)
    # Consider the end point of the line 2x longer than the offset.
    offset_translation = self.target_translation_for(
        state, target_block, direction)

    diff = offset_translation - target_block_translation
    # Consider all points half the distance from target block to offset.
    minpoint = diff * 0.5
    # Consider all points 10% further than the offset.
    maxpoint = diff * 1.1
    # Is the target block somewhere on the line between min point and max point?
    diffs = np.linspace(minpoint, maxpoint, 10)
    pushing_block_on_line = False
    for cand_offset in diffs:
      point = target_block_translation + cand_offset
      dist = np.linalg.norm(point - pushing_block_translation)
      if dist < BLOCK2BLOCK_REL_LOCATION_TARGET_DISTANCE:
        pushing_block_on_line = True
        break

    # Has the target block not been dragged too far from it's original point?
    amount_dragged = np.linalg.norm(self._target_block_reset_translation -
                                    target_block_translation)
    target_block_dragged = amount_dragged > DRAGGED_THRESHOLD

    reward = 0.0
    done = False
    if pushing_block_on_line and not target_block_dragged:
      if self._in_reward_zone_steps >= delay_reward_steps:
        reward = self._goal_reward
        done = True
      else:
        logging.info('In reward zone for %d steps', self._in_reward_zone_steps)
        self._in_reward_zone_steps += 1
    return reward, done

  def get_goal_region(self):
    return self._target_translation, BLOCK2BLOCK_REL_LOCATION_TARGET_DISTANCE

  def reward_for_info(self, state,
                      info):
    """Gets reward based on current block and target info."""
    return self.reward_for(
        state,
        pushing_block=info.block,
        target_block=info.target_block,
        direction=info.direction,
        delay_reward_steps=self._delay_reward_steps)

  def get_current_task_id(self):
    start_block = self._block
    target_block = self._target_block
    direction = self._direction
    task_string = f'{start_block}-{target_block}-{direction}'
    return UNIQUE_TASK_STRINGS[task_string]
