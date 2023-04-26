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

"""Defines block2relativelocation reset and reward."""

from absl import logging
from language_table.environments import blocks as blocks_module
from language_table.environments.rewards import reward as base_reward
from language_table.environments.rewards import synonyms
from language_table.environments.rewards import task_info
import numpy as np


MAGNITUDES = {
    'near': 0.15,
    'far': 0.25
}

# Cardinal directions you can push the block. The top left of board
# is (height=0, width=0), so UP == -1, LEFT == -1, etc.
UP = -1.
DOWN = 1.
LEFT = -1.
RIGHT = 1.

DIRECTIONS = {
    'up': [UP, 0.],  # good.
    'down': [DOWN, 0.],  # good.
    'left': [0., LEFT],  # good.
    'right': [0., RIGHT],  # good.
    'diagonal_up_left': [UP, LEFT] / np.linalg.norm([UP, LEFT]),
    'diagonal_up_right': [UP, RIGHT] / np.linalg.norm([UP, RIGHT]),
    'diagonal_down_left': [DOWN, LEFT] / np.linalg.norm([DOWN, LEFT]),
    'diagonal_down_right': [DOWN, RIGHT] / np.linalg.norm([DOWN, RIGHT]),
}


BLOCK2RELATIVELOCATION_VERBS = [
    'move the',
    'push the',
    'slide the',
]


# These cover [up, down, left, right].
SLIGHTLY_PREPOSITION_SYNONYMS = [
    'slightly',
    'slightly to the',
    'a bit',
    'a bit to the',
    'a little',
    'a little to the',
    'a little bit to the',
    'somewhat',
    'somewhat to the',
]


SLIGHTLY_SYNONYMS = [
    'slightly',
    'a bit',
    'a little',
    'a little bit',
    'somewhat',
]

BLOCK2RELATIVELOCATION_MODE_TO_PREPOSITIONS = {
    'near': SLIGHTLY_PREPOSITION_SYNONYMS,
    # empty string because 'push the red block left' is valid.
    'far': ['to the', '']
}


DIRECTION_SYNONYMS = {
    'up': ['up', 'upwards'],
    'down': ['down', 'downwards'],
    'left': ['to the left', 'left'],
    'right': ['to the right', 'right']
}

DIAGONAL_PREPOSITIONS = [
    '%s and %s',
    '%s and then %s',
    'diagonally %s and %s',
    '%s and %s diagonally',
]


BLOCK2RELATIVELOCATION_TARGET_DISTANCE = 0.1


def create_slightly_instruction(rng, verb, block, direction):
  """Created a `slightly` modified instruction."""
  mode = rng.choice(['slightly_first', 'prefix', 'suffix'])
  if mode == 'slightly_first':
    # e.g. 'slightly push the blue cube down and to the right'
    inst = f'slightly {verb} {block} {direction}'
  elif mode == 'prefix':
    # e.g. 'push the blue cube slightly down and to the right'
    slightly_syn = rng.choice(SLIGHTLY_SYNONYMS)
    inst = f'{verb} {block} {slightly_syn} {direction}'
  else:
    # e.g. 'push the blue cube down and right slightly'
    slightly_syn = rng.choice(SLIGHTLY_SYNONYMS)
    inst = f'{verb} {block} {direction} {slightly_syn}'
  return inst


def enumerate_slightly_instruction(verb, block, direction):
  # Slightly first.
  instructions = [f'slightly {verb} {block} {direction}']
  for slightly_syn in SLIGHTLY_SYNONYMS:
    # prefix.
    instructions.append(f'{verb} {block} {slightly_syn} {direction}')
    # suffix.
    instructions.append(f'{verb} {block} {direction} {slightly_syn}')
  for instruction in instructions:
    yield instruction


def get_diagonal_direction(rng, direction):
  """Map canonical diagonal direction `diagonal_up_right' to natural lang."""
  # Break out e.g. 'up', and 'right'
  _, first_dir, second_dir = direction.split('_')
  # Choose synonyms, e.g. 'right' -> 'to the right'.
  first_dir = rng.choice(DIRECTION_SYNONYMS[first_dir])
  second_dir = rng.choice(DIRECTION_SYNONYMS[second_dir])
  # Choose a diagonal preposition, e.g. '<dir> and then <dir>'.
  diagonal_prep = rng.choice(DIAGONAL_PREPOSITIONS)
  # Insert the actual directions, e.g. 'up and then to the right'.
  diagonal_direction = diagonal_prep % (first_dir, second_dir)
  return diagonal_direction


def enumerate_diagonal_direction(direction):
  """Map canonical diagonal direction `diagonal_up_right' to natural lang."""
  # Break out e.g. 'up', and 'right'
  _, first_dir, second_dir = direction.split('_')
  # Choose synonyms, e.g. 'right' -> 'to the right'.
  for first_dir in DIRECTION_SYNONYMS[first_dir]:
    for second_dir in DIRECTION_SYNONYMS[second_dir]:
      # Choose a diagonal preposition, e.g. '<dir> and then <dir>'.
      for diagonal_prep in DIAGONAL_PREPOSITIONS:
        # Insert the actual directions, e.g. 'up and then to the right'.
        diagonal_direction = diagonal_prep % (first_dir, second_dir)
        yield diagonal_direction


def generate_all_instructions(block_mode):
  """Generate all instructions for block2relativeposition."""
  all_instructions = []
  all_block_text_descriptions = blocks_module.get_blocks_text_descriptions(
      block_mode)
  for block_text in all_block_text_descriptions:
    for verb in BLOCK2RELATIVELOCATION_VERBS:
      for direction in DIRECTIONS:
        if 'diagonal' in direction:
          for direction_syn in enumerate_diagonal_direction(direction):
            # Add 'near diagonal' instructions.
            for near_inst in enumerate_slightly_instruction(
                verb, block_text, direction_syn):
              all_instructions.append(near_inst)
            # Add 'far diagonal' instructions.
            inst = f'{verb} {block_text} {direction_syn}'
            all_instructions.append(inst)
        else:
          for direction_syn in DIRECTION_SYNONYMS[direction]:
            # Add 'near' instructions.
            for near_inst in enumerate_slightly_instruction(
                verb, block_text, direction_syn):
              all_instructions.append(near_inst)
            # Add 'far' instructions.
            inst = f'{verb} {block_text} {direction_syn}'
            all_instructions.append(inst)
  return all_instructions


class BlockToRelativeLocationReward(base_reward.LanguageTableReward):
  """Calculates reward/instructions for 'push block to relative location'."""

  def _sample_instruction(
      self, block, distance_mode, direction, blocks_on_table):
    """Randomly sample a task involving two objects."""
    verb = self._rng.choice(BLOCK2RELATIVELOCATION_VERBS)
    # Sample synonym for block.
    block_syn = self._rng.choice(
        synonyms.get_block_synonyms(block, blocks_on_table))
    if 'diagonal' in direction:
      # Map canonical diagonal dir to natural language. E.g.
      # diagonal_up_right -> 'up and to the right diagonally'.
      direction = get_diagonal_direction(self._rng, direction)
    else:
      direction = self._rng.choice(DIRECTION_SYNONYMS[direction])
    if distance_mode == 'near':
      # Modify this with language like 'slightly'.
      inst = create_slightly_instruction(self._rng, verb, block_syn, direction)
    else:
      inst = f'{verb} {block_syn} {direction}'
    return inst

  def reset(self, state, blocks_on_table):
    """Chooses new target block, direction, and distance."""
    cnt = 0
    max_tries = 100
    while True:
      # Choose a random block.
      self._block = self._sample_object(blocks_on_table)

      # Get the current block location.
      block_translation, _ = self._get_pose_for_block(
          self._block, state)

      # Choose a direction.
      direction = self._rng.choice(sorted(list(DIRECTIONS.keys())))

      # Choose a magnitude.
      distance_mode = self._rng.choice(sorted(list(MAGNITUDES.keys())))
      magnitude = MAGNITUDES[distance_mode]

      # Define target_vector as % change in H,W.
      target_vector = np.array(DIRECTIONS[direction]) * magnitude
      # Define target_translation = direction * magnitude.
      target_translation = block_translation + target_vector

      # Only keep if target_translation is inside workspace bounds.
      if base_reward.target_inside_bounds(target_translation):
        break
      cnt += 1

      if cnt > max_tries:
        # Try again with a new board configuration.
        return task_info.FAILURE
    # Choose an instruction.
    self._instruction = self._sample_instruction(
        self._block, distance_mode, direction, blocks_on_table)

    # Cache the target location corresponding to the instruction.
    self._target_translation = np.copy(target_translation)

    self._in_reward_zone_steps = 0
    return task_info.Block2RelativeLocationTaskInfo(
        instruction=self._instruction,
        block=self._block,
        location=direction,
        target_translation=self._target_translation)

  def get_goal_region(self):
    return self._target_translation, BLOCK2RELATIVELOCATION_TARGET_DISTANCE

  def reward(self, state):
    """Calculates reward given state."""
    # Get current location of the target block.
    current_translation, _ = self._get_pose_for_block(
        self._block, state)

    # Compute distance between current translation and target.
    dist = np.linalg.norm(
        np.array(current_translation) - np.array(self._target_translation))
    reward = 0.0
    done = False
    if dist < BLOCK2RELATIVELOCATION_TARGET_DISTANCE:
      if self._in_reward_zone_steps >= self._delay_reward_steps:
        reward = self._goal_reward
        done = True
      else:
        logging.info('In reward zone for %d steps', self._in_reward_zone_steps)
        self._in_reward_zone_steps += 1

    return reward, done
