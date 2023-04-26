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

"""Defines separate block from block(s) reset and reward."""

from typing import Any, List, Tuple
from absl import logging
from language_table.environments import blocks as blocks_module
from language_table.environments.rewards import reward as base_reward
from language_table.environments.rewards import task_info
import numpy as np


# Blocks are in need of 'separation' if they are at least this close together.
CONSIDERED_JOINED_THRESHOLD = 0.08  # was 0.05

# How much to move in opposite direction.
MAGNITUDE = 0.1

# Consider task solved when we are this far away from invisible target.
DISTANCE_TO_TARGET_THRESHOLD = 0.025


SEPARATE_TEMPLATES = [
    'pull the %s apart from the %s',
    'move the %s away from the %s',
    'separate the %s from the %s'
]

GROUP_SYNONYMS = ['group', 'clump', 'group of blocks']
REST = 'rest of the blocks'


def generate_all_instructions(block_mode):
  """Generate all instructions for block2relativeposition."""
  all_instructions = []
  all_block_text_descriptions = blocks_module.get_blocks_text_descriptions(
      block_mode)

  for block_syn in all_block_text_descriptions:
    for idx in range(1, len(all_block_text_descriptions)):
      avoid_syns = all_block_text_descriptions[:idx]
      for group_syn in GROUP_SYNONYMS:
        if len(avoid_syns) == len(all_block_text_descriptions)-1:
          # Rare condition where all blocks are together.
          avoid_str = REST
        if len(avoid_syns) == 1:
          # Just one block to avoid.
          avoid_str = avoid_syns[0]
        if len(avoid_syns) == 2:
          # "red and blue".
          avoid_str = '%s and %s' % tuple(avoid_syns)
        if len(avoid_syns) == 3:
          # "red, blue, and green"
          avoid_str = '%s, %s, and %s' % tuple(avoid_syns)
        if len(avoid_syns) >= 4:
          avoid_str = group_syn
        # Choose a high level template.
        for template in SEPARATE_TEMPLATES:
          instruction = template % (block_syn, avoid_str)
          all_instructions.append(instruction)
  return all_instructions


class SeparateBlocksReward(base_reward.LanguageTableReward):
  """Calculates reward/instructions for 'separate block from block(s)'."""

  def __init__(self, goal_reward, rng, delay_reward_steps,
               block_mode):
    super(SeparateBlocksReward, self).__init__(goal_reward, rng,
                                               delay_reward_steps, block_mode)
    self._instruction = None
    self._block = None
    self._avoid_blocks = None
    self._target_translation = None
    self._avoid_centroid_xy = None

  def get_current_task_info(self, state):
    """Return updated task info for "separate blocks"."""
    if self._block is None:
      raise ValueError('must call .reset first')

    # Compute updated "avoid centroid" and implied target_translation,
    # which may have shifted if blocks moved a bit.
    self._target_translation = self.target_translation_for(
        state, self._block, self._avoid_blocks)
    return task_info.SeparateBlocksTaskInfo(
        instruction=self._instruction,
        block=self._block,
        avoid_blocks=self._avoid_blocks,
        target_translation=self._target_translation)

  def _sample_instruction(self, block, avoid_blocks,
                          blocks_on_table):
    """Randomly sample a task involving two objects."""
    # Choose block synonym names.
    block_syn = self._get_block_synonym(block, blocks_on_table)
    avoid_syns = [self._get_block_synonym(i, blocks_on_table)
                  for i in avoid_blocks]
    group_syn = np.random.choice(GROUP_SYNONYMS)
    if len(avoid_syns) == len(blocks_on_table)-1:
      # Rare condition where all blocks are together.
      avoid_str = REST
    if len(avoid_syns) == 1:
      # Just one block to avoid.
      avoid_str = avoid_syns[0]
    if len(avoid_syns) == 2:
      # "red and blue".
      avoid_str = '%s and %s' % tuple(avoid_syns)
    if len(avoid_syns) == 3:
      # "red, blue, and green"
      avoid_specific = '%s, %s, and %s' % tuple(avoid_syns)
      avoid_str = np.random.choice([avoid_specific, group_syn])
    if len(avoid_syns) >= 4:
      avoid_str = group_syn
    # Choose a high level template.
    template = np.random.choice(SEPARATE_TEMPLATES)
    instruction = template % (block_syn, avoid_str)
    return instruction

  def _get_closest_blocks(
      self, block,
      block_translation,
      block_translations):
    # Compute distances to other blocks.
    distances = [(name, np.linalg.norm(block_translation-translation))
                 for (name, translation) in block_translations if name != block]
    # Sort distances.
    distances = sorted(distances, key=lambda x: x[1])
    filtered = [i for i in distances if i[1] < CONSIDERED_JOINED_THRESHOLD]
    if not filtered:
      return [], np.inf
    avg_dist = np.mean([i[1] for i in filtered])
    return [i[0] for i in filtered], avg_dist

  def _get_blocks_to_separate(self, state, blocks_on_table):
    # For each block, compute distances to other blocks on table.
    block_translations = [(i, self._get_translation_for_block(i, state))
                          for i in blocks_on_table]
    block_to_translation = dict(block_translations)

    # Get list of (block, (list of close enough blocks, avg dist)),
    blocks_to_distances = [(src, self._get_closest_blocks(
        src, block_to_translation[src], block_translations))
                           for src in block_to_translation]
    # Sort by avg distance.
    blocks_to_distances = sorted(blocks_to_distances, key=lambda x: x[1][1])

    # Get the push block, the ones we're separating from, and the avg distance.
    # The push_block is the one that is closest to the most blocks.
    push_block, (avoid_blocks, avg_dist) = blocks_to_distances[0]
    return push_block, avoid_blocks, avg_dist

  def _get_avoid_direction(
      self, state, push_block, avoid_blocks):
    # Get position of the push_block.
    push_block_xy = self._get_translation_for_block(push_block, state)
    # Compute the centroid of the "avoid" blocks.
    avoid_centroid_xy = np.mean(
        [self._get_translation_for_block(i, state) for i in avoid_blocks],
        axis=0)
    # Cache avoid_centroid to use in reward.
    self._avoid_centroid_xy = avoid_centroid_xy
    dir_to_centroid = avoid_centroid_xy - push_block_xy
    dir_to_centroid = (dir_to_centroid) / (
        np.linalg.norm(dir_to_centroid) + np.finfo(np.float32).eps)
    return -dir_to_centroid

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
      # Get block we are pushing and the blocks we are pushing away from.
      push_block, avoid_blocks, _ = self._get_blocks_to_separate(
          state, blocks_on_table)

      if not avoid_blocks:
        # All blocks initialized too far from each other. Try again with a diff
        # reset.
        return task_info.FAILURE

      # Get target translation.
      target_translation = self.target_translation_for(
          state, push_block, avoid_blocks)

      # Only keep if target_translation is inside workspace bounds.
      if base_reward.target_inside_bounds(target_translation):
        break
      cnt += 1

      if cnt > max_tries:
        # Try again with a new board configuration.
        return task_info.FAILURE
    info = self.reset_to(state, push_block, avoid_blocks, blocks_on_table)
    return info

  @property
  def target_translation(self):
    return self._target_translation

  def reset_to(self, state, block, avoid_blocks,
               blocks_on_table):
    self._block = block
    self._avoid_blocks = avoid_blocks
    self._target_translation = self.target_translation_for(
        state, block, avoid_blocks)
    self._instruction = self._sample_instruction(
        self._block, self._avoid_blocks, blocks_on_table)
    self._in_reward_zone_steps = 0
    return self.get_current_task_info(state)

  def target_translation_for(self, state, block, avoid_blocks):
    """Get invisible target location to push block to away from avoid_blocks."""
    # Choose the "avoid_direction" as the opposite direction from the
    # center of the "avoid_blocks".
    avoid_direction = self._get_avoid_direction(state, block, avoid_blocks)
    # Define target_vector = direction * magnitude.
    target_vector = avoid_direction * MAGNITUDE
    # Define target_translation (where to push to) as some offset from the
    # center of the blocks you are trying to avoid.
    target_translation = self._avoid_centroid_xy + target_vector
    return target_translation

  def reward(self, state):
    """Calculates reward given state."""
    reward, done = self.reward_for(
        state, self._block, self._target_translation)
    return reward, done

  def debug_info(self, state):
    """Gets distance between pushing and target block. Useful for debugging."""
    dist = self._dist_for(state, self._block, self._target_translation)
    return dist

  def _dist_for(self, state, push_block, target_translation):
    push_block_xy = self._get_translation_for_block(push_block, state)
    dist = np.linalg.norm(push_block_xy - target_translation)
    return dist

  def reward_for(self, state, push_block, target_translation):
    """Gets reward for a pushing block and a relative location."""
    dist = self._dist_for(state, push_block, target_translation)
    reward = 0.0
    done = False
    if dist < DISTANCE_TO_TARGET_THRESHOLD:
      if self._in_reward_zone_steps >= self._delay_reward_steps:
        reward = self._goal_reward
        done = True
      else:
        logging.info('In reward zone for %d steps', self._in_reward_zone_steps)
        self._in_reward_zone_steps += 1
    return reward, done

  def reward_for_info(self, state,
                      info):
    """Gets reward based on current block and target info."""
    return self.reward_for(state,
                           push_block=info.block,
                           target_translation=info.target_translation)
