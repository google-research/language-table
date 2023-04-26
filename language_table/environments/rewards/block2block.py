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

"""Defines block2block reset and reward."""
import itertools

from absl import logging
from language_table.environments import blocks as blocks_module
from language_table.environments.rewards import constants
from language_table.environments.rewards import reward as base_reward
from language_table.environments.rewards import synonyms
from language_table.environments.rewards import task_info
import numpy as np


def generate_all_instructions(block_mode):
  """Generates all block2block instructions."""
  all_instructions = []
  all_block_text_descriptions = blocks_module.get_blocks_text_descriptions(
      block_mode)
  for start_block_text, target_block_text in itertools.permutations(
      all_block_text_descriptions, 2):
    for verb in synonyms.PUSH_VERBS:
      for preposition in synonyms.PREPOSITIONS:
        inst = f'{verb} {start_block_text} {preposition} {target_block_text}'
        all_instructions.append(inst)
  return all_instructions


# pytype: skip-file
class BlockToBlockReward(base_reward.LanguageTableReward):
  """Block2block reward."""

  def _sample_instruction(
      self, start_block, target_block, blocks_on_table):
    """Randomly sample a task involving two objects."""
    verb = self._rng.choice(synonyms.PUSH_VERBS)
    # Sample synonyms for start and target blocks.
    start_syn = self._rng.choice(
        synonyms.get_block_synonyms(start_block, blocks_on_table))
    target_syn = self._rng.choice(
        synonyms.get_block_synonyms(target_block, blocks_on_table))
    preposition = self._rng.choice(synonyms.PREPOSITIONS)
    return f'{verb} {start_syn} {preposition} {target_syn}'

  def reset(self, state, blocks_on_table):
    """Resets the start/target objects and returns a text instruction."""
    # pick two objects sufficiently far away and get their poses.
    # track start object and target object poses.
    max_attempts = 10
    num_attempts = 0
    while True:
      start_block, target_block = self._sample_objects(blocks_on_table)
      start_translation, _ = self._get_pose_for_block(
          start_block, state)
      target_translation, _ = self._get_pose_for_block(
          target_block, state)
      dist = np.linalg.norm(
          np.array(start_translation) - np.array(target_translation))
      if dist < constants.TARGET_BLOCK_DISTANCE + 0.01:
        num_attempts += 1
        if num_attempts > max_attempts:
          logging.info(
              'Exceeded max number of attempts to find start/target blocks. '
              'No valid reward found for the current object configuration.')
          return task_info.FAILURE
        continue
      else:
        self._start_block = start_block
        self._target_block = target_block
        break
    self._instruction = self._sample_instruction(
        start_block, target_block, blocks_on_table)
    self._in_reward_zone_steps = 0
    return task_info.Block2BlockTaskInfo(
        instruction=self._instruction,
        block1=self._start_block,
        block2=self._target_block)

  def get_goal_region(self):
    return self._target_translation, constants.TARGET_BLOCK_DISTANCE

  def reward(self, state):
    """Calculates reward given state."""
    # For now only have sparse reward.
    start_translation, _ = self._get_pose_for_block(self._start_block, state)
    target_translation, _ = self._get_pose_for_block(self._target_block, state)

    self._target_translation = target_translation

    # This check ignore whether start block was moved (rather than target object
    # being moved towards start object.
    # TODO(ayzaan): Add smarter logic here.
    dist = np.linalg.norm(
        np.array(start_translation) - np.array(target_translation))
    reward = 0.0
    done = False
    if dist < constants.TARGET_BLOCK_DISTANCE:
      if self._in_reward_zone_steps >= self._delay_reward_steps:
        reward = self._goal_reward
        done = True
      else:
        logging.info('In reward zone for %d steps', self._in_reward_zone_steps)
        self._in_reward_zone_steps += 1

    return reward, done
