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
import random

from typing import List
from absl import logging
import gin
from language_table.environments.rewards import constants
from language_table.environments.rewards import reward as base_reward
from language_table.environments.rewards import task_info
import numpy as np

BLOCKS4 = ['red moon', 'blue cube', 'green star', 'yellow pentagon']
BLOCKS8 = [
    'red moon', 'red pentagon', 'blue moon', 'blue cube', 'green cube',
    'green star', 'yellow star', 'yellow pentagon'
]
LOCATIONS = [
    'top left corner', 'top center', 'top right corner', 'center left',
    'center', 'center right', 'bottom left corner', 'bottom center',
    'bottom right corner'
]
ORDERINGS = list(itertools.permutations(BLOCKS4))


def obj_in_place_then_remainder_in_other(blocks, locations):
  instructions = []
  for b in blocks:
    for l0 in locations:
      for l1 in locations:
        if l0 != l1:
          inst = ('put the %s in the %s, then put the rest of the '
                  'blocks in the %s' % (b, l0, l1))
          instructions.append(inst)
  return instructions


def k_in_place_then_k_minus_1_in_other(blocks, locations):
  """Put k blocks in one place, then remainder in another."""
  instructions = []
  numbers = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight']
  numbers = numbers[:len(blocks)]
  for number in numbers[:-1]:
    for l0 in locations:
      for l1 in locations:
        if l0 != l1:
          block = 'block' if number == 'one' else 'blocks'
          inst = ('put %s %s in the %s, then put the '
                  'rest in the %s' % (number, block, l0, l1))
          instructions.append(inst)
  return instructions


def triangle_in_place_remainder_in_rest(locations):
  instructions = []
  for l0 in locations:
    for l1 in locations:
      if l0 != l1:
        inst = ('make a triangle out of three blocks and put it '
                'in the %s of the board, then put the remainder in '
                'the %s of the board' % (l0, l1))
        instructions.append(inst)
  return instructions


def order_by(orderings):
  instructions = []
  for orientation in ['top to bottom', 'left to right']:
    for ordering in orderings:
      o = ', '.join(ordering)
      inst = 'order the blocks from %s: %s' % (orientation, o)
      instructions.append(inst)
  return instructions


def expand_instructions(instructions):
  expanded = []
  for (seed, expansions) in instructions:
    if expansions is None:
      expanded.append(seed)
      continue
    for exp in expansions:
      inst = seed % exp
      expanded.append(inst)
  return expanded


@gin.configurable
def get_100_4block_instructions(num_train_per_family=20,
                                num_test_per_family=5,
                                return_train=True):
  """Gets 20 random from each 'instruction class'."""
  # Seed for determinism.
  train_inst, test_inst = [], []
  random.seed(0)
  def shuffle_split(inst_list):
    random.shuffle(inst_list)
    if num_train_per_family:
      return (inst_list[:num_train_per_family],
              inst_list[num_train_per_family:(
                  num_train_per_family+num_test_per_family)])
    else:
      return inst_list, []
  def accum_train_test(combo, train, test):
    combo_train, combo_test = shuffle_split(combo)
    train.extend(combo_train)
    test.extend(combo_test)
    return train, test
  combo0_instructions = [
      ('put all the blocks in a line', None),
      ('put all the blocks in a %s line', ['horizontal', 'vertical']),
      ('put all the blocks in a vertical line on the %s side of the board',
       ['left', 'center', 'right']),
      ('put all the blocks in a horizontal line on the %s side of the board',
       ['bottom', 'center', 'top']),
      ('put the blocks in a diagonal line from the %s',
       ['top left to bottom right', 'top right to bottom left']),
      ('surround the %s with the other blocks', BLOCKS4),
      ('put all the blocks in the %s', LOCATIONS),
      ('put blocks in all four corners', None),
      ('make a %s shape out of the blocks',
       ['rectangle', 'square', 'diamond', 'parallelogram']),
  ]
  combo0_instructions = expand_instructions(combo0_instructions)
  train_inst, test_inst = accum_train_test(
      combo0_instructions, train_inst, test_inst)
  combo1_instructions = obj_in_place_then_remainder_in_other(BLOCKS4, LOCATIONS)
  train_inst, test_inst = accum_train_test(
      combo1_instructions, train_inst, test_inst)
  combo2_instructions = k_in_place_then_k_minus_1_in_other(BLOCKS4, LOCATIONS)
  train_inst, test_inst = accum_train_test(
      combo2_instructions, train_inst, test_inst)
  combo3_instructions = triangle_in_place_remainder_in_rest(LOCATIONS)
  train_inst, test_inst = accum_train_test(
      combo3_instructions, train_inst, test_inst)
  combo4_instructions = order_by(ORDERINGS)
  train_inst, test_inst = accum_train_test(
      combo4_instructions, train_inst, test_inst)
  if return_train:
    return train_inst
  else:
    return test_inst


BLOCKS8 = [
    'red moon', 'red pentagon', 'blue moon', 'blue cube', 'green cube',
    'green star', 'yellow star', 'yellow pentagon'
]
LOCATIONS = [
    'top left corner', 'top center', 'top right corner', 'center left',
    'center', 'center right', 'bottom left corner', 'bottom center',
    'bottom right corner'
]
COLORS = ['red', 'blue', 'green', 'yellow']


def colors_in_locations():
  """Get tasks for putting colors in locations."""
  all_inst = []
  all_colors_in_all_locations = list(
      itertools.product(
          itertools.permutations(COLORS, 4),
          itertools.permutations(LOCATIONS, 4)))
  for (colors, locations) in all_colors_in_all_locations:
    inst = (f'put the {colors[0]} blocks in the {locations[0]}, '
            f'the {colors[1]} blocks in the {locations[1]}, '
            f'the {colors[2]} blocks in the {locations[2]}, '
            f'and the {colors[3]} blocks in the {locations[3]}.')
    if len(inst) > 256:  # 256 is max length.
      raise ValueError(f'Instruction greater than max length: {inst}')
    all_inst.append(inst)
  return all_inst


def group_color_pairs():
  all_inst = []
  perms = list(itertools.permutations(COLORS, len(COLORS)))
  for (color_i, color_j, color_k, color_l) in perms:
    all_inst.append(
        (f'put the {color_i} and {color_j} blocks together in a group, '
         f'then put the '
         f'{color_k} and {color_l} blocks together in a group.'))
  return all_inst


def group_color_pairs_in_locations():
  """Get tasks for putting pairs of colors in locations."""
  all_inst = []
  color_combos = unique_color_combos()
  location_pairs = list(itertools.permutations(LOCATIONS, 2))
  for (color_i, color_j, color_k, color_l) in color_combos:
    for (loc_i, loc_j) in location_pairs:
      all_inst.append(
          (f'put the {color_i} and {color_j} blocks together in the {loc_i}, '
           f'then put the '
           f'{color_k} and {color_l} blocks together in the {loc_j}.'))
  return all_inst


def unique_color_combos():
  color_combos = list(itertools.combinations(COLORS, 2))
  all_orders = []
  for (color_i, color_j) in color_combos:
    complement = [(i, j)
                  for (i, j) in color_combos
                  if color_i not in [i, j] and color_j not in [i, j]]
    all_orders.append((color_i, color_j, complement[0][0], complement[0][1]))
  return all_orders


def get_colors_in_lines():
  """Get 'put colors in lines' tasks."""
  all_inst = []
  color_combos = unique_color_combos()
  for mode_i in ['horizontal', 'vertical']:
    for mode_j in ['horizontal', 'vertical']:
      for (color_i, color_j, color_k, color_l) in color_combos:
        all_inst.append(
            (f'make one {mode_i} line out of the {color_i} '
             f'and {color_j} blocks, then '
             f'make a {mode_j} line out of the {color_k} and {color_l} blocks'))
  return all_inst


def get_line_tasks():
  """Get put blocks in a line tasks."""
  line_tasks = [
      'put the blocks in a line',
      'put all the blocks in a vertical line',
      'put all the blocks in a horizontal line',
  ]
  for mode in ['left', 'center', 'right']:
    line_tasks.append(
        f'put all the blocks in a vertical line on the {mode} of the board')
  for mode in ['bottom', 'center', 'top']:
    line_tasks.append(
        f'put all the blocks in a horizontal line on the {mode} of the board')
  for mode in ['top left to bottom right', 'top right to bottom left']:
    line_tasks.append(f'put the blocks in a diagonal line from the {mode}')
  return line_tasks


def get_surround_tasks():
  all_inst = []
  for block in BLOCKS8:
    all_inst.append(f'surround the {block} with the others')
  return all_inst


def blocks_in_order_outer_edge():
  """Get instructions for putting all blocks in locations on edge."""
  all_inst = []
  block_orderings = list(itertools.permutations(BLOCKS8, len(BLOCKS8)))
  outer_edge_locations = [
      'top left', 'top center', 'top right', 'center left', 'center right',
      'bottom left', 'bottom center', 'bottom right'
  ]
  for ordering in block_orderings:
    inst = 'put the '
    for (block_i, loc_i) in zip(ordering, outer_edge_locations):
      inst += f'{block_i} to {loc_i}, '
    if len(inst) > 256:  # 256 is max instruction length.
      raise ValueError(f'Instruction greater than max length: {inst}')
    all_inst.append(inst)
  return all_inst


def all_blocks_in_location():
  all_inst = []
  for loc_i in LOCATIONS:
    all_inst.append(f'put all the blocks in the {loc_i}')
  return all_inst


def k_blocks_in_location_i_rest_in_location_j():
  all_inst = []
  location_pairs = list(itertools.permutations(LOCATIONS, 2))
  for k in range(1, 8):
    for (loc_i, loc_j) in location_pairs:
      all_inst.append(
          f'put {k} blocks in the {loc_i}, then the rest in the {loc_j}')
  return all_inst


def get_shape_instructions():
  """Get shape instructions."""
  all_inst = []
  for shape in [
      'square', 'triangle', 'circle', 'diamond', 'parallelogram', 'G', 'O', 'L',
      'E', 'A', 'T', 'X', 'V', 'Y', 'U', 'S', 'C', 'Z', 'N', 'J'
  ]:
    all_inst.append(f'make a "{shape}"" shape out of all the blocks')
  all_inst.append('make a smiley face out of the blocks')
  all_inst.append('make a rainbow out of the blocks (red, yellow, green, '
                  'blue in a semicircle)')
  return all_inst


def get_sort_tasks():
  sort_tasks = ['group the blocks by color']
  return sort_tasks


def get_random_8block_instruction(rng):
  """Gets random from each 'instruction class'."""
  all_task_fns = [
      get_sort_tasks, colors_in_locations, group_color_pairs,
      get_colors_in_lines, group_color_pairs_in_locations, get_line_tasks,
      get_surround_tasks, blocks_in_order_outer_edge, all_blocks_in_location,
      k_blocks_in_location_i_rest_in_location_j, get_shape_instructions
  ]

  task_fn = rng.choice(all_task_fns)
  inst_choices = task_fn()
  inst = rng.choice(inst_choices)
  return inst


# pytype: skip-file
@gin.configurable
class PlayReward(base_reward.LanguageTableReward):
  """Dummy play reward."""

  def __init__(self, goal_reward, rng, delay_reward_steps, block_mode):
    super(PlayReward, self).__init__(goal_reward, rng, delay_reward_steps,
                                     block_mode)
    self.block_mode = block_mode.value
    if self.block_mode == 'BLOCK_4':
      # Get 20 random instructions from 5 long horizon families to make 100
      # training instructions. 353 test instructions.
      self._all_instructions = get_100_4block_instructions(
          num_train_per_family=20)

  def _sample_instruction(
      self, start_block, target_block, blocks_on_table):
    """Randomly sample a long horizon task."""
    if self.block_mode == 'BLOCK_4':
      return self._rng.choice(self._all_instructions)
    elif self.block_mode == 'BLOCK_8':
      return get_random_8block_instruction(self._rng)
    else:
      raise ValueError('Unsupported block mode: %s' % self.block_mode)

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

  def reward(self, state):
    """Calculates reward given state."""
    reward = 0.
    done = False
    return reward, done
