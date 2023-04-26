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

"""Defines the n choose k blocks on Language Table."""
import collections
import enum
import itertools

import numpy as np


class LanguageTableBlockVariants(enum.Enum):
  BLOCK_1 = 'BLOCK_1'  # 1 green star. Just for debugging.
  BLOCK_4 = 'BLOCK_4'  # The original 4 blocks.
  BLOCK_8 = 'BLOCK_8'  # 2 of each color, 2 of each shape, 8 total.
  BLOCK_4_WPOLE = 'BLOCK_4_WPOLE'  # original 4 blocks with purple pole as goal
  BLOCK_8_WPOLE = 'BLOCK_8_WPOLE'  # 8 blocks with purple pole as goal
  N_CHOOSE_K = 'N_CHOOSE_K'  # Combinatorial.


BLOCK_VARIANTS = [i.value for i in LanguageTableBlockVariants]


def get_all_block_subsets(mode, training):
  """Returns all subsets for the chosen mode."""
  if mode == LanguageTableBlockVariants.BLOCK_1:
    return [FIXED_1_COMBINATION]
  if mode == LanguageTableBlockVariants.BLOCK_4:
    return [FIXED_4_COMBINATION]
  elif mode == LanguageTableBlockVariants.BLOCK_8:
    return [FIXED_8_COMBINATION]
  elif mode == LanguageTableBlockVariants.N_CHOOSE_K:
    if training:
      return TRAIN_COMBINATIONS
    else:
      return TEST_COMBINATIONS
  elif mode == LanguageTableBlockVariants.BLOCK_4_WPOLE:
    return [FIXED_4_COMBINATION_WPOLE]
  elif mode == LanguageTableBlockVariants.BLOCK_8_WPOLE:
    return [FIXED_8_COMBINATION_WPOLE]
  else:
    raise ValueError('Unsupported block mode')


def get_block_set(mode):
  """Defines unique set of blocks by mode."""
  if mode == LanguageTableBlockVariants.BLOCK_1:
    return FIXED_1_COMBINATION
  if mode == LanguageTableBlockVariants.BLOCK_4:
    return FIXED_4_COMBINATION
  elif mode == LanguageTableBlockVariants.BLOCK_8:
    return FIXED_8_COMBINATION
  elif mode == LanguageTableBlockVariants.N_CHOOSE_K:
    return ALL_BLOCKS
  else:
    raise ValueError('Unsupported block mode')


def get_all_block_pairs(mode):
  """Defines all pairs of blocks. Useful for generating all instructions."""
  all_blocks = get_block_set(mode)
  all_pairs = itertools.permutations(all_blocks, 2)
  return all_pairs


def get_blocks_text_descriptions(mode):
  """Get text strings for all blocks on table by mode."""
  blocks = get_block_set(mode)
  blocks_text = [' '.join(i.split('_')) for i in blocks]
  return blocks_text


BLOCK_URDF_PATHS = collections.OrderedDict(
    # Red blocks.
    red_moon='third_party/py/language_table/environments/assets/blocks/red_moon.urdf',
    red_cube='third_party/py/language_table/environments/assets/blocks/red_cube.urdf',
    red_star='third_party/py/language_table/environments/assets/blocks/red_star.urdf',
    red_pentagon='third_party/py/language_table/environments/assets/blocks/red_pentagon.urdf',
    # Blue blocks.
    blue_moon='third_party/py/language_table/environments/assets/blocks/blue_moon.urdf',
    blue_cube='third_party/py/language_table/environments/assets/blocks/blue_cube.urdf',
    blue_star='third_party/py/language_table/environments/assets/blocks/blue_star.urdf',
    blue_pentagon='third_party/py/language_table/environments/assets/blocks/blue_pentagon.urdf',
    # Yellow blocks.
    yellow_moon='third_party/py/language_table/environments/assets/blocks/yellow_moon.urdf',
    yellow_cube='third_party/py/language_table/environments/assets/blocks/yellow_cube.urdf',
    yellow_star='third_party/py/language_table/environments/assets/blocks/yellow_star.urdf',
    yellow_pentagon='third_party/py/language_table/environments/assets/blocks/yellow_pentagon.urdf',
    # Green blocks.
    green_moon='third_party/py/language_table/environments/assets/blocks/green_moon.urdf',
    green_cube='third_party/py/language_table/environments/assets/blocks/green_cube.urdf',
    green_star='third_party/py/language_table/environments/assets/blocks/green_star.urdf',
    green_pentagon='third_party/py/language_table/environments/assets/blocks/green_pentagon.urdf',
)

POLE_URDF_PATHS = collections.OrderedDict(
    # Purple Pole.
    purple_pole='third_party/py/language_table/environments/assets/blocks/purple_pole.urdf',
)

# Use this just to define the observation space.
DUMMY_START_BLOCK = list(BLOCK_URDF_PATHS.keys())[0]
COLORS = ['red', 'blue', 'green', 'yellow']
SHAPES = ['moon', 'cube', 'star', 'pentagon']
ALL_BLOCKS = ['_'.join(i) for i in itertools.product(COLORS, SHAPES)]
MIN_K = 4
MAX_K = 10
ALL_COMBINATIONS = []
for k in range(MIN_K, MAX_K+1):
  k_combos = list(itertools.combinations(ALL_BLOCKS, k))
  ALL_COMBINATIONS.extend(k_combos)
# Seeded shuffle.
combo_rng = np.random.RandomState(seed=0)
combo_rng.shuffle(ALL_COMBINATIONS)
# Divide combinations by train / test.
TRAIN_COMBINATIONS = ALL_COMBINATIONS[:int(len(ALL_COMBINATIONS)*0.9)]
TEST_COMBINATIONS = ALL_COMBINATIONS[int(len(ALL_COMBINATIONS)*0.9):]

# 8 total, 2 of each color, 2 of each shape.
FIXED_8_COMBINATION = (
    'red_moon',
    'red_pentagon',
    'blue_moon',
    'blue_cube',
    'green_cube',
    'green_star',
    'yellow_star',
    'yellow_pentagon')

# The original "4-block" environment.
FIXED_4_COMBINATION = (
    'red_moon',
    'blue_cube',
    'green_star',
    'yellow_pentagon'
    )


# 8 total blocks + 1 goal purple pole, 2 of each color, 2 of each shape.
FIXED_8_COMBINATION_WPOLE = ('red_moon', 'red_pentagon', 'blue_moon',
                             'blue_cube', 'green_cube', 'green_star',
                             'yellow_star', 'yellow_pentagon', 'purple_pole')

# The original "4-block" environment + 1 goal purple pole.
FIXED_4_COMBINATION_WPOLE = ('red_moon', 'blue_cube', 'green_star',
                             'yellow_pentagon', 'purple_pole')
# 1-block debugging environment.
FIXED_1_COMBINATION = ['green_star']
