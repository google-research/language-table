# coding=utf-8
# Copyright 2024 The Language Tale Authors.
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

"""Defines synonyms for the structured language 2d board environment."""
import collections


def get_block_synonyms(block, blocks_on_table):
  """Get synonyms for blocks based on what is on table."""
  color, shape = get_color_shape(block)
  color_counts, shape_counts = count_color_shape(blocks_on_table)
  synonyms = []
  if color_counts[color] == 1:
    # There is only one 'red' block, so feel free to refer to it as
    # 'red block'.
    synonyms.append('%s block' % color)
  if shape_counts[shape] == 1:
    # There is only one 'star' block, so feel free to refer to it as
    # 'star'.
    synonyms.append(shape)
  # (color, shape) is always unique.
  synonyms.append('%s %s' % (color, shape))
  return synonyms


def count_color_shape(blocks_on_table):
  colors, shapes = zip(*[get_color_shape(i) for i in blocks_on_table])
  color_counts = collections.Counter(colors)
  shape_counts = collections.Counter(shapes)
  return color_counts, shape_counts


def get_color_shape(block):
  color, shape = block.split('_')
  return color, shape


PUSH_VERBS = [
    'push the',
    'move the',
    'slide the',
    'put the',
]

PREPOSITIONS = [
    'to the',
    'towards the',
    'close to the',
    'next to the',
]

POINT_PREPOSITIONS = [
    'point next to the',
    'point close to the',
    'point to the',
    'point at the',
    'move the arm next to the',
    'move the arm close to the',
    'move the arm to the',
    'move your arm next to the',
    'move your arm close to the',
    'move your arm to the',
    'move next to the',
    'move close to the',
    'move to the',
]
