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

"""Utilities for getting all instructions from the different 2d board envs."""
from language_table.environments.rewards import block2absolutelocation
from language_table.environments.rewards import block2block
from language_table.environments.rewards import block2block_relative_location
from language_table.environments.rewards import block2relativelocation
from language_table.environments.rewards import point2block
from language_table.environments.rewards import separate_blocks


CLIP_VOCAB_SIZE = 49408


def generate_all_instructions(block_mode):
  """Gets all instructions across all environments."""
  return (block2block.generate_all_instructions(block_mode) +
          point2block.generate_all_instructions(block_mode) +
          block2relativelocation.generate_all_instructions(block_mode) +
          block2absolutelocation.generate_all_instructions(block_mode) +
          block2block_relative_location.generate_all_instructions(block_mode) +
          separate_blocks.generate_all_instructions(block_mode))


def vocab_size(block_mode):
  words = set()
  for instruction in generate_all_instructions(block_mode):
    for word in instruction.split(' '):
      words.add(word)
  return len(words)
