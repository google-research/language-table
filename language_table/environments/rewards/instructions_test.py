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

"""Tests for instructions."""

from language_table.environments import blocks
from language_table.environments.rewards import instructions
import tensorflow as tf


class InstructionsTest(tf.test.TestCase):

  def test_expected_instructions_generated(self):
    # This ensures that the same fixed number of instructions are generated
    # for each block mode.
    inst_block4 = instructions.generate_all_instructions(
        blocks.LanguageTableBlockVariants.BLOCK_4)
    self.assertLen(inst_block4, 12652)
    inst_block8 = instructions.generate_all_instructions(
        blocks.LanguageTableBlockVariants.BLOCK_8)
    self.assertLen(inst_block8, 30264)
    inst_n_choose_k = instructions.generate_all_instructions(
        blocks.LanguageTableBlockVariants.N_CHOOSE_K)
    self.assertLen(inst_n_choose_k, 80368)

if __name__ == '__main__':
  tf.test.main()
