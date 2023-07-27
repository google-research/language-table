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

"""Example for running the Language-Table environment."""

from collections.abc import Sequence

from absl import app

from language_table.environments import blocks
from language_table.environments import language_table
from language_table.environments.rewards import block2block

from matplotlib import pyplot as plt


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  env = language_table.LanguageTable(
      block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
      reward_factory=block2block.BlockToBlockReward,
      control_frequency=10.0,
  )
  _ = env.reset()

  # Take a few random actions.
  for _ in range(5):
    env.step(env.action_space.sample())

  # Save a rendered image.
  plt.imsave('/tmp/language_table_render.png', env.render())


if __name__ == '__main__':
  app.run(main)
