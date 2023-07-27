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

"""Saves a scenic CLIP checkpoint."""

from collections.abc import Sequence

from absl import app
from absl import flags

from clu import checkpoint
from scenic.projects.baselines.clip import model
import tensorflow as tf

_CHECKPOINT_DIRECTORY = flags.DEFINE_string(
    'checkpoint_directory', None, 'The directory to save the checkpoint.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  model_vars = model.load_model_vars(model_name='vit_b16')

  out_directory = _CHECKPOINT_DIRECTORY.value
  if not tf.io.gfile.exists(out_directory):
    tf.io.gfile.makedirs(out_directory)
  ckpt = checkpoint.Checkpoint(base_directory=out_directory)

  ckpt.save(model_vars)


if __name__ == '__main__':
  app.run(main)
