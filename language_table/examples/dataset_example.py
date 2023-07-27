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

"""Example for loading the Language-Table dataset.

Language-Table data is in the [RLDS](https://github.com/google-research/rlds)
format.
See the [RLDS Tutorial](https://colab.research.google.com/github/
google-research/rlds/blob/main/rlds/examples/rlds_tutorial.ipynb)
for more details on how to use RLDS datasets.
"""

from collections.abc import Sequence

from absl import app

import tensorflow_datasets as tfds

dataset_paths = {
    'language_table': 'gs://gresearch/robotics/language_table/0.0.1/',
    'language_table_sim': 'gs://gresearch/robotics/language_table_sim/0.0.1/',
}


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Iterate through 5 items in language_table.
  builder = tfds.builder_from_directory(dataset_paths['language_table'])
  ds = builder.as_dataset(split='train')
  ds = ds.flat_map(lambda x: x['steps'])  # get the dataset as individual steps
  for item in iter(ds.take(5)):
    print(item)

  # Iterate through 5 items in language_table_sim.
  builder = tfds.builder_from_directory(dataset_paths['language_table_sim'])
  ds = builder.as_dataset(split='train')
  ds = ds.flat_map(lambda x: x['steps'])  # get the dataset as individual steps
  for item in iter(ds.take(5)):
    print(item)


if __name__ == '__main__':
  app.run(main)
