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

from absl.testing import parameterized
from clu import parameter_overview
import jax
import jax.numpy as jnp
from language_table.train.networks import resnet_v1
import tensorflow as tf


class ResNetV1Test(tf.test.TestCase, parameterized.TestCase):
  """Test cases for ResNet V1."""

  @parameterized.named_parameters(
      ("ResNet18", resnet_v1.ResNet18, 11_689_512),
      ("ResNet34", resnet_v1.ResNet34, 21_797_672),
      ("ResNet50", resnet_v1.ResNet50, 25_557_032),
      ("ResNet101", resnet_v1.ResNet101, 44_549_160),
      ("ResNet152", resnet_v1.ResNet152, 60_192_808),
      ("ResNet200", resnet_v1.ResNet200, 64_673_832),
  )
  def test_architecture(self, cls, param_count):
    rng = jax.random.PRNGKey(0)
    model = cls(num_classes=1000)
    variables = model.init(rng, jnp.ones([2, 224, 224, 3]), train=False)
    params = variables["params"]
    self.assertEqual(param_count, parameter_overview.count_parameters(params))


if __name__ == "__main__":
  tf.test.main()
