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

"""Residual dense block."""

from flax import linen as nn
import jax


class ResnetDenseBlock(nn.Module):
  """Single dense resnet block."""
  width: int

  @nn.compact
  def __call__(self, x, *, train):
    normal_initializer = jax.nn.initializers.normal(stddev=0.05)
    y = nn.relu(x)
    y = nn.Dense(
        self.width // 4,
        kernel_init=normal_initializer,
        bias_init=normal_initializer)(
            y)
    y = nn.relu(y)
    y = nn.Dense(
        self.width // 4,
        kernel_init=normal_initializer,
        bias_init=normal_initializer)(
            y)
    y = nn.relu(y)
    y = nn.Dense(
        self.width,
        kernel_init=normal_initializer,
        bias_init=normal_initializer)(
            y)

    return x + y


class DenseResnet(nn.Module):
  """Dense Resnet module."""

  width: int
  num_blocks: int
  value_net: bool

  @nn.compact
  def __call__(self, x, *, train):
    normal_initializer = jax.nn.initializers.normal(stddev=0.05)
    x = nn.Dense(
        self.width,
        kernel_init=normal_initializer,
        bias_init=normal_initializer)(
            x)
    for _ in range(self.num_blocks):
      x = ResnetDenseBlock(self.width)(x, train=train)

    if self.value_net:
      x = nn.Dense(
          1, kernel_init=normal_initializer, bias_init=normal_initializer)(
              x)
    return x
