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

"""Simple Pixel + Language network impelementations."""

import flax.linen as nn
import jax
import jax.numpy as jnp

from language_table.train.networks import dense_resnet


class LanguageFusion(nn.Module):
  """Fuses language information multiplicatively."""

  @nn.compact
  def __call__(self, lang, image):
    norm_init = jax.nn.initializers.normal(stddev=0.05)
    lang = nn.Dense(
        jnp.shape(image)[-1], kernel_init=norm_init, bias_init=norm_init)(
            lang)

    img_shape = jnp.shape(image)
    h = img_shape[1]
    w = img_shape[2]
    lang = jnp.tile(lang[:, None, None, :], [1, h, w, 1])

    # Fuse.
    fused = image * lang
    return fused


class ConvMaxpoolLanguageEncoder(nn.Module):
  """Simple Conv + Maxpool encoder that multiplicatively fuses language."""

  @nn.compact
  def __call__(self, rgb, lang_embedding, *, train):

    x = rgb

    fuse_from = 2
    conv_channels = [32, 64, 128, 256]
    for idx, ch in enumerate(conv_channels):
      x = nn.Conv(features=ch, kernel_size=(3, 3), padding="SAME")(x)

      if fuse_from <= idx + 1:
        x = LanguageFusion()(lang_embedding, x)

      x = nn.relu(x)
      x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")

    x = jnp.mean(x, axis=(1, 2), keepdims=False)

    norm_init = jax.nn.initializers.normal(stddev=0.05)
    if fuse_from <= len(conv_channels) + 1:
      lang_info = nn.Dense(
          conv_channels[-1], kernel_init=norm_init, bias_init=norm_init)(
              lang_embedding)
      x *= lang_info

    x = nn.relu(x)

    x = nn.LayerNorm()(x)

    return x


class PixelLangMSE(nn.Module):
  """Simple Pixel Language network."""

  action_size: int

  dense_resnet_width: int
  dense_resnet_num_blocks: int

  def setup(self):
    self.encoder = ConvMaxpoolLanguageEncoder()
    self.dense_resnet = dense_resnet.DenseResnet(
        width=self.dense_resnet_width,
        num_blocks=self.dense_resnet_num_blocks,
        value_net=False)
    norm_init = jax.nn.initializers.normal(stddev=0.05)
    self.action_projection = nn.Dense(
        self.action_size, kernel_init=norm_init, bias_init=norm_init)

  def __call__(self, obs, *, train):
    rgb = obs["rgb"]
    # Reshape to stack images channelwise.
    sh = jnp.shape(rgb)
    b, n, w, h, c = sh
    rgb = jnp.reshape(rgb, (b, w, h, c * n))

    lang = obs["clip_embedding"]
    lang = lang[:, -1, Ellipsis]
    encoded_obs = self.encoder(rgb, lang, train=train)

    x = self.dense_resnet(encoded_obs, train=train)
    x = self.action_projection(x)
    return x
