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

"""LAVA Architecture."""

from typing import Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from language_table.train.networks import dense_resnet
from language_table.train.networks import resnet_v1

import numpy as np

from scenic.projects.baselines.clip import layers as clip_layers


class ConvMaxpoolCNNEncoder(nn.Module):
  """Encodes RGB images to a pyramid of visual features from ConvNet."""

  @nn.compact
  def __call__(self, rgb, *, train):
    x = rgb

    pyramid = []

    conv_channels = [32, 64, 128, 256]
    for conv_size in conv_channels:
      x = nn.Conv(features=conv_size, kernel_size=(3, 3), padding="SAME")(x)
      x = nn.relu(x)
      x = nn.max_pool(x, (2, 2), strides=(2, 2), padding="VALID")
      pyramid.append(x)

    x = nn.max_pool(x, (2, 2), strides=(2, 2), padding="VALID")
    pyramid.append(x)

    return pyramid


class ResNetVisualEncoder(nn.Module):
  """Encodes RGB images to a pyramid of visual features."""

  @nn.compact
  def __call__(self, rgb, *, train):
    rgb = normalize_image_resnet(rgb)
    resnet_features = resnet_v1.MultiscaleResNet(
        stage_sizes=(3, 4), block_cls=resnet_v1.BottleneckResNetBlock)(
            rgb, train=False)  # train=False since these are frozen.

    # Construct the full feature pyramid.
    pyramid = [resnet_features[0], resnet_features[1]]

    # Start from the second feature map.
    x = resnet_features[1]

    conv_channels = [128, 256]
    for conv_size in conv_channels:
      x = nn.Conv(features=conv_size, kernel_size=(3, 3), padding="SAME")(x)
      x = nn.relu(x)
      x = nn.max_pool(x, (2, 2), strides=(2, 2), padding="VALID")
      pyramid.append(x)

    # Final maxpool.
    x = nn.max_pool(x, (2, 2), strides=(2, 2), padding="VALID")
    pyramid.append(x)

    # The shape here should be:
    # 0 (1, 112, 112, 64)
    # 1 (1, 56, 56, 64)
    # 2 (1, 28, 28, 128)
    # 3 (1, 14, 14, 256)
    # 4 (1, 7, 7, 256)
    return pyramid


def normalize_image_resnet(images):
  bs = images.shape[0]
  mean_rgb = (0.485, 0.456, 0.406)
  stddev_rgb = (0.229, 0.224, 0.225)
  normed_images = images - jnp.array(mean_rgb).reshape((1, 1, 1, 3))
  normed_images = normed_images / jnp.array(stddev_rgb).reshape((1, 1, 1, 3))
  normed_images = jax.image.resize(
      normed_images, (bs, 224, 224, 3), method="bilinear", antialias=False)
  return normed_images


def sinusoidal_init(max_len, max_timescale = 1.0e4):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
      max_len: maximum possible length for the input.
      max_timescale: Maximum time scale.

  Returns:
      output: init function returning `(1, max_len, d_feature)`
  """

  def init(key,
           shape,
           dtype = jnp.float32):
    """Sinusoidal init.

    The defined API by JAX for a custom initializer is:
      `def init(key, shape, dtype)`

    Even though some of args might be not used, the signature should follow
    this API as JAX passes all the three arguments (key, shape, dtype)
    to the initializers.

    Args:
      key: JAXPRNG key.
      shape: Shape used for making the initialized values.
      dtype: JAX data type.

    Returns:
      Initialized values
    """
    del key, dtype
    d_feature = shape[-1]
    pos_emb = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_feature, 2) * -(np.log(max_timescale) / d_feature))
    pos_emb[:, 0::2] = np.sin(position * div_term)
    pos_emb[:, 1::2] = np.cos(position * div_term)
    pe = pos_emb[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


class Add1DPositionEmbedding(nn.Module):
  """Adds 1-dimensional positional embeddings to the inputs.

  Attributes:
    rescale_from: tuple; If not None, embeddings are rescaled from this shape.
    max_len: int; Maximum possible length for the input. If None, the max_len is
      set to the inputs sequence length.
    posemb_init: Positional embedding initializer.
    param_name: The name of the parameter that stores the positional embedding.
  """

  rescale_from: Optional[Sequence[int]] = None
  max_len: Optional[int] = None
  param_name: str = "pos_embedding"

  @nn.compact
  def __call__(self, inputs):
    """Applies Add1DPositionEmbedding module.

    Args:
      inputs: nd-arrary; Input data.

    Returns:
      Output: `(bs, timesteps, in_dim)`.
    """
    assert inputs.ndim == 3, ("Number of dimensions should be 3,"
                              " but it is: %d" % inputs.ndim)
    length = inputs.shape[1]
    max_len = self.max_len or length
    embedding_length = max_len

    if self.rescale_from:  # Shape: `[len, c]`.
      embedding_length = self.rescale_from[0]

    pos_emb_shape = (1, embedding_length, inputs.shape[-1])
    # Use a fixed (non-learned) sinusoidal position embedding.
    pos_embedding = sinusoidal_init(max_len=embedding_length)(None,  # pytype: disable=wrong-arg-types  # jax-ndarray
                                                              pos_emb_shape,
                                                              None)
    pe = pos_embedding[:, :length, :]
    return inputs + pe


def positional_encoding2d(d_model, height, width, flatten=True):
  """Creates a 2d fixed sin/cos embedding."""
  if d_model % 4 != 0:
    raise ValueError("Cannot use sin/cos positional encoding with "
                     "odd dimension (got dim={:d})".format(d_model))
  pe = np.zeros([d_model, height, width])
  # Each dimension use half of d_model
  d_model = int(d_model / 2)
  div_term = np.exp(np.arange(0., d_model, 2) * -(np.log(10000.0) / d_model))
  pos_w = np.expand_dims(np.arange(0., width), 1)
  pos_h = np.expand_dims(np.arange(0., height), 1)

  pe[0:d_model:2, :, :] = np.tile(
      np.expand_dims(np.transpose(np.sin(pos_w * div_term)), 1), [1, height, 1])

  pe[1:d_model:2, :, :] = np.tile(
      np.expand_dims(np.transpose(np.cos(pos_w * div_term)), 1), [1, height, 1])

  pe[d_model::2, :, :] = np.tile(
      np.expand_dims(np.transpose(np.sin(pos_h * div_term)), 2), [1, 1, width])

  pe[d_model + 1::2, :, :] = np.tile(
      np.expand_dims(np.transpose(np.cos(pos_h * div_term)), 2), [1, 1, width])

  if flatten:
    pe = np.reshape(pe, [height * width, d_model * 2])
  else:
    pe = np.reshape(pe, [height, width, d_model * 2])
  # Add batch axis.
  return pe[None, Ellipsis]


class VisualDescriptorsNet(nn.Module):
  """Produces visual sentence."""

  pyramid_fuse_layers: Sequence[int]
  d_model: int

  @nn.compact
  def __call__(self, x, *, train):

    pixel_features = x
    layer_pixel_x = []

    normal_initializer = jax.nn.initializers.normal(stddev=0.05)
    for pyr_idx in self.pyramid_fuse_layers:
      pixel_x = pixel_features[pyr_idx]

      h = jnp.shape(pixel_x)[1]
      w = jnp.shape(pixel_x)[2]

      pixel_x = nn.Dense(
          self.d_model,
          kernel_init=normal_initializer,
          bias_init=normal_initializer)(
              pixel_x)

      pixel_x = self._flatten(pixel_x)
      # Scale the embeddings.
      pixel_x *= jnp.sqrt(float(self.d_model))

      pos_2d = positional_encoding2d(self.d_model, h, w)

      pixel_x = pixel_x + pos_2d

      layer_pixel_x.append(pixel_x)

    pixel_sentence = jnp.concatenate(layer_pixel_x, axis=1)
    return pixel_sentence

  @staticmethod
  def _flatten(x):
    nb = x.shape[0]
    nw = x.shape[1]
    nh = x.shape[2]
    nc = x.shape[3]
    return jnp.reshape(x, (nb, nw * nh, nc))


class PrenormPixelLangEncoder(nn.Module):
  """Prenorm pixel lang MHA."""

  num_heads: int
  dropout_rate: float
  mha_dropout_rate: float
  dff: int

  @nn.compact
  def __call__(self, pixel_x, lang_x, *, train):
    residual_lang = lang_x

    pixel_x = nn.LayerNorm()(pixel_x)
    lang_x = nn.LayerNorm()(lang_x)

    x2 = nn.MultiHeadDotProductAttention(
        self.num_heads, dropout_rate=self.mha_dropout_rate)(
            lang_x, pixel_x, deterministic=not train)
    x2 = nn.Dropout(self.dropout_rate)(x2, deterministic=not train)

    # Residual, only on the language path.
    x3 = residual_lang + x2

    # layer norm just the ffn input.
    x4 = nn.LayerNorm()(x3)

    # ffn.
    norm_init = jax.nn.initializers.normal(stddev=0.05)
    x5 = nn.Dense(self.dff, kernel_init=norm_init, bias_init=norm_init)(x4)
    x5 = nn.relu(x5)
    x5 = nn.Dense(self.dff, kernel_init=norm_init, bias_init=norm_init)(x5)
    x5 = nn.Dropout(self.dropout_rate)(x5, deterministic=not train)

    x = x5 + x3

    return x


class PrenormEncoderLayer(nn.Module):
  """Prenorm MHA layer."""

  num_heads: int
  dropout_rate: float
  mha_dropout_rate: float
  dff: int

  @nn.compact
  def __call__(self, x, *, train):
    x1 = nn.LayerNorm()(x)
    x2 = nn.MultiHeadDotProductAttention(
        self.num_heads, dropout_rate=self.mha_dropout_rate)(
            x1, x1, deterministic=not train)
    x2 = nn.Dropout(self.dropout_rate)(x2, deterministic=not train)

    x3 = x + x2

    x4 = nn.LayerNorm()(x3)

    # ffn.
    norm_init = jax.nn.initializers.normal(stddev=0.05)
    x5 = nn.Dense(self.dff, kernel_init=norm_init, bias_init=norm_init)(x4)
    x5 = nn.relu(x5)
    x5 = nn.Dense(self.dff, kernel_init=norm_init, bias_init=norm_init)(x5)

    x5 = nn.Dropout(self.dropout_rate)(x5, deterministic=not train)
    return x3 + x5


class TemporalTransformer(nn.Module):
  """Transformer over time."""

  num_layers: int
  d_model: int
  num_heads: int
  dff: int
  sequence_length: int

  @nn.compact
  def __call__(self, x, *, train):
    normal_initializer = jax.nn.initializers.normal(stddev=0.05)
    x = nn.Dense(
        self.d_model,
        kernel_init=normal_initializer,
        bias_init=normal_initializer)(
            x)

    x *= jnp.sqrt(self.d_model)

    x = Add1DPositionEmbedding(max_len=self.sequence_length)(x)

    x = nn.Dropout(0.1)(x, deterministic=not train)

    for _ in range(self.num_layers):
      x = PrenormEncoderLayer(
          num_heads=self.num_heads,
          dropout_rate=0.1,
          mha_dropout_rate=0.0,
          dff=self.dff)(
              x, train=train)

    x = jnp.mean(x, axis=1)

    x = nn.LayerNorm()(x)
    return x


class SequenceLAVAEncoder(nn.Module):
  """Full Sequence EBM."""

  image_encoder: str
  lang_encoder: str

  num_layers: int = 2

  sequence_length: int = 4
  temporal_transformer_num_layers: int = 2

  d_model: int = 128
  num_heads: int = 2
  pyramid_fuse_layers: Tuple[int, Ellipsis] = (2, 3, 4)

  @nn.compact
  def __call__(self, x, *, train):
    """Apply the ResNet to the inputs `x`.

    Args:
      x: Inputs.
      train: Whether to use BatchNorm in training or inference mode.

    Returns:
      The output head with `num_classes` entries.
    """
    rgb = x["rgb"]
    bs = rgb.shape[0]
    seqlen = rgb.shape[1]
    h = rgb.shape[2]
    w = rgb.shape[3]
    c = rgb.shape[4]
    rgb = jnp.reshape(rgb, (bs * seqlen, h, w, c))

    if self.image_encoder == "resnet":
      features = ResNetVisualEncoder()(rgb, train=train)
    elif self.image_encoder == "conv_maxpool":
      features = ConvMaxpoolCNNEncoder()(rgb, train=train)
    else:
      raise NotImplementedError

    visual_sentence = VisualDescriptorsNet(
        d_model=self.d_model, pyramid_fuse_layers=self.pyramid_fuse_layers)(
            features, train=train)
    visual_sentence = nn.Dropout(0.1)(visual_sentence, deterministic=not train)

    if self.lang_encoder == "clip_in_obs":
      lang_embedding = x["clip_embedding"]
      lang_embedding = jnp.reshape(lang_embedding, [bs * seqlen, -1])
    elif self.lang_encoder == "clip":
      tokens = x["instruction_tokenized_clip"].astype(jnp.int32)[:, 0]
      lang_embedding = clip_layers.TextEncoder(
          vocab_size=49408,
          features=512,
          num_heads=8,
          num_layers=12,
          out_features=512,
      )(
          tokens)
      lang_embedding = jnp.tile(lang_embedding[:, None, :], [1, seqlen, 1])
      lang_embedding = jnp.reshape(lang_embedding, [bs * seqlen, -1])
      lang_embedding /= jnp.linalg.norm(lang_embedding, axis=-1, keepdims=True)
    else:
      raise NotImplementedError

    normal_initializer = jax.nn.initializers.normal(stddev=0.05)

    lang_embedding = nn.Dense(
        self.d_model,
        kernel_init=normal_initializer,
        bias_init=normal_initializer)(
            lang_embedding)

    lang_embedding *= jnp.sqrt(self.d_model)  # scale emb.
    lang_embedding = nn.Dropout(0.1)(lang_embedding, deterministic=not train)

    lang_embedding = lang_embedding[:, None, :]
    lang_query = lang_embedding

    fused_x = lang_query
    for _ in range(self.num_layers):
      fused_x = PrenormPixelLangEncoder(
          num_heads=2, dropout_rate=0.1, mha_dropout_rate=0.0,
          dff=self.d_model)(
              visual_sentence, fused_x, train=train)

    fused_x = jnp.squeeze(fused_x, axis=1)

    fused_x = nn.LayerNorm()(fused_x)

    seq_obs_encoding = jnp.reshape(fused_x, [bs, seqlen, -1])

    obs_encoding = TemporalTransformer(
        num_layers=self.temporal_transformer_num_layers,
        d_model=self.d_model,
        num_heads=self.num_heads,
        dff=self.d_model,
        sequence_length=self.sequence_length)(
            seq_obs_encoding, train=train)
    return obs_encoding


class SequenceLAVMSE(nn.Module):
  """Full Sequence LAV MSE model."""

  action_size: int

  dense_resnet_width: int
  dense_resnet_num_blocks: int

  lava_num_layers: int
  lava_sequence_length: int
  lava_temporal_transformer_num_layers: int
  lava_d_model: int
  lava_num_heads: int
  lava_pyramid_fuse_layers: Tuple[int, Ellipsis]
  lava_image_encoder: str
  lava_lang_encoder: str

  def setup(self):
    self.encoder = SequenceLAVAEncoder(
        num_layers=self.lava_num_layers,
        sequence_length=self.lava_sequence_length,
        temporal_transformer_num_layers=self
        .lava_temporal_transformer_num_layers,
        d_model=self.lava_d_model,
        num_heads=self.lava_num_heads,
        pyramid_fuse_layers=self.lava_pyramid_fuse_layers,
        image_encoder=self.lava_image_encoder,
        lang_encoder=self.lava_lang_encoder)
    self.dense_resnet = dense_resnet.DenseResnet(
        width=self.dense_resnet_width,
        num_blocks=self.dense_resnet_num_blocks,
        value_net=False)
    normal_initializer = jax.nn.initializers.normal(stddev=0.05)
    self.action_projection = nn.Dense(
        self.action_size,
        kernel_init=normal_initializer,
        bias_init=normal_initializer)

  def __call__(self, obs, *, train):
    x = self.encoder(obs, train=train)
    x = self.dense_resnet(x, train=train)
    x = self.action_projection(x)
    return x
