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

"""Environment wrappers."""

from typing import Any, Optional

from language_table.common import clip_tokenizer
import numpy as np
import tensorflow as tf
from tf_agents.environments import wrappers


class ClipTokenWrapper(wrappers.PyEnvironmentBaseWrapper):
  """Environment wrapper that adds CLIP tokens to the obs."""

  def __init__(self, env, context_length = 77):
    """Centrally crops an image from a dict observation."""
    super(ClipTokenWrapper, self).__init__(env)
    self._context_length = context_length
    vocab_lookup = clip_tokenizer.create_vocab()
    self._tokenizer = clip_tokenizer.ClipTokenizer(vocab_lookup)
    self._current_tokens = None

  def _reset(self):
    time_step = self._env.reset()
    self._current_tokens = self._tokenize(time_step.observation['instruction'])
    new_obs = time_step.observation
    new_obs['instruction_tokenized_clip'] = self._current_tokens
    return time_step._replace(observation=new_obs)

  def _step(self, action):
    time_step = self._env.step(action)
    new_obs = time_step.observation
    new_obs['instruction_tokenized_clip'] = self._current_tokens
    return time_step._replace(observation=new_obs)

  def _tokenize(self, instruction):
    bytes_list = instruction
    non_zero = bytes_list[np.where(bytes_list != 0)]
    if non_zero.shape[0] == 0:
      decoded = ''
    else:
      bytes_list = bytes(non_zero.tolist())
      decoded = bytes_list.decode('utf-8')
    tokens = clip_tokenizer.tokenize_text(decoded, self._tokenizer)[0]
    return tokens


class CentralCropImageWrapper(wrappers.PyEnvironmentBaseWrapper):
  """Environment wrapper that crops image observations."""

  def __init__(self,
               env,
               target_height,
               target_width,
               random_crop_factor = None):
    """Centrally crops an image from a dict observation."""
    super(CentralCropImageWrapper, self).__init__(env)
    self._target_height = target_height
    self._target_width = target_width
    self._random_crop_factor = random_crop_factor

  def _reset(self):
    time_step = self._env.reset()
    new_obs = self._crop_observation(time_step.observation)
    return time_step._replace(observation=new_obs)

  def _step(self, action):
    time_step = self._env.step(action)
    new_obs = self._crop_observation(time_step.observation)
    return time_step._replace(observation=new_obs)

  def _crop_observation(self, obs):
    new_obs = obs
    image = obs['rgb']
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Apply average crop augmentation.
    image = crop_test_image(image, self._random_crop_factor)
    image = resize_images(image, self._target_height, self._target_width)
    new_obs['rgb'] = image.numpy()
    return new_obs


def crop_test_image(images, random_crop_factor):
  """Get the average crop applied during crop training augmentation."""

  def take_center_crop_consistent_with_random(im):
    im_raw_size = tf.shape(im)
    raw_height = tf.cast(im_raw_size[0], tf.float32)
    raw_width = tf.cast(im_raw_size[1], tf.float32)
    scaled_height = raw_height * random_crop_factor
    scaled_width = raw_width * random_crop_factor
    offset_height = tf.cast((raw_height - scaled_height) // 2, tf.int32)
    offset_width = tf.cast((raw_width - scaled_width) // 2, tf.int32)
    target_height = tf.cast(scaled_height, tf.int32)
    target_width = tf.cast(scaled_width, tf.int32)
    im = tf.image.crop_to_bounding_box(
        im,
        offset_height=offset_height,
        offset_width=offset_width,
        target_height=target_height,
        target_width=target_width)
    return im

  if len(images.shape) == 3:
    return take_center_crop_consistent_with_random(images)
  images = tf.map_fn(take_center_crop_consistent_with_random, images)
  return images


def resize_images(images, target_height=None, target_width=None):
  """Resizes images to target_height, target_width."""
  assert target_height
  assert target_width

  # Resize to target height and width.
  def _resize(im):
    return tf.image.resize(im, [target_height, target_width])

  images = _resize(images)

  return images
