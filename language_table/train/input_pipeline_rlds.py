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

"""RLDS input pipeline."""

import dataclasses
import functools
import json
import os
import time
from typing import Optional, Tuple
from absl import logging

from clu import preprocess_spec

import jax
from language_table.common import clip_tokenizer
from language_table.train import normalization
import rlds
import tensorflow as tf
import tensorflow_datasets as tfds
import tree


Features = preprocess_spec.Features
FlatFeatures = preprocess_spec.FlatFeatures


def _is_not_terminal(step):
  if step[rlds.IS_TERMINAL]:
    return False
  return True


def create_datasets(
    rng,
    dataset_path,
    normalization_path,
    sequence_length,
    per_device_batch_size,
    num_steps_per_train_iter,
    target_width=320,
    target_height=180,
    random_crop_factor=None,
    normalization_num_samples=3200,
    skip_normalize_keys=None,
    cache=False,
    shuffle=True,
    shuffle_buffer_size=50_000,
    prefetch_size=tf.data.AUTOTUNE,
    tf_data_service_address=None,
):
  """Create an RLDS dataset."""
  # Determinism. If we're using tf data service, we want to shuffle randomly
  # to avoid any problems where each data worker uses the same seed.
  if not tf_data_service_address:
    deterministic_shuffle = True
  else:
    deterministic_shuffle = False
  logging.info("deterministic_shuffle=%s", deterministic_shuffle)

  logging.info("Using dataset_path=%s", dataset_path)

  builder = tfds.builder_from_directory(builder_dir=dataset_path)

  dataset_options = tf.data.Options()
  dataset_options.experimental_optimization.map_parallelization = True
  dataset_options.threading.private_threadpool_size = 48
  dataset_options.threading.max_intra_op_parallelism = 1

  # The read_config contains a shuffle seed so that each worker will shuffle
  # the input files differently.
  shuffle_rng, rng = jax.random.split(rng)
  shuffle_rng = shuffle_rng[0] if deterministic_shuffle else None
  read_config = tfds.ReadConfig(
      shuffle_seed=shuffle_rng,
      shuffle_reshuffle_each_iteration=True,
      options=dataset_options,
  )

  ds = builder.as_dataset(
      split="train",
      decoders={"steps": {"observation": {"rgb": tfds.decode.SkipDecoding()}}},
      shuffle_files=True,
      read_config=read_config,
  )

  # Pad the beginning of every episode. We do this before getting subsequences
  # using a Pattern Dataset, since the pattern will not be applied until there
  # are at least `sequence_length` items in the queue for an episode. We
  # replicate the first step of each episode an additional sequence_length - 1
  # times so that these steps are included in the dataset for training.
  def _pad_episode(episode, padding):
    first_item_tensor = episode["steps"].take(1).get_single_element()
    first_item_ds = tf.data.Dataset.from_tensors(first_item_tensor)

    first_item_mid_tensor = tf.nest.map_structure(
        tf.identity, first_item_tensor
    )
    first_item_mid_tensor[rlds.IS_FIRST] = False
    padding_ds = tf.data.Dataset.from_tensors(first_item_mid_tensor).repeat(
        padding
    )

    full_padding = rlds.transformations.concatenate(first_item_ds, padding_ds)
    episode["steps"] = rlds.transformations.concatenate(
        full_padding, episode["steps"].skip(1)
    )
    return episode

  ds = ds.map(
      functools.partial(_pad_episode, padding=sequence_length - 1),
      tf.data.AUTOTUNE,
  )

  if cache:
    ds = ds.cache()

  # Extract subsequences.
  # Apply a reverb pattern to extract subsequences of length `sequence_length`
  # for each episode. This will add the time dim to axis 0.
  def get_seqlen_pattern(step):
    return {
        rlds.OBSERVATION: tree.map_structure(
            lambda x: x[-sequence_length:], step[rlds.OBSERVATION]
        ),
        rlds.ACTION: step[rlds.ACTION][-1],
        rlds.IS_TERMINAL: step[rlds.IS_TERMINAL][-1],
    }

  ds = rlds.transformations.pattern_map_from_transform(
      episodes_dataset=ds,
      transform_fn=get_seqlen_pattern,
      # By setting this to true, we don't generate transitions that mix steps
      # from two episodes.
      respect_episode_boundaries=True,
  )

  # Filter out IS_TERMINAL steps, since in RLDS this contains just the terminal
  # observation where the action and reward are meaningless.
  ds = ds.filter(_is_not_terminal)

  # Shuffle, repeat the dataset.
  if shuffle:
    shuffle_rng, rng = jax.random.split(rng)
    shuffle_rng = shuffle_rng[0] if deterministic_shuffle else None
    ds = ds.shuffle(shuffle_buffer_size, shuffle_rng)

  ds = ds.repeat()

  # Image augmentation.
  # Apply preprocessing of the dataset, including decoding JPEGs and adding
  # random crops of images in the observation, and other photometric distortions
  # if set.
  preprocessors = [
      DecodeAndRandomResizedCrop(
          random_crop_factor=random_crop_factor,
          resize_size=(target_height, target_width),
      ),
      PhotometricDistortions(),
  ]
  train_preprocess = preprocess_spec.PreprocessFn(  # pytype: disable=wrong-arg-types  # re-none
      preprocessors, only_jax_types=True
  )

  def _preprocess_fn(example_index, features):
    example_index = tf.cast(example_index, tf.int32)
    features[preprocess_spec.SEED_KEY] = (
        tf.random.experimental.stateless_fold_in(
            tf.cast(rng, tf.int64), example_index
        )
    )
    processed = train_preprocess(features)
    return processed

  ds = ds.enumerate().map(_preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

  # Normalize observations and actions.
  # Normalization statistics are computed only on process_index() == 0. Other
  # processes wait until process 0 writes normalization stats. If we're
  # resuming a job from preemption, we can reload the normalization stats
  # from file.
  if not tf.io.gfile.exists(normalization_path):
    tf.io.gfile.makedirs(normalization_path)

  if jax.process_index() == 0:
    # Check if we already have some normalization info written.
    norm_info = _load_normalization_info(normalization_path)
    if norm_info is not None:
      flat_obs_statistics, flat_act_statistics, min_actions, max_actions = (
          norm_info
      )
      logging.info(
          "Loaded normalization info from file on process %d",
          jax.process_index(),
      )
    else:
      (flat_obs_statistics, flat_act_statistics, min_actions, max_actions) = (
          normalization.compute_dataset_statistics(
              ds.repeat().batch(1),
              num_samples=normalization_num_samples,
              nested_obs=True,
              nested_actions=False,
          )
      )
      logging.info(
          "Computed normalization info on process %d", jax.process_index()
      )
      _save_normalization_info(
          normalization_path,
          flat_obs_statistics,
          flat_act_statistics,
          min_actions,
          max_actions,
      )
  else:
    (flat_obs_statistics, flat_act_statistics, min_actions, max_actions) = (
        _wait_for_normalization_info(normalization_path)
    )
    logging.info(
        "Loaded normalization info from file on process %d", jax.process_index()
    )

  observation_spec = ds.element_spec["observation"]
  action_spec = ds.element_spec["action"]
  obs_normalizers, act_normalizers = normalization.construct_normalizers(
      observation_spec,
      action_spec,
      flat_obs_statistics,
      flat_act_statistics,
      min_actions,
      max_actions,
  )

  obs_statistics = tf.nest.pack_sequence_as(
      observation_spec, flat_obs_statistics
  )
  act_statistics = flat_act_statistics[0]

  skip_normalize_keys = skip_normalize_keys or []

  def _normalize_fn(features):
    flat_obs_keys = tf.nest.flatten(features["observation"].keys())
    flat_obs = tf.nest.flatten(features["observation"])
    flat_obs_normalizers = tf.nest.flatten(obs_normalizers)

    new_observation = {}

    for obs_name, obs_value, normalizer in zip(
        flat_obs_keys, flat_obs, flat_obs_normalizers
    ):
      if obs_name in skip_normalize_keys:
        logging.info("Skipping normalization for: %s", obs_name)
        new_observation[obs_name] = obs_value
        continue
      logging.info("Normalizing: %s", obs_name)
      normalized_obs_value = normalizer(obs_value)
      new_observation[obs_name] = normalized_obs_value

    normalized_action = act_normalizers(features["action"])

    return {
        "observation": new_observation,
        "action": normalized_action,
    }

  ds = ds.map(_normalize_fn, num_parallel_calls=tf.data.AUTOTUNE)

  vocab_lookup = clip_tokenizer.create_vocab()
  tokenizer = clip_tokenizer.ClipTokenizer(vocab_lookup)

  def _tokenize_instruction(features):
    instruction_bytes = features["observation"]["instruction"][0]
    instruction_encoded = tf.strings.unicode_encode(
        instruction_bytes, output_encoding="UTF-8"
    )
    # Remove trailing padding.
    instruction_encoded = tf.strings.split(instruction_encoded, "\x00")[0]

    tokens = clip_tokenizer.tokenize_text(instruction_encoded, tokenizer)
    tokens = tf.tile(tokens, [sequence_length, 1])
    features["observation"]["instruction_tokenized_clip"] = tokens
    return features

  ds = ds.map(_tokenize_instruction, num_parallel_calls=tf.data.AUTOTUNE)

  # Batching.
  # There are 3 batch dimensions. Starting from inner to outer we have:
  # - The per device batch size, this is the batch size per TPU/GPU chip.
  # - The number of steps per train iteration. Since we use a multistep training
  #   loop we add this as a batch dim.
  # - The number of local devices.
  ds = ds.batch(per_device_batch_size, drop_remainder=True)

  if tf_data_service_address:
    ds = ds.apply(
        tf.data.experimental.service.distribute(
            processing_mode=tf.data.experimental.service.ShardingPolicy.OFF,
            service=tf_data_service_address,
            job_name="shared_job",
        )
    )
    ds = ds.apply(
        tf.data.experimental.assert_cardinality(tf.data.INFINITE_CARDINALITY)
    )

  ds = ds.batch(num_steps_per_train_iter, drop_remainder=True)
  ds = ds.batch(jax.local_device_count(), drop_remainder=True)
  ds = ds.prefetch(prefetch_size)
  return ds, obs_statistics, act_statistics, min_actions, max_actions


@dataclasses.dataclass(frozen=True)
class DecodeAndRandomResizedCrop(preprocess_spec.RandomMapTransform):
  """Decodes the images and extracts a random crop."""

  random_crop_factor: Optional[float] = None
  resize_size: Tuple[int, int] = (180, 320)

  def _transform(self, features, seed):
    image = features["observation"]["rgb"]
    shape = tf.io.extract_jpeg_shape(image[0])
    raw_height, raw_width = shape[0], shape[1]
    raw_height = tf.cast(raw_height, tf.float32)
    raw_width = tf.cast(raw_width, tf.float32)

    if self.random_crop_factor is None:
      random_crop_factor = 1.0
      offset_width = 0
      offset_height = 0
      scaled_height = raw_height
      scaled_width = raw_width
    else:
      random_crop_factor = tf.constant(
          self.random_crop_factor, dtype=tf.float32
      )
      scaled_height = raw_height * random_crop_factor
      scaled_width = raw_width * random_crop_factor

      next_rng, rng = tf.unstack(tf.random.experimental.stateless_split(seed))
      offset_height = tf.random.stateless_uniform(
          shape=(),
          seed=next_rng,
          minval=0,
          maxval=tf.cast(raw_height - scaled_height, dtype=tf.int32),
          dtype=tf.int32,
      )

      next_rng, rng = tf.unstack(tf.random.experimental.stateless_split(rng))
      offset_width = tf.random.stateless_uniform(
          shape=(),
          seed=next_rng,
          minval=0,
          maxval=tf.cast(raw_width - scaled_width, dtype=tf.int32),
          dtype=tf.int32,
      )

    def apply_decode_and_crop(image):
      image = tf.image.decode_and_crop_jpeg(
          image,
          [
              offset_height,
              offset_width,
              tf.cast(scaled_height, tf.int32),
              tf.cast(scaled_width, tf.int32),
          ],
          channels=3,
      )
      return image

    image = tf.map_fn(apply_decode_and_crop, image, dtype=tf.uint8)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, self.resize_size)
    features["observation"]["rgb"] = image
    return features


@dataclasses.dataclass(frozen=True)
class PhotometricDistortions(preprocess_spec.RandomMapTransform):
  """Applies photometric distortions to an input image batch."""

  brightness_max_delta: float = 0.1
  contrast_lower: float = 0.8
  contrast_upper: float = 1.2
  hue_max_delta: float = 0.03
  saturation_lower: float = 0.8
  saturation_upper: float = 1.2

  def _transform(self, features, seed):
    """Applies all distortions."""
    image = features["observation"]["rgb"]
    rng0, rng1, rng2, rng3, seed = tf.unstack(
        tf.random.experimental.stateless_split(seed, num=5)
    )

    if self.brightness_max_delta != 0.0:
      # Adjust brightness to a random level.
      image = self._distort(
          image,
          distortion_fn=tf.image.adjust_brightness,
          lower=-self.brightness_max_delta,
          upper=self.brightness_max_delta,
          rng=rng0,
      )
    if self.saturation_lower != 1.0 or self.saturation_upper != 1.0:
      image = self._distort(
          image,
          distortion_fn=tf.image.adjust_saturation,
          lower=self.saturation_lower,
          upper=self.saturation_upper,
          rng=rng1,
      )
    if self.hue_max_delta != 0.0:
      image = self._distort(
          image,
          distortion_fn=tf.image.adjust_hue,
          lower=-self.hue_max_delta,
          upper=self.hue_max_delta,
          rng=rng2,
      )
    # Adjust contrast to a random level.
    if self.contrast_lower != 1.0 or self.contrast_upper != 1.0:
      image = self._distort(
          image,
          distortion_fn=tf.image.adjust_contrast,
          lower=self.contrast_lower,
          upper=self.contrast_upper,
          rng=rng3,
      )
    features["observation"]["rgb"] = image
    return features

  def _distort(self, images, distortion_fn, lower, upper, rng):
    """Applies distortion_fn to video."""

    def get_distortion(rng):
      # Samples distortion value uniformly.
      next_rng, rng = tf.unstack(tf.random.experimental.stateless_split(rng))
      return tf.random.stateless_uniform(
          [], seed=next_rng, minval=lower, maxval=upper
      )

    # Define per-video distortion.
    video_delta = get_distortion(rng)
    return distortion_fn(images, video_delta)


def _add_goal_observations(episode, observation_keys, key_prefix="last_"):
  """Transformation to add observations from the last step of the episode."""
  steps = episode[rlds.STEPS]

  # We batch the episode so that we can process it with the single
  # `process_episode` function.
  episode_length = rlds.transformations.episode_length(steps)
  batched_steps = steps.batch(tf.cast(episode_length, tf.int64))

  def _process_episode(episode_data):
    last_step = tf.nest.map_structure(lambda x: x[-1], episode_data)
    for key in last_step[rlds.OBSERVATION]:
      if key in observation_keys:
        last_step_obs = last_step[rlds.OBSERVATION][key]
        last_step_obs = tf.expand_dims(last_step_obs, 0)
        tiled_last_step_obs = tf.tile(
            last_step_obs, [episode_length] + [tf.shape(last_step_obs)]
        )
        episode_data[rlds.OBSERVATION][
            f"{key_prefix}{key}"
        ] = tiled_last_step_obs
    return episode_data

  batched_steps = batched_steps.map(_process_episode)

  # After applying the function we unbatch to restore to the datastep of steps.
  new_steps = batched_steps.unbatch()
  episode[rlds.STEPS] = new_steps

  return episode


def _load_normalization_info(norm_checkpoint_path):
  """Loads normalization info from the given path, None if files don't exist."""
  checkpoint_contents = tf.io.gfile.listdir(norm_checkpoint_path)
  if (
      "obs.json" not in checkpoint_contents
      or "act.json" not in checkpoint_contents
  ):
    return None
  obs_file = os.path.join(norm_checkpoint_path, "obs.json")
  act_file = os.path.join(norm_checkpoint_path, "act.json")
  act_min_max_file = os.path.join(norm_checkpoint_path, "act_min_max.json")
  # Restore the normalization constants from these files.
  with tf.io.gfile.GFile(obs_file) as f:
    obs_statistics = json.load(f)

  with tf.io.gfile.GFile(act_file) as f:
    act_statistics = json.load(f)

  with tf.io.gfile.GFile(act_min_max_file, "r") as f:
    act_min_max = json.load(f)
    min_actions, max_actions = act_min_max

  return obs_statistics, act_statistics, min_actions, max_actions


def _wait_for_normalization_info(norm_checkpoint_path):
  """Waits for normalization info to be written to given path."""
  checkpoint_contents = tf.io.gfile.listdir(norm_checkpoint_path)
  while (
      "obs.json" not in checkpoint_contents
      or "act.json" not in checkpoint_contents
      or "act_min_max.json" not in checkpoint_contents
  ):
    time.sleep(1.0)
    checkpoint_contents = tf.io.gfile.listdir(norm_checkpoint_path)

  # Wait another few seconds to make sure the file is written.
  time.sleep(5.0)

  norm_info = _load_normalization_info(norm_checkpoint_path)
  if norm_info is None:
    raise ValueError("Got None normalization info.")

  return norm_info


def _save_normalization_info(
    norm_checkpoint_path,
    flat_obs_statistics,
    flat_act_statistics,
    min_actions,
    max_actions,
):
  """Saves normalization info to the given path."""
  # Save these statistics into the proper files.
  obs_file = os.path.join(norm_checkpoint_path, "obs.json")
  act_file = os.path.join(norm_checkpoint_path, "act.json")
  act_min_max_file = os.path.join(norm_checkpoint_path, "act_min_max.json")
  tf.io.gfile.makedirs(norm_checkpoint_path)
  with tf.io.gfile.GFile(obs_file, "w") as f:
    json.dump(flat_obs_statistics, f)

  with tf.io.gfile.GFile(act_file, "w") as f:
    json.dump(flat_act_statistics, f)

  with tf.io.gfile.GFile(act_min_max_file, "w") as f:
    json.dump([min_actions, max_actions], f)

  logging.info("Saved normalization info to %s", norm_checkpoint_path)
