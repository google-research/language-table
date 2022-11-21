# coding=utf-8
# Copyright 2022 The Language Tale Authors.
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

"""A config for training sim human 8 block with BC."""

import ml_collections


def get_config():
  """Config for training on imagenet. Should be used on a 4x4 TPU slice."""
  config = ml_collections.ConfigDict()
  # Note: This is currently not used but it's nice to document the itended
  # binary and allows sharing a XM launcher script for multiple binaries.
  config.binary = "//third_party/py/language_table/train:main"

  config.sequence_length = 4

  config.model_name = "sequence_lav_mse"
  config.model = ml_collections.ConfigDict()
  config.model.dense_resnet_width = 1024
  config.model.dense_resnet_num_blocks = 2
  config.model.lava_sequence_length = config.sequence_length
  config.model.lava_num_layers = 4
  config.model.lava_temporal_transformer_num_layers = 2
  config.model.lava_d_model = 128
  config.model.lava_num_heads = 2
  config.model.lava_pyramid_fuse_layers = (2, 3, 4)
  config.model.lava_image_encoder = "conv_maxpool"
  config.model.lava_lang_encoder = "clip"

  config.agent_name = "bc"
  config.agent = ml_collections.ConfigDict()
  config.agent.learning_rate = 1e-3
  config.agent.pretrained_checkpoints = [
      ("/usr/local/google/home/ayzaan/tmp/scenic_clip_ckpt/",
       [("params/text", "params/encoder/TextEncoder_0")])
  ]
  config.agent.freeze_keys = ["TextEncoder_0"]

  config.dataset_path = "/usr/local/google/home/ayzaan/tmp/language_table_data/language_table_sim/"
  config.data_normalize_rgb = False
  config.data_target_width = 320
  config.data_target_height = 180
  config.image_photometric_distortions = True
  config.image_augment_crop = True
  config.random_crop_factor = 0.95
  config.data_normalization_num_samples = 32
  config.data_skip_normalize_keys = ["rgb", "instruction"]
  config.synthetic_data = False

  config.num_train_steps = 1_000_000
  config.per_device_batch_size = 4  # 4096 is used for 64 TPUs.
  config.replay_capacity = 5_000
  config.num_steps_per_train_iter = 1

  config.log_loss_every_steps = 50
  config.checkpoint_every_steps = 50

  config.seed = 42

  config.trial = 0  # Dummy for repeated runs.
  return config


def get_hyper(h):
  return h.product([
      h.sweep("config.trial", range(1)),
  ])
