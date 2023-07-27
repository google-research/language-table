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

"""Main file for running the trainer."""

from absl import app
from absl import flags
from absl import logging

from clu import platform
import jax
from language_table.train import train
from ml_collections import config_flags
import tensorflow as tf


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True
)
flags.DEFINE_string("workdir", None, "Work unit directory.")
flags.DEFINE_string("tf_data_service_address", None, "TF Data address.")
flags.mark_flags_as_required(["config", "workdir"])
# Flags --jax_backend_target and --jax_xla_backend are available through JAX.


def main(argv):
  del argv

  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], "GPU")

  if FLAGS.jax_backend_target:
    logging.info("Using JAX backend target %s", FLAGS.jax_backend_target)
    jax_xla_backend = (
        "None" if FLAGS.jax_xla_backend is None else FLAGS.jax_xla_backend
    )
    logging.info("Using JAX XLA backend %s", jax_xla_backend)

  logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
  logging.info("JAX devices: %r", jax.devices())

  platform.work_unit().set_task_status(
      f"process_index: {jax.process_index()}, "
      f"process_count: {jax.process_count()}"
  )
  platform.work_unit().create_artifact(
      platform.ArtifactType.DIRECTORY, FLAGS.workdir, "workdir"
  )

  train.train(FLAGS.config, FLAGS.workdir, FLAGS.tf_data_service_address)


if __name__ == "__main__":
  jax.config.config_with_absl()
  app.run(main)
