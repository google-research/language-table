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

"""BC trainer for Language-Table."""
import functools
import os
from typing import Optional
from absl import logging
from clu import checkpoint
from clu import metric_writers
from clu import parameter_overview
from clu import periodic_actions
import flax.jax_utils as flax_utils
from flax.training import checkpoints as flax_checkpoints
import jax
from language_table.train import bc
from language_table.train import input_pipeline_rlds
from language_table.train.networks import lava
from language_table.train.networks import pixel
import ml_collections
import tensorflow as tf


def multi_train_step(state, batches, agent, initial_metrics, rng):
  """Runs multiple iterations of a train step at once."""
  num_batches = batches["action"].shape[0]

  def _body_fun(step, state_and_metrics):
    state, _ = state_and_metrics
    # Important to call fold_in since `rng` is the same on every call.
    train_rng = jax.random.fold_in(rng, state.step)
    train_rng = jax.random.fold_in(train_rng, jax.lax.axis_index("batch"))
    new_state, metrics_update = agent.train(
        state=state,
        batch=jax.tree.map(lambda x: x[step], batches),
        rng=train_rng,
    )
    return new_state, metrics_update

  return jax.lax.fori_loop(
      lower=0,
      upper=num_batches,
      body_fun=_body_fun,
      init_val=(state, initial_metrics),
  )


def train(
    config,
    workdir,
    tf_data_service_address,
):
  """Runs a training loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
    tf_data_service_address: Address of the TF Data Service. If None, all
      dataset computation will run locally.
  """
  tf.io.gfile.makedirs(workdir)
  rng = jax.random.PRNGKey(config.seed)
  logging.info(
      "Global batch size = %d",
      jax.device_count() * config.per_device_batch_size,
  )
  rng, data_rng = jax.random.split(rng)
  data_rng = jax.random.fold_in(data_rng, jax.process_index())
  train_ds, obs_statistics, act_statistics, min_actions, max_actions = (
      input_pipeline_rlds.create_datasets(
          data_rng,
          dataset_path=config.dataset_path,
          normalization_path=os.path.join(workdir, "norm_info"),
          sequence_length=config.sequence_length,
          per_device_batch_size=config.per_device_batch_size,
          num_steps_per_train_iter=config.num_steps_per_train_iter,
          target_width=config.data_target_width,
          target_height=config.data_target_height,
          random_crop_factor=config.random_crop_factor,
          normalization_num_samples=config.data_normalization_num_samples,
          skip_normalize_keys=config.data_skip_normalize_keys,
          cache=True,
          shuffle=True,
          shuffle_buffer_size=config.replay_capacity,
          prefetch_size=tf.data.AUTOTUNE,
          tf_data_service_address=tf_data_service_address,
      )
  )
  train_iter = train_ds.as_numpy_iterator()
  rng, agent_rng = jax.random.split(rng)
  sample_batch = jax.tree.map(lambda x: x[0][0], next(train_iter))
  agent = create_agent(
      config.agent_name,
      config.model_name,
      config.agent,
      config.model,
      config.sequence_length,
      sample_batch,
      obs_statistics,
      act_statistics,
      min_actions,
      max_actions,
  )
  state, initial_metrics = agent.create_train_state(sample_batch, agent_rng)
  # Save a file with the agent parameters.
  if jax.process_index() == 0:
    with tf.io.gfile.GFile(os.path.join(workdir, "parameters.txt"), "w") as f:
      f.write(parameter_overview.get_parameter_overview(state.params))
  # Set up checkpointing of the model and the input pipeline.
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  ckpt = checkpoint.MultihostCheckpoint(checkpoint_dir, max_to_keep=1_000_000)
  state = ckpt.restore_or_initialize(state)
  initial_step = int(state.step)
  initial_multi_step = initial_step // config.num_steps_per_train_iter
  if jax.process_index() == 0:
    flax_checkpoint_dir = os.path.join(workdir, "flax_checkpoints")
    tf.io.gfile.makedirs(flax_checkpoint_dir)
    flax_checkpoints.save_checkpoint(
        flax_checkpoint_dir,
        state,
        step=int(state.step),
        keep=10_000,
        keep_every_n_steps=config.checkpoint_every_steps,
        overwrite=True,
    )
  # Distribute training.
  state = flax_utils.replicate(state)
  rng, train_rng = jax.random.split(rng)
  # Create the pmapped multi_train_step.
  p_train_step = jax.pmap(
      functools.partial(
          multi_train_step,
          agent=agent,
          initial_metrics=initial_metrics,
          rng=train_rng,
      ),
      axis_name="batch",
  )
  if config.num_train_steps > 0:
    num_train_steps = config.num_train_steps
  logging.info("num_train_steps=%d", num_train_steps)
  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.process_index() > 0
  )
  if initial_step == 0:
    writer.write_hparams(dict(config))
  # We use a single thread threadpool for saving checkpoints, so that we're not
  # blocked.
  hooks = []
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps, writer=writer
  )
  if jax.process_index() == 0:
    hooks += [
        report_progress,
    ]
  train_metrics = None
  logging.info("Starting training loop at step %d.", initial_step)
  with metric_writers.ensure_flushes(writer):
    for multi_step in range(
        initial_multi_step, num_train_steps // config.num_steps_per_train_iter
    ):
      # Compute the actual step by multiplying the amount of steps we run
      # per train iteration.
      step = multi_step * config.num_steps_per_train_iter
      # `step` is a Python integer. `state.step` is JAX integer on the GPU/TPU
      # devices.
      is_last_step = step == num_train_steps
      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        batch = next(train_iter)
        state, metrics_update = p_train_step(state=state, batches=batch)
        metric_update = flax_utils.unreplicate(metrics_update)
        train_metrics = (
            metric_update
            if train_metrics is None
            else train_metrics.merge(metric_update)
        )
      logging.log_first_n(
          logging.INFO, "Finished multi training step %d.", 5, multi_step
      )
      logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)
      for h in hooks:
        h(step)
      if step % config.log_loss_every_steps == 0 or is_last_step:
        writer.write_scalars(step, train_metrics.compute())
        train_metrics = None

      def _save_checkpoints(state, step):
        with report_progress.timed("checkpoint"):
          unreplicated_state = flax_utils.unreplicate(state)
          ckpt.save(unreplicated_state)
          if jax.process_index() == 0:
            flax_checkpoints.save_checkpoint(
                flax_checkpoint_dir,
                unreplicated_state,
                step=step,
                keep=10_000,
                keep_every_n_steps=config.checkpoint_every_steps,
                overwrite=True,
            )

      if step % config.checkpoint_every_steps == 0 or is_last_step:
        state = merge_batch_stats(state)
        _save_checkpoints(state, step)
  logging.info("Finishing training at step %d", num_train_steps)


def create_agent(
    agent_name,
    model_name,
    agent_config,
    model_config,
    sequence_length,
    sample_batch,
    obs_statistics,
    act_statistics,
    min_actions,
    max_actions,
):
  """Creates an agent using an agent and model config."""
  # Automatically infer the action size.
  action = sample_batch["action"]
  action_size = action.shape[-1]
  if model_name == "sequence_lav_mse":
    model = lava.SequenceLAVMSE(action_size=action_size, **model_config)
  elif model_name == "pixel_lang_mse":
    model = pixel.PixelLangMSE(action_size=action_size, **model_config)
  else:
    raise NotImplementedError(f"{model_name} not implemented.")
  if agent_name == "bc":
    agent = bc.BCAgent(
        model=model,
        sequence_length=sequence_length,
        observation_statistics=obs_statistics,
        action_statistics=act_statistics,
        action_min=min_actions,
        action_max=max_actions,
        **agent_config,
    )
  else:
    raise NotImplementedError(f"{agent_name} not implemented.")
  return agent


def merge_batch_stats(replicated_state):
  """Merge model batch stats."""
  if jax.tree.leaves(replicated_state.batch_stats):
    cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, "x"), "x")
    return replicated_state.replace(
        batch_stats=cross_replica_mean(replicated_state.batch_stats)
    )
  else:
    return replicated_state
