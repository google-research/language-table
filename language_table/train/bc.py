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

"""Behavioral Cloning Agent."""

from typing import Any, List, Optional

from absl import logging

from clu import checkpoint
from clu import metrics
from clu import parameter_overview

import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import optax


@flax.struct.dataclass
class TrainState:
  step: int
  params: Any
  opt_state: optax.OptState
  batch_stats: Any
  norm_info: Any


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
  """Train metrics for the IBC Agent."""

  learning_rate: metrics.LastValue.from_output("learning_rate")
  loss: metrics.Average.from_output("loss")
  loss_std: metrics.Std.from_output("loss")


class BCAgent(object):
  """Behavioral Cloning Agent from ..."""

  def __init__(self,
               model,
               sequence_length,
               learning_rate,
               observation_statistics,
               action_statistics,
               action_min,
               action_max,
               pretrained_checkpoints = None,
               freeze_keys = None):
    """Creates the agent."""
    self.model = model

    self.sequence_length = sequence_length

    self.pretrained_checkpoints = pretrained_checkpoints or []
    self.freeze_keys = freeze_keys or []

    self.optimizer = None

    self.learning_rate = learning_rate

    self.observation_statistics = observation_statistics
    self.action_statistics = action_statistics

    self.action_min = action_min
    self.action_max = action_max

    logging.info("action_min=%s,action_max=%s", action_min, action_max)

  def create_train_state(self, batch, rng):
    """Creates the train state and initial metrics for agent."""
    obs_input = batch["observation"]

    rng, encoder_rng = jax.random.split(rng)
    variables = self.model.init(encoder_rng, obs_input, train=False)

    # Try to restore.
    flat_variables = flax.traverse_util.flatten_dict(variables, sep="/")

    for pretrained_checkpoint in self.pretrained_checkpoints:
      checkpoint_path, replacements = pretrained_checkpoint
      variable_dict_ckpt = checkpoint.load_state_dict(checkpoint_path)
      flat_variable_ckpt = flax.traverse_util.flatten_dict(
          variable_dict_ckpt, sep="/")
      for ckpt_variable_name, to_variable_name in replacements:
        for key in flat_variable_ckpt:
          if not key.startswith(ckpt_variable_name):
            continue
          variable_key = key.replace(ckpt_variable_name, to_variable_name)
          if variable_key in flat_variables:
            new_value = flat_variable_ckpt[key]
            flat_variables[variable_key] = new_value
            logging.info("Loading %s into %s: shape %s", key, variable_key,
                         new_value.shape)

    variables = flax.traverse_util.unflatten_dict(flat_variables, sep="/")

    params = variables["params"]
    if variables.get("batch_stats"):
      batch_stats = variables["batch_stats"]
    else:
      batch_stats = {}

    # Optionally freeze variables.
    if self.freeze_keys:

      def _should_freeze(path):
        for freeze_key in self.freeze_keys:
          if freeze_key in path:
            logging.info("Freezing param: %s", path)
            return True
        logging.info("Not freezing param: %s", path)
        return False

      label_fn = flattened_traversal(
          lambda path, _: "zero" if _should_freeze("/".join(path)) else "adam")

      optimizer = optax.multi_transform(
          {
              "adam": optax.adam(learning_rate=self.learning_rate, eps=1e-7),
              "zero": optax.set_to_zero()
          }, label_fn)

      self.optimizer = optimizer
    else:
      self.optimizer = optax.adam(learning_rate=self.learning_rate, eps=1e-7)

    parameter_overview.log_parameter_overview(params)
    train_state = TrainState(
        step=0,
        params=params,
        opt_state=self.optimizer.init(params),
        batch_stats=batch_stats,
        norm_info={
            "observation_statistics":
                dict(
                    jax.tree_util.tree_map(
                        jnp.asarray,
                        self.observation_statistics,
                        is_leaf=lambda x: isinstance(x, list))),
            "action_statistics":
                dict(
                    jax.tree_util.tree_map(
                        jnp.asarray,
                        self.action_statistics,
                        is_leaf=lambda x: isinstance(x, list))),
            "action_min":
                self.action_min,
            "action_max":
                self.action_max
        })
    initial_metrics = TrainMetrics.single_from_model_output(
        loss=jnp.zeros((1,)),
        logits=jnp.zeros((1,)),
        negative_logits=jnp.zeros((1,)),
        learning_rate=jnp.zeros((1,)))
    return (train_state, initial_metrics)

  def train(self, batch, state, rng):
    """Performs a single training step."""
    logging.info("train_step(batch=%s)", batch)

    rng, loss_rng = jax.random.split(rng)
    def loss_fn(params):
      variables = {"params": params, "batch_stats": state.batch_stats}
      per_example_loss, new_variables = self.bc_loss(
          self.model, batch=batch, variables=variables, rng=loss_rng)
      loss = jnp.mean(per_example_loss)
      return loss, new_variables["batch_stats"]

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, new_batch_stats), grad = grad_fn(state.params)

    # Compute average gradient across multiple workers.
    grad = jax.lax.pmean(grad, axis_name="batch")
    # Also get the average loss.
    loss = jax.lax.pmean(loss, axis_name="batch")
    updates, new_opt_state = self.optimizer.update(grad, state.opt_state,
                                                   state.params)

    new_params = optax.apply_updates(state.params, updates)
    new_state = state.replace(  # pytype: disable=attribute-error
        step=state.step + 1,
        params=flax.core.unfreeze(new_params),  # pytype: disable=wrong-arg-types  # numpy-scalars
        opt_state=flax.core.unfreeze(new_opt_state),
        batch_stats=flax.core.unfreeze(new_batch_stats))

    metrics_update = TrainMetrics.gather_from_model_output(
        loss=loss, learning_rate=self.learning_rate)
    return new_state, metrics_update

  def bc_loss(
      self,
      model,
      batch,
      variables,
      rng,
  ):
    """Implements the BC loss."""
    # Generate action counter examples.
    # Expand actions on dimension 1.
    observation = batch["observation"]
    action = batch["action"]

    # First, we encode the observations using the model.encode method.
    # This will give us an observation encoding (for the entire sequence).
    rng, params_rng = jax.random.split(rng)
    rng, dropout_rng = jax.random.split(rng)
    predicted_actions, new_variables = model.apply(
        variables,
        observation,
        train=True,
        mutable=["batch_stats"],
        rngs={
            "params": params_rng,
            "dropout": dropout_rng
        })

    per_example_loss = jnp.mean(jnp.square(predicted_actions - action))
    return per_example_loss, new_variables


def flattened_traversal(fn):
  """Returns function that is called with `(path, param)` instead of pytree."""

  def mask(tree):
    flat = flax.traverse_util.flatten_dict(tree)
    return flax.traverse_util.unflatten_dict(
        {k: fn(k, v) for k, v in flat.items()})

  return mask
