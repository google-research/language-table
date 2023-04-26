# coding=utf-8
# Copyright 2023 The Language Tale Authors.
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

"""PyPolicy for BC Jax."""

from flax.training import checkpoints
import jax
import jax.numpy as jnp
import numpy as np
from tf_agents.policies import py_policy
from tf_agents.trajectories import policy_step

EPS = jnp.finfo(jnp.float32).eps


class BCJaxPyPolicy(py_policy.PyPolicy):
  """Runs inference with a BC policy."""

  def __init__(self, time_step_spec, action_spec, model, checkpoint_path,
               rng, params=None, action_statistics=None):
    super(BCJaxPyPolicy, self).__init__(time_step_spec, action_spec)
    self.model = model
    self.rng = rng

    if params is not None and action_statistics is not None:
      variables = {
          "params": params,
          "batch_stats": {}
      }
    else:
      state_dict = checkpoints.restore_checkpoint(checkpoint_path, None)
      variables = {
          "params": state_dict["params"],
          "batch_stats": state_dict["batch_stats"]
      }

    if action_statistics is not None:
      self.action_mean = np.array(action_statistics["mean"])
      self.action_std = np.array(action_statistics["std"])
    else:
      # We can load the observation and action statistics from the state dict.
      self.action_mean = np.array(
          state_dict["norm_info"]["action_statistics"]["mean"])
      self.action_std = np.array(
          state_dict["norm_info"]["action_statistics"]["std"])

      self._rgb_mean = jnp.array(
          state_dict["norm_info"]["observation_statistics"]["rgb"]["mean"])
      self._rgb_std = jnp.array(
          state_dict["norm_info"]["observation_statistics"]["rgb"]["std"])

    self.variables = variables

    self._run_action_inference_jit = jax.jit(self._run_action_inference)

  def _run_action_inference(self, observation):
    # Add a batch dim.
    observation = jax.tree_map(lambda x: jnp.expand_dims(x, 0), observation)

    normalized_action = self.model.apply(
        self.variables, observation, train=False)
    action = (
        normalized_action * jnp.maximum(self.action_std, EPS) +
        self.action_mean)

    # Clip the action to spec.
    action = jnp.clip(action, self.action_spec.minimum,
                      self.action_spec.maximum)

    return action

  def _action(self, time_step, policy_state=(), seed=0):
    observation = time_step.observation
    action = self._run_action_inference_jit(observation)[0]
    return policy_step.PolicyStep(action=action)
