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

"""Simple offline evaluation script for language table sim."""

import collections
from collections.abc import Sequence
import os

from absl import app
from absl import flags
from absl import logging

import jax

from language_table.environments import blocks
from language_table.environments import language_table
from language_table.environments.oracles import push_oracle_rrt_slowdown
from language_table.environments.rewards import block2absolutelocation
from language_table.environments.rewards import block2block
from language_table.environments.rewards import block2block_relative_location
from language_table.environments.rewards import block2relativelocation
from language_table.environments.rewards import separate_blocks
from language_table.eval import wrappers as env_wrappers
from language_table.train import policy as jax_policy
from language_table.train.networks import lava

import mediapy as mediapy_lib
from ml_collections import config_flags

import tensorflow as tf
from tf_agents.environments import gym_wrapper
from tf_agents.environments import wrappers as tfa_wrappers

_CONFIG = config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)
_WORKDIR = flags.DEFINE_string("workdir", None, "Evaluation result directory.")
_CHECKPOINT_PATH = flags.DEFINE_string("checkpoint_path", None,
                                       "FLAX checkpoint path.")


def evaluate_checkpoint(checkpoint_path, workdir, config):
  """Evaluates the given checkpoint and writes results to workdir."""
  video_dir = os.path.join(workdir, "videos")
  if not tf.io.gfile.exists(video_dir):
    tf.io.gfile.makedirs(video_dir)
  rewards = {
      "blocktoblock":
          block2block.BlockToBlockReward,
      "blocktoabsolutelocation":
          block2absolutelocation.BlockToAbsoluteLocationReward,
      "blocktoblockrelativelocation":
          block2block_relative_location.BlockToBlockRelativeLocationReward,
      "blocktorelativelocation":
          block2relativelocation.BlockToRelativeLocationReward,
      "separate":
          separate_blocks.SeparateBlocksReward,
  }

  num_evals_per_reward = 50
  max_episode_steps = 200

  policy = None
  model = lava.SequenceLAVMSE(action_size=2, **config.model)

  results = collections.defaultdict(lambda: 0)
  for reward_name, reward_factory in rewards.items():
    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
        reward_factory=reward_factory,
        seed=0)
    env = gym_wrapper.GymWrapper(env)
    env = env_wrappers.ClipTokenWrapper(env)
    env = env_wrappers.CentralCropImageWrapper(
        env,
        target_width=config.data_target_width,
        target_height=config.data_target_height,
        random_crop_factor=config.random_crop_factor)
    env = tfa_wrappers.HistoryWrapper(
        env, history_length=config.sequence_length, tile_first_step_obs=True)

    if policy is None:
      policy = jax_policy.BCJaxPyPolicy(
          env.time_step_spec(),
          env.action_spec(),
          model=model,
          checkpoint_path=checkpoint_path,
          rng=jax.random.PRNGKey(0))

    for ep_num in range(num_evals_per_reward):
      # Reset env. Choose new init if oracle cannot find valid motion plan.
      # Get an oracle. We use this at the moment to decide whether an
      # environment initialization is valid. If oracle can motion plan,
      # init is valid.
      oracle_policy = push_oracle_rrt_slowdown.ObstacleOrientedPushOracleBoard2dRRT(
          env, use_ee_planner=True)
      plan_success = False
      while not plan_success:
        ts = env.reset()
        raw_state = env.compute_state()
        plan_success = oracle_policy.get_plan(raw_state)
        if not plan_success:
          logging.info(
              "Resetting environment because the "
              "initialization was invalid (could not find motion plan).")

      frames = [env.render()]

      episode_steps = 0
      while not ts.is_last():
        policy_step = policy.action(ts, ())
        ts = env.step(policy_step.action)
        frames.append(env.render())
        episode_steps += 1

        if episode_steps > max_episode_steps:
          break

      success_str = ""
      if env.succeeded:
        results[reward_name] += 1
        logging.info("Episode %d: success.", ep_num)
        success_str = "success"
      else:
        logging.info("Episode %d: failure.", ep_num)
        success_str = "failure"

      # Write out video of rollout.
      video_path = os.path.join(workdir, "videos/",
                                f"{reward_name}_{ep_num}_{success_str}.mp4")
      mediapy_lib.write_video(video_path, frames, fps=10)

    print(results)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  evaluate_checkpoint(
      checkpoint_path=_CHECKPOINT_PATH.value,
      workdir=_WORKDIR.value,
      config=_CONFIG.value,
  )


if __name__ == "__main__":
  app.run(main)
