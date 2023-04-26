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

"""Tests for language_table."""

from language_table.environments import blocks
from language_table.environments import language_table
from language_table.environments.rewards import block2block
import numpy as np
import tensorflow as tf


class LanguageTableTest(tf.test.TestCase):

  def test_save_restore_pybullet_state(self):
    block_modes = [
        blocks.LanguageTableBlockVariants.BLOCK_4,
        blocks.LanguageTableBlockVariants.BLOCK_8
    ]
    for block_mode in block_modes:
      env = language_table.LanguageTable(
          block_mode=block_mode,
          reward_factory=block2block.BlockToBlockReward,
          control_frequency=10.0,
          seed=0)
      obs0 = env.reset()
      actions = [
          np.random.uniform(size=(2,), low=-0.03, high=0.03) for _ in range(40)
      ]
      observations = [obs0]
      pbstates = [env.get_pybullet_state()]

      # Take actions in environment, storing obs and pbstates.
      for act in actions:
        obs, _, _, _ = env.step(act)
        observations.append(obs)
        pbstates.append(env.get_pybullet_state())

      # Reset env.
      env.reset()

      # Replay states into observations.
      reconstructed_obs = []
      for pb in pbstates:
        env.set_pybullet_state(pb)
        obs = env._compute_state(request_task_update=False)
        reconstructed_obs.append(obs)
      self.assertEqual(len(observations), len(reconstructed_obs))
      for i, (orig_obs,
              recon_obs) in enumerate(zip(observations, reconstructed_obs)):
        for k in orig_obs:
          if k == 'rgb':
            # pybullet rgb is sometimes flaky as the rendering is
            # non-deterministic. Since the individual pixel errors
            # are high magnitude, just make sure the average error
            # is low.
            self.assertLess(
                np.abs(orig_obs[k] - recon_obs[k]).mean(), 1e-6,
                f'Observation {k} at step {i} was not equal.')
          elif k == 'effector_translation':
            self.assertLess(
                np.abs(orig_obs[k] - recon_obs[k]).max(), 1e-3,
                f'Observation {k} at step {i} was not equal.')
          else:
            self.assertLess(
                np.abs(orig_obs[k] - recon_obs[k]).max(), 1e-6,
                f'Observation {k} at step {i} was not equal.')
          self.assertEqual(orig_obs[k].dtype, recon_obs[k].dtype)

  def test_save_restore_pybullet_state_instruction_len(self):
    block_modes = [
        blocks.LanguageTableBlockVariants.BLOCK_4,
        blocks.LanguageTableBlockVariants.BLOCK_8
    ]
    for block_mode in block_modes:
      env = language_table.LanguageTable(
          block_mode=block_mode,
          reward_factory=block2block.BlockToBlockReward,
          control_frequency=10.0,
          seed=0)
      obs0 = env.reset()
      actions = [
          np.random.uniform(size=(2,), low=-0.03, high=0.03) for _ in range(40)
      ]
      observations = [obs0]
      pbstates = [env.get_pybullet_state()]

      # Take actions in environment, storing obs and pbstates.
      for act in actions:
        obs, _, _, _ = env.step(act)
        observations.append(obs)

        pybullet_state = env.get_pybullet_state()

        # Modify the pybullet state to make the instruction length different.
        pybullet_state['instruction'] = pybullet_state['instruction'][:128]

        pbstates.append(pybullet_state)

      # Reset env.
      env.reset()

      # Replay states into observations.
      reconstructed_obs = []
      for pb in pbstates:
        env.set_pybullet_state(pb)
        state = env._compute_state(request_task_update=False)
        obs = env._compute_observation(state=state)

        self.assertTrue(env.observation_space.contains(obs))

        # 256 is the constants.INSTRUCTION_LENGTH
        self.assertEqual(obs['instruction'].size, 512)

        reconstructed_obs.append(obs)
      for i, (orig_obs,
              recon_obs) in enumerate(zip(observations, reconstructed_obs)):
        for k in orig_obs:
          if k == 'rgb':
            # pybullet rgb is sometimes flaky as the rendering is
            # non-deterministic. Since the individual pixel errors
            # are high magnitude, just make sure the average error
            # is low.
            self.assertLess(
                np.abs(orig_obs[k] - recon_obs[k]).mean(), 1e-6,
                f'Observation {k} at step {i} was not equal.')
          elif k == 'effector_translation':
            self.assertLess(
                np.abs(orig_obs[k] - recon_obs[k]).max(), 1e-3,
                f'Observation {k} at step {i} was not equal.')
          else:
            self.assertLess(
                np.abs(orig_obs[k] - recon_obs[k]).max(), 1e-6,
                f'Observation {k} at step {i} was not equal.')
          self.assertEqual(orig_obs[k].dtype, recon_obs[k].dtype)

  def test_environment_initializes_and_resets(self):
    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_4,
        reward_factory=block2block.BlockToBlockReward,
        control_frequency=10.0)
    env.reset()

  def test_environment_obs_space_contains_obs(self):
    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_4,
        reward_factory=block2block.BlockToBlockReward,
        control_frequency=10.0)
    obs = env.reset()
    self.assertTrue(env.observation_space.contains(obs))
    for k in obs:
      self.assertEqual(obs[k].dtype, env.observation_space[k].dtype)

  def test_environment_steps_block4(self):
    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_4,
        reward_factory=block2block.BlockToBlockReward,
        control_frequency=10.0,
        seed=0)
    obs = env.reset()
    for _ in range(5):
      obs, _, _, _ = env.step(env.action_space.sample())
      self.assertTrue(env.observation_space.contains(obs))

  def test_environment_steps_block8(self):
    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
        reward_factory=block2block.BlockToBlockReward,
        control_frequency=10.0,
        seed=0)
    obs = env.reset()
    for _ in range(5):
      obs, _, _, _ = env.step(env.action_space.sample())
      self.assertTrue(env.observation_space.contains(obs))

  def test_environment_steps_nchoosek(self):
    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.N_CHOOSE_K,
        reward_factory=block2block.BlockToBlockReward,
        control_frequency=10.0,
        seed=0)
    obs = env.reset()
    for _ in range(5):
      obs, _, _, _ = env.step(env.action_space.sample())
      self.assertTrue(env.observation_space.contains(obs))


if __name__ == '__main__':
  tf.test.main()
