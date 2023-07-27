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

"""Language Table environment implementation."""

import collections
import copy
import math
import textwrap
import time
from typing import Dict, List, Tuple, Union

import cv2
import gym
from gym import spaces

from language_table.environments import blocks as blocks_module
from language_table.environments import constants
from language_table.environments.rewards import task_info
from language_table.environments.utils import utils_pybullet
from language_table.environments.utils import xarm_sim_robot
from language_table.environments.utils.pose3d import Pose3d
from language_table.environments.utils.utils_pybullet import add_visual_sphere
from language_table.environments.utils.utils_pybullet import ObjState
from language_table.environments.utils.utils_pybullet import XarmState

import numpy as np
from scipy.spatial import transform
import pybullet
import pybullet_utils.bullet_client as bullet_client


class LanguageTable(gym.Env):
  """LanguageTable env."""

  def __init__(self,
               block_mode,
               training=True,
               reward_factory=None,
               control_frequency=10.0,
               shared_memory=False,
               seed=None,
               delay_reward_steps=0,
               render_text_in_image=True,
               debug_visuals = False,
               add_invisible_walls = False):
    """Creates an env instance.

    Args:
      block_mode: Enum, defining whether we are using 4 blocks, 8 blocks, or "n
        choose k" blocks. See language_table/blocks.py.
      training: Bool, whether we are in training or test mode. Only useful for
        selecting "training" or "test" "n choose k" block subsets.
      reward_factory: Optional, a factory for creating a reward, otherwise
        episode will run forever.
      control_frequency: Control frequency for the arm. Each env step will
        advance the simulation by 1/control_frequency seconds.
      shared_memory: If True `pybullet.SHARED_MEMORY` is used to connect to
        pybullet. Useful to debug.
      seed: Optional seed for the environment.
      delay_reward_steps: int, delay returning reward for number of steps.
        During the delay, state should be within the reward-triggering region.
      render_text_in_image: bool, Whether to add instruction info to the image.
      debug_visuals: bool, Whether to render debug visuals.
      add_invisible_walls: bool, (Experimental) Whether to add walls that
        prevent the blocks from sliding off the edges.
    """
    self._block_mode = block_mode
    self._training = training
    self._reward_calculator = None
    self._instruction = None
    self._instruction_str = None
    self._start_block = blocks_module.DUMMY_START_BLOCK
    self._oracle_target_block = None
    # Optional invisible target for "push to location" type tasks.
    self._oracle_target_translation = None
    self._target_absolute_location = None
    self._target_relative_location = None

    self._debug_visuals = debug_visuals
    self._add_invisible_walls = add_invisible_walls

    self._render_text_in_image = render_text_in_image

    self._rng = np.random.RandomState(seed=seed)

    self._camera_pose = constants.CAMERA_POSE
    self._camera_orientation = constants.CAMERA_ORIENTATION
    self._workspace_bounds = constants.WORKSPACE_BOUNDS
    self._image_size = (constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH)
    self._camera_instrinsics = constants.CAMERA_INTRINSICS

    self._connection_mode = pybullet.DIRECT
    if shared_memory:
      self._connection_mode = pybullet.SHARED_MEMORY
    self._setup_pybullet_scene()
    self._saved_state = None

    image_size = (constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH)
    self.image_size = image_size

    self._control_frequency = control_frequency
    self._step_frequency = (
        1 / self._pybullet_client.getPhysicsEngineParameters()['fixedTimeStep'])

    self._last_loop_time = None
    self._last_loop_frame_sleep_time = None
    if self._step_frequency % self._control_frequency != 0:
      raise ValueError('Control frequency should be a multiple of the '
                       'configured Bullet TimeStep.')
    self._sim_steps_per_step = int(self._step_frequency /
                                   self._control_frequency)

    # Use saved_state and restore to make reset safe as no simulation state has
    # been updated at this state, but the assets are now loaded.
    self._save_state()
    self.reset()

    if reward_factory is not None:
      # Construct the reward factory into a reward object.
      self._reward_calculator = reward_factory(
          goal_reward=100.0,
          rng=self._rng,
          delay_reward_steps=delay_reward_steps,
          block_mode=block_mode)

    self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))  # x, y
    # Recreate the observation space after creating the reward.
    self.observation_space = self._create_observation_space(image_size)

  def reset(self, reset_poses = True):
    # Choose a subset of the possible blocks to be on the table.
    all_combinations = blocks_module.get_all_block_subsets(
        self._block_mode, self._training)
    combo_idx = self._rng.choice(range(len(all_combinations)))
    blocks_on_table = all_combinations[combo_idx]

    if reset_poses:
      self._reset_poses_randomly(blocks_on_table)

    self._blocks_on_table = blocks_on_table
    # Recompute state to include text instruction.
    state = self._compute_state()
    self._previous_state = state

    # Add goal region sphere.
    if self._reward_calculator is not None and self._debug_visuals:
      target_translation, goal_radius = self._reward_calculator.get_goal_region(
      )
      if target_translation is not None and goal_radius is not None:
        center = (target_translation[0], target_translation[1], 0.0)
        self._pybullet_client.removeBody(self._goal_sphere)
        add_visual_sphere(
            self._pybullet_client,
            center=center,
            radius=goal_radius,
            rgba=(0, 1, 0, 0.5))

    observation = self._compute_observation(state=state)
    return observation

  def seed(self, seed=None):
    rng = np.random.RandomState(seed=seed)
    self._rng = rng
    self._reward_calculator.seed(rng)

  def step(self, action):
    self._step_robot_and_sim(action)

    state = self._compute_state()

    if self._reward_calculator is None:
      reward = 0.0
      done = False
    else:
      reward, done = self._reward_calculator.reward(state)
    observation = self._compute_observation(state=state)
    return observation, reward, done, {}

  def render(self, mode='rgb_array'):
    image = self._render_camera(image_size=self._image_size)
    if not self._render_text_in_image:
      return image
    debug_info = {}
    if self._instruction is not None:
      debug_info['instruction'] = self._instruction_str
    return add_debug_info_to_image(image, debug_info)

  @property
  def succeeded(self):
    state = self._compute_state()
    reward, _ = self._reward_calculator.reward(state)
    # Assume all board2d rewards are sparse.
    return reward > 0.

  @staticmethod
  def encode_instruction(instruction):
    if not instruction:
      return np.array([0] * constants.INSTRUCTION_LENGTH, dtype=np.int32)
    else:
      # Convert to bytes.
      encoded_instruction = instruction.encode('utf-8')
      bytes_list = list(encoded_instruction)
      if len(bytes_list) > constants.INSTRUCTION_LENGTH:
        raise ValueError(
            'Instruction length too long %d > %d; %s' %
            (len(bytes_list), constants.INSTRUCTION_LENGTH, instruction))
      # pad to context length.
      bytes_list = bytes_list + [0] * (
          constants.INSTRUCTION_LENGTH - len(bytes_list))
      return np.array(bytes_list, dtype=np.int32)

  @staticmethod
  def decode_instruction(bytes_list):
    non_zero = bytes_list[np.where(bytes_list != 0)]
    if non_zero.shape[0] == 0:
      return ''
    else:
      bytes_list = bytes(non_zero.tolist())
      return bytes_list.decode('utf-8')

  def get_pybullet_state(self):
    """Save pybullet state of the scene.

    Returns:
      dict containing 'robots', 'robot_end_effectors', 'targets', 'objects',
        each containing a list of ObjState.
    """
    state: Dict[str, List[ObjState]] = {}

    state['robots'] = [
        XarmState.get_bullet_state(
            self._pybullet_client,
            self._robot.xarm,
            target_effector_pose=self._target_effector_pose,
            goal_translation=None)
    ]

    state['robot_end_effectors'] = []
    if self._robot.end_effector:
      state['robot_end_effectors'].append(
          ObjState.get_bullet_state(self._pybullet_client,
                                    self._robot.end_effector))

    state['objects'] = []
    for obj_id in self._block_ids:
      state['objects'].append(
          ObjState.get_bullet_state(self._pybullet_client, obj_id))

    # Add oracle task info to pybullet state.
    oracle_info = dict(
        start_block=self._start_block,
        oracle_target_block=self._oracle_target_block,
        target_absolute_location=self._target_absolute_location,
        target_relative_location=self._target_relative_location,
        oracle_target_translation=self._oracle_target_translation)
    for k, v in oracle_info.items():
      if v is not None:
        state[k] = v

    if self._start_block is not None:
      state['start_block'] = self.encode_instruction(self._start_block).tolist()
    if self._oracle_target_block is not None:
      state['oracle_target_block'] = self.encode_instruction(
          self._oracle_target_block).tolist()
    if self._target_absolute_location is not None:
      state['target_absolute_location'] = self.encode_instruction(
          self._target_absolute_location).tolist()
    if self._target_relative_location is not None:
      state['target_relative_location'] = self.encode_instruction(
          self._target_relative_location).tolist()
    if self._oracle_target_translation is not None:
      state['oracle_target_translation'] = (
          self._oracle_target_translation.tolist())

    # Add language instruction state.
    if self._instruction_str is not None:
      state['instruction_str'] = self.encode_instruction(
          self._instruction_str).tolist()
    if self._instruction is not None:
      state['instruction'] = self._instruction.tolist()
    return state

  def set_pybullet_state(self, state):
    """Restore pyullet state.

    WARNING: py_environment wrapper assumes environments aren't reset in their
    constructor and will often reset the environment unintentionally. It is
    always recommeneded that you call env.reset on the tfagents wrapper before
    playback (replaying pybullet_state).

    Args:
      state: dict containing 'robots', 'robot_end_effectors', 'targets',
        'objects', each containing a list of ObjState.
    """
    assert isinstance(state['robots'][0], XarmState)
    xarm_state: XarmState = state['robots'][0]
    xarm_state.set_bullet_state(self._pybullet_client, self._robot.xarm)
    self._set_robot_target_effector_pose(xarm_state.target_effector_pose)

    def _set_state_safe(obj_state, obj_id):
      if obj_state is not None:
        assert obj_id is not None, 'Cannot set state for missing object.'
        obj_state.set_bullet_state(self._pybullet_client, obj_id)
      else:
        assert obj_id is None, f'No state found for obj_id {obj_id}'

    robot_end_effectors = state['robot_end_effectors']
    _set_state_safe(None if not robot_end_effectors else robot_end_effectors[0],
                    self._robot.end_effector)

    obj_ids = self._block_ids

    assert len(state['objects']) == len(obj_ids)
    for obj_state in state['objects']:
      _set_state_safe(obj_state, obj_state.obj_id)

    # Set oracle task info.
    if 'start_block' in state:
      self._start_block = self.decode_instruction(
          np.array(state['start_block']))
    if 'oracle_target_block' in state:
      self._oracle_target_block = self.decode_instruction(
          np.array(state['oracle_target_block']))
    if 'target_absolute_location' in state:
      self._target_absolute_location = self.decode_instruction(
          np.array(state['target_absolute_location']))
    if 'target_relative_location' in state:
      self._target_relative_location = self.decode_instruction(
          np.array(state['target_relative_location']))
    if 'oracle_target_translation' in state:
      self._oracle_target_translation = np.array(
          state['oracle_target_translation'])

    # Set the instruction state.
    if state.get('instruction_str'):
      instruction_str = state['instruction_str']
      self._instruction_str = self.decode_instruction(np.array(instruction_str))
    if state.get('instruction'):
      instruction = state['instruction']
      # Some backwards compatibility to fix cases where the old instruction
      # length was different.
      if len(instruction) < constants.INSTRUCTION_LENGTH:
        instruction = np.pad(
            instruction, (0, constants.INSTRUCTION_LENGTH - len(instruction)))
      self._instruction = np.array(instruction, dtype=np.int32)
    self.reset(reset_poses=False)

  @property
  def pybullet_client(self):
    return self._pybullet_client

  def calc_camera_params(self, image_size):
    # Mimic RealSense D415 camera parameters.
    intrinsics = self._camera_instrinsics

    # Set default camera poses.
    front_position = self._camera_pose
    front_rotation = self._camera_orientation
    front_rotation = self._pybullet_client.getQuaternionFromEuler(
        front_rotation)
    # Default camera configs.
    zrange = (0.01, 10.)

    # OpenGL camera settings.
    lookdir = np.float32([0, 0, 1]).reshape(3, 1)
    updir = np.float32([0, -1, 0]).reshape(3, 1)
    rotation = self._pybullet_client.getMatrixFromQuaternion(front_rotation)
    rotm = np.float32(rotation).reshape(3, 3)
    lookdir = (rotm @ lookdir).reshape(-1)
    updir = (rotm @ updir).reshape(-1)
    lookat = front_position + lookdir
    focal_len = intrinsics[0]
    znear, zfar = zrange
    viewm = self._pybullet_client.computeViewMatrix(front_position, lookat,
                                                    updir)
    fovh = (image_size[0] / 2) / focal_len
    fovh = 180 * np.arctan(fovh) * 2 / np.pi

    # Notes: 1) FOV is vertical FOV 2) aspect must be float
    aspect_ratio = image_size[1] / image_size[0]
    projm = self._pybullet_client.computeProjectionMatrixFOV(
        fovh, aspect_ratio, znear, zfar)

    return viewm, projm, front_position, lookat, updir

  def compute_state(
      self,
      request_task_update=True):
    # Non-private method for _compute_state.
    # NOTE: this is for project-reach case where gym fails to read private
    # members of wrapped envs.
    return self._compute_state(request_task_update)

  def _compute_observation(self, state=None):
    if state is None:
      state = self._compute_state()
    obs = collections.OrderedDict(
        effector_translation=state['effector_translation'],
        effector_target_translation=state['effector_target_translation'],
        instruction=state['instruction'],
        rgb=state['rgb'],
    )
    return obs

  def _compute_state(
      self,
      request_task_update=True):
    """Computes the state.

    Args:
      request_task_update: Bool, whether to request updated task information
        from the reward definition object. We do this if instruction, targets
        are changing over long horizon.

    Returns:
      obs: ordered obs dict.
    """
    effector_pose = self._robot.forward_kinematics()

    poses = {}
    for block in self._get_urdf_paths().keys():
      block_id = self._block_to_pybullet_id[block]
      block_position_and_orientation = self._pybullet_client.getBasePositionAndOrientation(
          block_id)
      poses[block] = block_position_and_orientation

    e_target_trans = np.array(self._target_effector_pose.translation[0:2],
                              np.float32)
    obs = collections.OrderedDict(
        effector_target_to_start_block_translation=np.array(
            poses[self._start_block][0][0:2] - e_target_trans, np.float32),
        start_block_orientation=self._yaw_from_pose(poses[self._start_block]))

    for k in self._get_urdf_paths().keys():
      obs['block_%s_translation' % k] = np.array(poses[k][0][0:2], np.float32)
      obs['block_%s_orientation' % k] = self._yaw_from_pose(poses[k])
      mask = 1. if k in self._blocks_on_table else 0.
      obs['block_%s_mask' % k] = np.array([mask], dtype=np.float32)

    # For long horizon tasks, we may have updated what the `start_block` is.
    # E.g. if "push red to blue" is satisfied and it's time for "push green to
    # red", we need to update `start_block` from "red" to "green". Perform
    # this check.
    if (hasattr(self._reward_calculator, 'get_current_task_info') and
        request_task_update):
      # Get updated task info.
      updated_info = self._reward_calculator.get_current_task_info(obs)
      # This updates 1) instruction, 2) which block we're pushing aka
      # `start_block`, 3) where we're pushing to aka `target_translation`.
      self._set_task_info(updated_info)
      # If we've changed which block is the `pushing_block` we need to
      # update the observation.
      obs = self._update_obs_with_oracle_task_info(obs, poses, e_target_trans)

    obs = self._update_obs_with_oracle_task_info(obs, poses, e_target_trans)
    obs.update(
        collections.OrderedDict(
            effector_translation=np.array(effector_pose.translation[0:2],
                                          np.float32),
            effector_target_translation=e_target_trans))
    obs['instruction'] = self._instruction

    image = self._render_camera(self._image_size)
    obs['rgb'] = image
    return obs

  def _create_observation_space(self, image_size):
    # Observations.
    obs_dict = collections.OrderedDict(
        effector_translation=spaces.Box(
            # Small buffer for IK noise.
            low=constants.WORKSPACE_BOUNDS[0] - 0.1,
            high=constants.WORKSPACE_BOUNDS[1] + 0.1,
        ),  # x,y
        effector_target_translation=spaces.Box(
            # Small buffer for IK noise.
            low=constants.WORKSPACE_BOUNDS[0] - 0.1,
            high=constants.WORKSPACE_BOUNDS[1] + 0.1,
        ))  # x,y

    # Holds onto language_table.encode encoded instruction.
    obs_dict['instruction'] = spaces.Box(
        low=0,
        high=2147483647,
        shape=(constants.INSTRUCTION_LENGTH,),
        dtype=np.int32)

    obs_dict['rgb'] = spaces.Box(
        low=0,
        high=255,
        shape=(image_size[0], image_size[1], 3),
        dtype=np.uint8)

    return spaces.Dict(obs_dict)

  def _update_obs_with_oracle_task_info(self, obs, poses, e_target_trans):
    # Tell obs what the 'start_block' is. This is the block the oracle is
    # pushing.
    obs['effector_target_to_start_block_translation'] = np.array(
        poses[self._start_block][0][0:2] - e_target_trans, np.float32)
    obs['start_block_orientation'] = self._yaw_from_pose(
        poses[self._start_block])

    # Tell obs what the 'target_translation' is. This is the invisible target
    # the oracle is pushing to.
    # We prioritize using target_translation over target_block if the task
    # gives us this info.
    if self._oracle_target_translation is not None:
      # TODO(coreylynch): right now we are just reusing this
      # "distance to target" feature across multiple tasks
      # (block2block, block2location) as info that allows the oracle to
      # complete the task. We may want to refactor this.
      obs['effector_target_to_task_target_translation'] = np.array(
          self._oracle_target_translation - e_target_trans, np.float32)
      obs['task_target_orientation'] = np.array([0.], np.float32)
    # Tell obs what the 'target_block' is. This is the block the oracle is
    # pushing to.
    elif self._oracle_target_block is not None:
      obs['effector_target_to_task_target_translation'] = np.array(
          poses[self._oracle_target_block][0][0:2] - e_target_trans, np.float32)
      obs['task_target_orientation'] = self._yaw_from_pose(
          poses[self._oracle_target_block])
    else:
      obs['effector_target_to_task_target_translation'] = np.array([0., 0.],
                                                                   np.float32)
      obs['task_target_orientation'] = np.array([0.], np.float32)
    return obs

  def _yaw_from_pose(self, pose):
    return np.array([self._pybullet_client.getEulerFromQuaternion(pose[1])[-1]],
                    np.float32)

  def _step_simulation_to_stabilize(self, nsteps = 100):
    for _ in range(nsteps):
      self._pybullet_client.stepSimulation()

  def _setup_workspace_and_robot(self, end_effector = 'suction'):
    self._pybullet_client.resetSimulation()
    self._pybullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
    self._pybullet_client.setPhysicsEngineParameter(enableFileCaching=0)
    self._pybullet_client.setGravity(0, 0, -9.8)

    utils_pybullet.load_urdf(
        self._pybullet_client,
        constants.PLANE_URDF_PATH,
        basePosition=[0, 0, -0.001])
    self._workspace_uid = utils_pybullet.load_urdf(
        self._pybullet_client,
        constants.WORKSPACE_URDF_PATH,
        basePosition=[0.35, 0, 0.0])

    initial_joint_positions = constants.INITIAL_JOINT_POSITIONS
    self._robot = xarm_sim_robot.XArmSimRobot(
        self._pybullet_client,
        initial_joint_positions=initial_joint_positions,
        end_effector=end_effector,
        color='white')

  def _save_state(self):
    self._saved_state = self._pybullet_client.saveState()

  def _set_robot_target_effector_pose(self, pose):
    self._target_effector_pose = pose
    self._robot.set_target_effector_pose(pose)

  def _render_camera(self, image_size, return_seg_mask=False):
    """Render RGB image with RealSense configuration."""
    viewm, projm, _, _, _ = self.calc_camera_params(image_size)  # pytype: disable=wrong-arg-types  # always-use-return-annotations

    # Render with OpenGL camera settings.
    _, _, color, _, _ = self._pybullet_client.getCameraImage(
        width=image_size[1],
        height=image_size[0],
        viewMatrix=viewm,
        projectionMatrix=projm,
        flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        renderer=pybullet.ER_TINY_RENDERER)

    # Get color image.
    color_image_size = (image_size[0], image_size[1], 4)
    color = np.array(color, dtype=np.uint8).reshape(color_image_size)
    color = color[:, :, :3]  # remove alpha channel

    return color.astype(np.uint8)

  def _step_robot_and_sim(self, action):
    """Steps the robot and pybullet sim."""
    # Compute target_effector_pose by shifting the effector's pose by the
    # action.
    target_effector_translation = np.array(
        self._target_effector_pose.translation) + np.array(
            [action[0], action[1], 0])

    target_effector_translation[0:2] = np.clip(target_effector_translation[0:2],
                                               self._workspace_bounds[0],
                                               self._workspace_bounds[1])
    target_effector_translation[-1] = constants.EFFECTOR_HEIGHT
    target_effector_pose = Pose3d(
        rotation=constants.EFFECTOR_DOWN_ROTATION,
        translation=target_effector_translation)

    self._set_robot_target_effector_pose(target_effector_pose)

    # Update sleep time dynamically to stay near real-time.
    frame_sleep_time = 0
    if self._connection_mode == pybullet.SHARED_MEMORY:
      cur_time = time.time()
      if self._last_loop_time is not None:
        # Calculate the total, non-sleeping time from the previous frame, this
        # includes the actual step as well as any compute that happens in the
        # caller thread (model inference, etc).
        compute_time = (
            cur_time - self._last_loop_time -
            self._last_loop_frame_sleep_time * self._sim_steps_per_step)
        # Use this to calculate the current frame's total sleep time to ensure
        # that env.step runs at policy rate. This is an estimate since the
        # previous frame's compute time may not match the current frame.
        total_sleep_time = max((1 / self._control_frequency) - compute_time, 0)
        # Now spread this out over the inner sim steps. This doesn't change
        # control in any way, but makes the animation appear smooth.
        frame_sleep_time = total_sleep_time / self._sim_steps_per_step
      else:
        # No estimate of the previous frame's compute, assume it is zero.
        frame_sleep_time = 1 / self._step_frequency

      # Cache end of this loop time, to compute sleep time on next iteration.
      self._last_loop_time = cur_time
      self._last_loop_frame_sleep_time = frame_sleep_time

    for _ in range(self._sim_steps_per_step):
      if self._connection_mode == pybullet.SHARED_MEMORY:
        sleep_spin(frame_sleep_time)
      self._pybullet_client.stepSimulation()

  def _setup_pybullet_scene(self):
    self._pybullet_client = bullet_client.BulletClient(self._connection_mode)

    # Temporarily disable rendering to speed up loading URDFs.
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

    self._setup_workspace_and_robot(end_effector='cylinder_real')

    self._block_ids = []
    self._block_to_pybullet_id = {}

    for block, path in self._get_urdf_paths().items():
      block_id = utils_pybullet.load_urdf(self._pybullet_client, path)
      self._block_ids.append(block_id)
      self._block_to_pybullet_id[block] = block_id

    if self._debug_visuals:
      self._goal_sphere = add_visual_sphere(
          self._pybullet_client,
          center=(0, 0, 0),
          radius=0.02,
          rgba=(0, 1, 0, 0))
      # Board boundary markers.
      # Top left.
      buffer = 0.02
      buffer_x_max = 0.04
      # TODO(peteflornece): these adjusted positions are not yet
      # reflected in metrics.
      add_visual_sphere(
          self._pybullet_client,
          center=(constants.X_MIN + buffer, constants.Y_MIN + buffer, 0),
          radius=0.005,
          rgba=(1, 0, 0, 0))
      # Top center.
      add_visual_sphere(
          self._pybullet_client,
          center=(constants.CENTER_X, constants.Y_MIN + buffer, 0),
          radius=0.005,
          rgba=(1, 0, 0, 0))
      # Top right.
      add_visual_sphere(
          self._pybullet_client,
          center=(constants.X_MAX - buffer_x_max, constants.Y_MIN + buffer, 0),
          radius=0.005,
          rgba=(1, 0, 0, 0))
      # Middle left.
      add_visual_sphere(
          self._pybullet_client,
          center=(constants.X_MIN + buffer, constants.CENTER_Y, 0),
          radius=0.005,
          rgba=(1, 0, 0, 0))
      # Middle center.
      add_visual_sphere(
          self._pybullet_client,
          center=(constants.CENTER_X, constants.CENTER_Y, 0),
          radius=0.005,
          rgba=(1, 0, 0, 0))
      # Middle right.
      add_visual_sphere(
          self._pybullet_client,
          center=(constants.X_MAX - buffer_x_max, constants.CENTER_Y, 0),
          radius=0.005,
          rgba=(1, 0, 0, 0))
      # Bottom left.
      add_visual_sphere(
          self._pybullet_client,
          center=(constants.X_MIN + buffer, constants.Y_MAX, 0),
          radius=0.005,
          rgba=(1, 0, 0, 0))
      # Bottom middle.
      add_visual_sphere(
          self._pybullet_client,
          center=(constants.CENTER_X, constants.Y_MAX - buffer, 0),
          radius=0.005,
          rgba=(1, 0, 0, 0))
      # Bottom right.
      add_visual_sphere(
          self._pybullet_client,
          center=(constants.X_MAX - buffer_x_max, constants.Y_MAX - buffer, 0),
          radius=0.005,
          rgba=(1, 0, 0, 0))

    # Re-enable rendering.
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

    if self._add_invisible_walls:
      self._construct_invisible_walls()

    self._step_simulation_to_stabilize()

  def _get_urdf_paths(self):
    paths = copy.deepcopy(blocks_module.BLOCK_URDF_PATHS)
    return paths

  def _set_task_info(self, info):
    """Cache info defining the task from a reward calculator."""
    self._task_info = info
    if isinstance(info, task_info.Block2BlockTaskInfo):
      # Get the info parameterizing "block2block" tasks.
      instruction_str = info.instruction
      self._start_block = info.block1
      self._oracle_target_block = info.block2
      self._oracle_target_translation = None
      self._target_absolute_location = None
      self._target_relative_location = None
    elif isinstance(info, task_info.Block2LocationTaskInfo):
      # Get info parameterizing "block2location" tasks.
      instruction_str = info.instruction
      self._start_block = info.block
      self._oracle_target_block = None
      self._oracle_target_translation = info.target_translation
      self._target_absolute_location = info.location
      self._target_relative_location = None
      self._target_translation = info.target_translation
    elif isinstance(info, task_info.Block2LineTaskInfo):
      # Get the info parameterizing "block2goal" tasks.
      instruction_str = info.instruction
      self._start_block = info.block
      self._oracle_target_block = None
      self._target_absolute_location = None
      self._target_relative_location = None
      self._oracle_target_translation = info.target_translation
    elif isinstance(info, task_info.Block2RelativeLocationTaskInfo):
      # Get info parameterizing "block2location" tasks.
      instruction_str = info.instruction
      self._start_block = info.block
      self._oracle_target_block = None
      self._target_absolute_location = None
      self._target_relative_location = info.location
      self._oracle_target_translation = info.target_translation
    elif isinstance(info, task_info.Block2BlockRelativeLocationTaskInfo):
      # Get info parameterizing "block2location" tasks.
      instruction_str = info.instruction
      self._start_block = info.block
      self._oracle_target_block = info.target_block
      self._target_absolute_location = None
      self._target_relative_location = info.direction
      self._oracle_target_translation = info.target_translation
    elif isinstance(info, task_info.SeparateBlocksTaskInfo):
      # Get info parameterizing "block2location" tasks.
      instruction_str = info.instruction
      self._start_block = info.block
      self._oracle_target_block = None
      self._target_absolute_location = None
      self._target_relative_location = None
      self._oracle_target_translation = info.target_translation
    elif isinstance(info, task_info.Point2BlockTaskInfo):
      # Get the info parameterizing "point2block" tasks.
      instruction_str = info.instruction
      self._start_block = info.block_target
      self._oracle_target_block = info.block_target
      self._oracle_target_translation = None
      self._target_absolute_location = None
      self._target_relative_location = None
    elif isinstance(info, task_info.Block2PoleTaskInfo):
      # Get the info parameterizing "block2pole" tasks.
      instruction_str = info.instruction
      self._start_block = info.block1
      self._oracle_target_block = info.goal
      self._oracle_target_translation = None
      self._target_absolute_location = None
      self._target_relative_location = None
    else:
      raise ValueError('Unknown task info: %s' % info)

    if (self._oracle_target_block is None and
        self._oracle_target_translation is None):
      raise ValueError('We are not providing the oracle with a target location '
                       'to push a block to. Either _oracle_target_block or '
                       '_oracle_target_translation must be set by the reward '
                       'calculator.')
    self._instruction_str = instruction_str
    self._instruction = self.encode_instruction(self._instruction_str)

  def _reset_poses_randomly(self, blocks_on_table):
    self._pybullet_client.restoreState(self._saved_state)
    # TODO(oars): Seems like restoreState doesn't clear JointMotorControl.

    # Move the visible blocks off board for a bit while resetting end
    # effector.
    visible_block_ids = [self._block_to_pybullet_id[i] for i in blocks_on_table]
    for block_id in visible_block_ids:
      far_translation = np.array([5., 5., 0.])
      far_rotation = transform.Rotation.from_rotvec([np.pi / 2, 0, 0])
      self._pybullet_client.resetBasePositionAndOrientation(
          block_id, far_translation.tolist(),
          far_rotation.as_quat().tolist())
    # Step simulation to move blocks out of the way.
    self._step_simulation_to_stabilize()

    xmin, ymin = (constants.X_MIN + constants.WORKSPACE_BOUNDS_BUFFER,
                  constants.Y_MIN + constants.WORKSPACE_BOUNDS_BUFFER)
    xmax, ymax = (constants.X_MAX - constants.WORKSPACE_BOUNDS_BUFFER,
                  constants.Y_MAX - constants.WORKSPACE_BOUNDS_BUFFER)

    rotation = transform.Rotation.from_rotvec([0, math.pi, 0])

    # Sample random robot start pose.
    robot_translation = self._rng.uniform(
        low=[xmin, ymin, constants.EFFECTOR_HEIGHT],
        high=[xmax, ymax, constants.EFFECTOR_HEIGHT])
    robot_translation[2] = constants.EFFECTOR_HEIGHT
    starting_pose = Pose3d(rotation=rotation, translation=robot_translation)
    self._set_robot_target_effector_pose(starting_pose)
    # Step simulation to move arm in place.
    self._step_simulation_to_stabilize()

    # workspace bounds are
    # low=(0.15, -0.5), high=(0.7, 0.5)
    # or
    # low=(0.15, -0.3048), high=(0.6, 0.3048)
    def _generate_pose():
      block_translation = self._rng.uniform(
          low=[xmin, ymin, 0.0], high=[xmax, ymax, 0.0])
      block_sampled_angle = self._rng.uniform(low=0.0, high=2 * np.pi)
      block_rotation = transform.Rotation.from_rotvec(
          [np.pi / 2, 0, block_sampled_angle])
      block_rotation = self._pybullet_client.getQuaternionFromEuler(
          [np.pi / 2, 0, block_sampled_angle])
      return block_translation, block_rotation

    # Reset block poses. Sample random poses and reject those that might
    # result in intersection between blocks or with the robot end effector.
    obj_translations = [self._target_effector_pose.translation]
    obj_translations = []
    num_reward_attempts = 0  # 20 tries to find a valid reward.
    max_num_reward_attempts = 20

    # Get the list of "visible" blocks and "invisible" blocks on table.
    visible_block_ids = [self._block_to_pybullet_id[i] for i in blocks_on_table]
    invisible_block_ids = [
        i for i in self._block_ids if i not in visible_block_ids
    ]

    # Cast the "invisible" blocks way off the board.
    for block_id in invisible_block_ids:
      far_translation = np.array([5., 5., 0])
      far_rotation = transform.Rotation.from_rotvec([np.pi / 2, 0, 0])
      self._pybullet_client.resetBasePositionAndOrientation(
          block_id, far_translation.tolist(),
          far_rotation.as_quat().tolist())

    # Next find locations for the "visible" blocks on the board.
    while True:
      max_num_attempts = 20  # 20 tries to find a good pose for each block.
      for block_id in visible_block_ids:
        num_attempts = 0
        while True:
          candidate_translation, candidate_rotation = _generate_pose()
          if not obj_translations or min([
              np.linalg.norm(candidate_translation - p)
              for p in obj_translations
          ]) > constants.BLOCK_DISTANCE_THRESHOLD:
            if (np.linalg.norm(candidate_translation[:2] -
                               self._target_effector_pose.translation[:2]) >
                constants.ARM_DISTANCE_THRESHOLD):
              # Use a higher distance from the arm to avoid
              # collisions between end effector and blocks on initialization.
              obj_translations.append(candidate_translation)
              self._pybullet_client.resetBasePositionAndOrientation(
                  block_id, candidate_translation.tolist(), candidate_rotation)
              break
          num_attempts += 1
          if num_attempts > max_num_attempts:
            raise ValueError('Exceeded max attempts for generating block pose.')

      self._step_simulation_to_stabilize(nsteps=200)

      if self._reward_calculator is not None:
        info = self._reward_calculator.reset(
            self._compute_state(request_task_update=False),
            # Define an instruction over just
            # these blocks.
            blocks_on_table=blocks_on_table)
        if info == task_info.FAILURE:
          # Try again with a new configuration of blocks.
          continue
        self._set_task_info(info)
        num_reward_attempts += 1
        if self._instruction_str is None:
          if num_reward_attempts >= max_num_reward_attempts:
            raise ValueError('Cannot find a block config with valid reward.')
          continue  # Try again with a new configuration of blocks.
      break

  def _construct_invisible_walls(self):
    walls = []

    mass = 0
    visible = False
    if visible:
      vis_shape_id = -1
    else:
      vis_shape_id = pybullet.createVisualShape(
          shapeType=pybullet.GEOM_SPHERE,
          radius=1e-6,
          rgbaColor=[1, 1, 1, 0.])

    extent_width = 0.04
    extent_height = 0.1

    distance_x = (constants.X_MAX - constants.X_MIN) / 2.0
    distance_y = (constants.Y_MAX - constants.Y_MIN) / 2.0
    buffer = -0.01

    # RIGHT
    box = pybullet.createCollisionShape(
        pybullet.GEOM_BOX,
        halfExtents=[distance_x, extent_width, extent_height])
    position = [constants.CENTER_X, constants.Y_MAX - buffer, 0]
    wall = pybullet.createMultiBody(
        mass, box, vis_shape_id, position, useMaximalCoordinates=0)
    walls.append(wall)

    # LEFT
    box = pybullet.createCollisionShape(
        pybullet.GEOM_BOX,
        halfExtents=[distance_x, extent_width, extent_height])
    position = [constants.CENTER_X, constants.Y_MIN + buffer, 0]
    wall = pybullet.createMultiBody(
        mass, box, vis_shape_id, position, useMaximalCoordinates=0)
    walls.append(wall)

    # TOP
    box = pybullet.createCollisionShape(
        pybullet.GEOM_BOX,
        halfExtents=[extent_width, distance_y, extent_height])
    position = [constants.X_MIN + buffer - 0.01, constants.CENTER_Y, 0]
    wall = pybullet.createMultiBody(
        mass, box, vis_shape_id, position, useMaximalCoordinates=0)
    walls.append(wall)

    # BOTTOM
    box = pybullet.createCollisionShape(
        pybullet.GEOM_BOX,
        halfExtents=[extent_width, distance_y, extent_height])
    position = [constants.X_MAX - buffer - 0.02, constants.CENTER_Y, 0]
    wall = pybullet.createMultiBody(
        mass, box, vis_shape_id, position, useMaximalCoordinates=0)
    walls.append(wall)

    # This turns off collision between walls and blocks
    # for block_id in self._block_ids:
    #   pybullet.setCollisionFilterPair(block_id, sphereUid, -1, -1, 0)

    # This turns off collision between walls and end-effector
    for wall in walls:
      for i in range(10):  # iterate over links of end-effector
        pybullet.setCollisionFilterPair(self._robot.end_effector,
                                        wall, i-1, -1, 0)


def add_debug_info_to_image(image,
                            info_dict,
                            pos=(0, 0),
                            font=cv2.FONT_HERSHEY_DUPLEX,
                            font_scale=1,
                            font_thickness=1,
                            text_color=(0, 0, 0)):
  """Draw debugging text."""
  # Increase the image size so that debug text fits.
  image = cv2.resize(image, (640, 360))
  whitespace_per_line = 0.08
  if 'instruction' in info_dict:
    formatted_text = 'instruction: %s' % info_dict['instruction']
  else:
    formatted_text = ''
  wrapped_text = textwrap.wrap(formatted_text, width=35)
  whitespace_height = int(3 * int(image.shape[0] * whitespace_per_line))
  # Add whitespace below image.
  whitespace = np.ones([whitespace_height, image.shape[1], 3],
                       dtype=np.uint8) * 255
  image = np.concatenate([whitespace, image], 0)
  x, y = pos
  wrapped_text = textwrap.wrap(formatted_text, width=35)
  for _, line in enumerate(wrapped_text):
    text_size, _ = cv2.getTextSize(line, font, font_scale, font_thickness)
    _, text_h = text_size
    cv2.putText(image, line, (x, y + text_h + font_scale - 1), font, font_scale,
                text_color, font_thickness)
    y += int(text_h * 1.2)
  return image


def sleep_spin(sleep_time_sec):
  """Spin wait sleep. Avoids time.sleep accuracy issues on Windows."""
  if sleep_time_sec <= 0:
    return
  t0 = time.perf_counter()
  while time.perf_counter() - t0 < sleep_time_sec:
    pass
