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

"""Shared language table constants."""

import math
import numpy as np
from scipy.spatial import transform

PLANE_URDF_PATH = ('third_party/bullet/examples/pybullet/gym/pybullet_data/'
                   'plane.urdf')

EFFECTOR_HEIGHT = 0.145
EFFECTOR_DOWN_ROTATION = transform.Rotation.from_rotvec([0, math.pi, 0])

X_MIN = 0.15
X_MAX = 0.6
Y_MIN = -0.3048
Y_MAX = 0.3048
CENTER_X = (X_MAX - X_MIN) / 2. + X_MIN
CENTER_Y = (Y_MAX - Y_MIN) / 2. + Y_MIN

WORKSPACE_BOUNDS_BUFFER = 0.08

BLOCK_DISTANCE_THRESHOLD = 0.0175
ARM_DISTANCE_THRESHOLD = 0.06
INSTRUCTION_LENGTH = 512  # max number of chars in instruction

WORKSPACE_BOUNDS = np.array(((X_MIN, Y_MIN), (X_MAX, Y_MAX)))
WORKSPACE_URDF_PATH = 'third_party/py/language_table/environments/assets/workspace_real.urdf'
CAMERA_POSE = (0.75, 0, 0.5)
CAMERA_ORIENTATION = (np.pi / 5, np.pi, -np.pi / 2)

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 180
CAMERA_INTRINSICS = (
    0.803 * IMAGE_WIDTH,  # fx
    0,
    IMAGE_WIDTH / 2.,  # cx
    0,
    0.803 * IMAGE_WIDTH,  # fy
    IMAGE_HEIGHT / 2.,  # cy
    0,
    0,
    1)

# Corresponds to:
# rotation = transform.Rotation.from_rotvec([0, math.pi, 0])
# translation = np.array([0.3, -0.2, 0.145])
INITIAL_JOINT_POSITIONS = np.array([
    -0.5875016909413221, 0.15985553866983415, -0.4992862770497537,
    0.0017427885915130214, 0.33927183830553914, -3.7249551487437524
])
