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

"""Data classes holding info returned to environment for each reset."""

import dataclasses
from typing import List

import numpy as np


@dataclasses.dataclass
class Block2BlockTaskInfo:
  """Data class defining a chosen block2block task after reset."""
  instruction: str
  block1: str
  block2: str


@dataclasses.dataclass
class Block2LocationTaskInfo:
  """Class defining a chosen block2block task after reset."""
  instruction: str
  block: str
  target_translation: np.ndarray
  location: str


@dataclasses.dataclass
class Block2LineTaskInfo:
  """Class defining a chosen block2block task after reset."""
  instruction: str
  block: str
  target_translation: np.ndarray


@dataclasses.dataclass
class Block2PoleTaskInfo:
  """Data class defining a chosen block2pole task after reset."""
  instruction: str
  block1: str
  goal: str


@dataclasses.dataclass
class Block2RelativeLocationTaskInfo:
  """Class defining a chosen block2block task after reset."""
  instruction: str
  block: str
  target_translation: np.ndarray
  location: str


@dataclasses.dataclass
class Block2BlockRelativeLocationTaskInfo:
  """Class defining a chosen block2block task after reset."""
  instruction: str
  block: str
  target_block: str
  direction: str
  target_translation: np.ndarray


@dataclasses.dataclass
class SeparateBlocksTaskInfo:
  """Class defining a chosen "separate blocks" task after reset."""
  instruction: str
  block: str
  avoid_blocks: List[str]
  target_translation: np.ndarray


@dataclasses.dataclass
class Point2BlockTaskInfo:
  """Data class defining a chosen point2block task after reset."""
  instruction: str
  block_target: str


ALL_TASKS = [
    Block2BlockTaskInfo,
    Block2LocationTaskInfo,
    Block2RelativeLocationTaskInfo,
    Block2BlockRelativeLocationTaskInfo,
    SeparateBlocksTaskInfo,
    Point2BlockTaskInfo,
    Block2LineTaskInfo,
    Block2PoleTaskInfo,
]

# Return this if cannot create a valid board state and need to reset.
FAILURE = 'failure'
