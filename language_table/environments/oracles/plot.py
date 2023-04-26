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

"""Utils for plotting RRT and writing debug images."""

import imageio
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf


DEBUG_IMAGE_PATH = '/tmp/rrt_debug_img.png'


class PlotRRT:
  """Plotting helper for visualizing RRT."""

  def __init__(self,
               x_start,
               x_goal,
               obs_boundary,
               obs_circle,
               image_path=DEBUG_IMAGE_PATH):
    self._x_start, self._x_goal = x_start, x_goal
    self._obs_bound = obs_boundary
    self._obs_circle = obs_circle
    self._image_path = image_path

  def animation(self, nodelist, path, name, animation=False):
    self.plot_grid(name)
    self.plot_visited(nodelist, animation)
    self.plot_path(path)

  def plot_grid(self, name):
    """Plot whole grid."""
    self.fig, ax = plt.subplots()
    for (ox, oy, w, h) in self._obs_bound:
      ax.add_patch(
          patches.Rectangle(
              (ox, oy), w, h,
              edgecolor='black',
              facecolor='black',
              fill=True)
      )
    for (ox, oy, r) in self._obs_circle:
      ax.add_patch(
          patches.Circle(
              (ox, oy), r,
              edgecolor='black',
              facecolor='gray',
              fill=True)
      )

    radius = self._obs_circle[0][2]
    ax.add_patch(
        patches.Circle(
            (self._x_start[0], self._x_start[1]), radius,
            edgecolor='black',
            facecolor='blue',
            fill=True)
        )
    ax.add_patch(
        patches.Circle(
            (self._x_goal[0], self._x_goal[1]), radius,
            edgecolor='black',
            facecolor='green',
            fill=True)
        )
    plt.title(name)
    plt.axis('equal')

  def plot_visited(self, nodelist, animation):
    """Plot visited."""
    if animation:
      count = 0
      for node in nodelist:
        count += 1
        if node.parent:
          plt.plot([node.parent.x, node.x], [node.parent.y, node.y], '-g')
    else:
      for node in nodelist:
        if node.parent:
          plt.plot([node.parent.x, node.x], [node.parent.y, node.y], '-g')

  def plot_path(self, path):
    if path:
      plt.plot([x[0] for x in path], [x[1] for x in path], '-r', linewidth=2)
    plt.show()

  def fig_to_array(self, figure):
    """Converts a matplotlib figure to a numpy array."""
    figure.canvas.draw()
    np_fig = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    np_fig = np_fig.reshape(figure.canvas.get_width_height()[::-1] + (3,))
    return np_fig

  def save_debug_image(self):
    """Writes an image to a debug path on cns."""
    array = self.fig_to_array(self.fig)
    with tf.io.gfile.GFile(self._image_path, 'wb') as f:
      imageio.imwrite(f, array * 255.0, format='png')
