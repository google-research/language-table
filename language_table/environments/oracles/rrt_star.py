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

"""RRT squared 2D motion planner."""
import collections
import math

from absl import logging
from language_table.environments.oracles import plot as plot_module
import numpy as np


def get_shortest_path_no_collisions(xy_start,  # (x,y) of moving object.
                                    xy_goal,  # (x,y) of target location.
                                    x_range,  # (xmin, xmax).
                                    y_range,  # (ymin, ymax).
                                    obstacle_xy,  # [(x,y)] of other objects.
                                    obstacle_widths,  # [w], each ones diameter.
                                    delta,  # tune this.
                                    step_length,  # tune this.
                                    goal_sample_rate,  # tune this.
                                    search_radius,  # tune this.
                                    iter_max,  # tune this.
                                    boundary_width=0.01,  # get from env.
                                    plot=False,
                                    raise_error_on_plan_failure=False):
  """Uses RRT-* to get the shortest path without collisions."""
  # Create list of (x, y, w) "circle" obstacles.
  obstacles = []
  for (x, y), width in zip(obstacle_xy, obstacle_widths):
    obstacles.append((x, y, width))

  # Create a list of "boundary" rectangles (x, y, w, h).
  x_min, x_max = x_range
  y_min, y_max = y_range
  width_x = (x_max - x_min)
  height_y = (y_max - y_min)
  boundary = [
      [x_min, y_min, boundary_width, height_y],
      [x_max, y_min, boundary_width, height_y],
      [x_min, y_min, width_x, boundary_width],
      [x_min, y_max, width_x, boundary_width]
  ]

  # Create the RRT-* planner.
  planner = RRTStar(xy_start,
                    xy_goal,
                    obstacles,
                    boundary,
                    x_range,
                    y_range,
                    delta,
                    step_length,
                    goal_sample_rate,
                    search_radius,
                    iter_max,
                    plot)
  # Plan.
  planner.plan()

  # Return shortest path.
  if not planner.success:
    message = 'Could not find path! Consider retuning RRT-*.'
    if raise_error_on_plan_failure:
      raise ValueError(message)
    else:
      # The compromise behavior here is to just return the direct path between
      # start and goal. This assumes that we will "just try it" for a few
      # seconds then replan again in the future.
      logging.info(message)
      direct_path = [xy_goal, xy_start]
      return direct_path, planner.success
  return planner.path, planner.success


class RRTStar:
  """RRT-* planning algorithm."""

  def __init__(self,
               x_start,
               x_goal,
               obstacles,
               boundary,
               x_range,
               y_range,
               delta,
               step_len,
               goal_sample_rate,
               search_radius,
               iter_max,
               plot_debug_image=True):
    self.s_start = Node(x_start)
    self.s_goal = Node(x_goal)
    self.obstacles = obstacles
    self.boundary = boundary
    self.delta = delta
    self.step_len = step_len
    self.goal_sample_rate = goal_sample_rate
    self.search_radius = search_radius
    self.iter_max = iter_max
    self.vertex = [self.s_start]
    self.path = []
    self.x_range = x_range
    self.y_range = y_range
    self.plot_debug_image = plot_debug_image
    if plot_debug_image:
      self.plotting = plot_module.PlotRRT(
          x_start, x_goal, boundary, obstacles)

  def plan(self):
    """Plans a collision-free path from x_start to x_goal."""
    # First check if "source block" in collision w/ others.
    if inside_obstacles(self.s_start, self.delta, self.obstacles):
      logging.info('RRT-* failed to find a solution because x_start '
                   'was initialized inside another obstacle. Consider retuning '
                   'delta.')
      self.path = []
      self.success = False
      return

    for _ in range(self.iter_max):
      node_rand = self.get_random_node(self.goal_sample_rate)
      node_near = self.nearest_neighbor(self.vertex, node_rand)
      node_new = self.new_state(node_near, node_rand)

      if node_new and not is_collision(
          node_near, node_new,
          self.delta, self.obstacles, self.boundary):
        neighbor_index = self.find_nearest_neighbors(node_new)
        self.vertex.append(node_new)

        if neighbor_index:
          self.choose_parent(node_new, neighbor_index)
          self.rewire(node_new, neighbor_index)

    index = self.search_goal_parent()
    if index is None:
      logging.info('RRT-* failed to find a solution.')
      self.path = []
      self.success = False
      return

    self.path = self.extract_path(self.vertex[index])

    if self.plot_debug_image:
      self.plotting.animation(
          self.vertex, self.path, 'rrt*, N = %d' % self.iter_max)
      self.plotting.save_debug_image()
    self.success = True

  def new_state(self, node_start, node_goal):
    dist, theta = self.get_distance_and_angle(node_start, node_goal)

    dist = min(self.step_len, dist)
    node_new = Node((node_start.x + dist * math.cos(theta),
                     node_start.y + dist * math.sin(theta)))

    node_new.parent = node_start

    return node_new

  def choose_parent(self, node_new, neighbor_index):
    cost = [self.get_new_cost(self.vertex[i], node_new) for i in neighbor_index]

    cost_min_index = neighbor_index[int(np.argmin(cost))]
    node_new.parent = self.vertex[cost_min_index]

  def rewire(self, node_new, neighbor_index):
    for i in neighbor_index:
      node_neighbor = self.vertex[i]

      if self.cost(node_neighbor) > self.get_new_cost(node_new, node_neighbor):
        node_neighbor.parent = node_new

  def search_goal_parent(self):
    """Searches for a goal."""
    dist_list = [get_dist(n, self.s_goal) for n in self.vertex]
    node_index = [i for i in range(len(dist_list))
                  if dist_list[i] <= self.step_len]
    if node_index:
      # pylint: disable=g-complex-comprehension
      cost_list = [
          dist_list[i] + self.cost(self.vertex[i])
          for i in node_index
          if not is_collision(self.vertex[i], self.s_goal, self.delta,
                              self.obstacles, self.boundary)
      ]
      # pylint: enable=g-complex-comprehension
      if not cost_list:
        # Graph search failed.
        return None
      return node_index[int(np.argmin(cost_list))]
    return len(self.vertex) - 1

  def get_new_cost(self, node_start, node_end):
    dist, _ = self.get_distance_and_angle(node_start, node_end)

    return self.cost(node_start) + dist

  def get_random_node(self, goal_sample_rate):
    if np.random.random() > goal_sample_rate:
      return Node((np.random.uniform(self.x_range[0] + self.delta,
                                     self.x_range[1] - self.delta),
                   np.random.uniform(self.y_range[0] + self.delta,
                                     self.y_range[1] - self.delta)))

    return self.s_goal

  def find_nearest_neighbors(self, node_new):
    """Get nearest neighbors for new node."""
    n = len(self.vertex) + 1
    r = min(self.search_radius * math.sqrt((math.log(n) / n)), self.step_len)
    dist_table = [get_dist(nd, node_new) for nd in self.vertex]
    # pylint: disable=g-complex-comprehension
    dist_table_index = [ind for ind in range(len(dist_table)) if
                        dist_table[ind] <= r and not is_collision(
                            node_new, self.vertex[ind],
                            self.delta, self.obstacles, self.boundary)]
    # pylint: enable=g-complex-comprehension
    return dist_table_index

  @staticmethod
  def nearest_neighbor(node_list, n):
    return node_list[int(np.argmin([get_dist(nd, n) for nd in node_list]))]

  @staticmethod
  def cost(node_p):
    node = node_p
    cost = 0.0

    while node.parent:
      cost += get_dist(node, node.parent)
      node = node.parent
    return cost

  def update_cost(self, parent_node):
    queue = FIFOQueue()
    queue.put(parent_node)
    while not queue.empty():
      node = queue.get()
      if not node.child:
        continue
      for node_c in node.child:
        node_c.Cost = self.get_new_cost(node, node_c)
        queue.put(node_c)

  def extract_path(self, node_end):
    path = [[self.s_goal.x, self.s_goal.y]]
    node = node_end
    while node.parent is not None:
      path.append([node.x, node.y])
      node = node.parent
    path.append([node.x, node.y])
    return path

  @staticmethod
  def get_distance_and_angle(node_start, node_end):
    dx = node_end.x - node_start.x
    dy = node_end.y - node_start.y
    return math.hypot(dx, dy), math.atan2(dy, dx)


def is_collision(start, end, delta, obstacles, obs_boundary):
  """Check if ray between start and end leads to collision."""
  start_in_obs = is_inside_obstacles_or_boundary(
      start, delta, obstacles, obs_boundary)
  end_in_obs = is_inside_obstacles_or_boundary(
      end, delta, obstacles, obs_boundary)
  if start_in_obs or end_in_obs:
    return True
  orig, direc = get_ray(start, end)
  for (x, y, rad) in obstacles:
    if is_intersect_circle(orig, direc, [x, y], rad, delta):
      return True
  return False


def is_inside_obstacles_or_boundary(node, delta, obstacles, obs_boundary):
  if inside_obstacles(node, delta, obstacles):
    return True
  if inside_boundary(node, delta, obs_boundary):
    return True
  return False


def inside_obstacles(node, delta, obstacles):
  for (x, y, r) in obstacles:
    if math.hypot(node.x - x, node.y - y) <= r + delta:
      return True
  return False


def inside_boundary(node, delta, obs_boundary):
  for (x, y, w, h) in obs_boundary:
    if (0 <= node.x - (x - delta) <= w + 2 * delta
        and 0 <= node.y - (y - delta) <= h + 2 * delta):
      return True
  return False


def is_intersect_circle(o, d, a, r, delta):
  d2 = np.dot(d, d)
  if d2 == 0:
    return False
  t = np.dot([a[0] - o[0], a[1] - o[1]], d) / d2
  if 0 <= t <= 1:
    shot = Node((o[0] + t * d[0], o[1] + t * d[1]))
    if get_dist(shot, Node(a)) <= r + delta:
      return True
  return False


def get_ray(start, end):
  orig = [start.x, start.y]
  direc = [end.x - start.x, end.y - start.y]
  return orig, direc


def get_dist(start, end):
  return math.hypot(end.x - start.x, end.y - start.y)


class Node:
  """Node in graph."""

  def __init__(self, n):
    self.x = n[0]
    self.y = n[1]
    self.parent = None


class FIFOQueue:
  """First in, first out queue."""

  def __init__(self):
    self.queue = collections.deque()

  def empty(self):
    return False if self.queue else True

  def put(self, node):
    self.queue.append(node)

  def get(self):
    return self.queue.popleft()
