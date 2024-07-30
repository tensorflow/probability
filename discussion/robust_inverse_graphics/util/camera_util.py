# Copyright 2024 The TensorFlow Probability Authors.
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
# ============================================================================
"""Camera utilities."""

from typing import Optional

import numpy as np
import pyquaternion


def look_at_quat(
    position: np.ndarray,
    target: np.ndarray,
    up: np.ndarray = np.array([0., 1., 0.]),
    front: np.ndarray = np.array([0., 0., -1.]),
    quaternion_atol: float = 1e-8,
    quaternion_rtol: float = 1e-5
) -> tuple[float, float, float, float]:
  """Constructs a quaternion looking at `target` from `position`.

  Args:
    position: Camera position. Shape: [3]
    target: Camera target. Shape: [3]
    up: World up unit vector. Shape: [3]
    front: World front unit vector. Shape: [3]
    quaternion_atol: atol for pyquaternion matrix orthogonality checks.
    quaternion_rtol: rtol for pyquaternion matrix orthogonality checks.

  Returns:
    Quaternion as a 4-tuple.
  """

  right = np.cross(up, front)

  normalize = lambda x: x / (np.linalg.norm(x, axis=-1) + 1e-20)

  look_at_front = normalize(target - position)
  look_at_right = normalize(np.cross(up, look_at_front))
  if np.linalg.norm(look_at_right, axis=-1) == 0.:
    look_at_right = right

  look_at_up = normalize(np.cross(look_at_front, look_at_right))

  rotation_matrix1 = np.stack([look_at_right, look_at_up, look_at_front])
  rotation_matrix2 = np.stack([right, up, front])

  return tuple(
      pyquaternion.Quaternion(matrix=(rotation_matrix1.T @ rotation_matrix2),
                              atol=quaternion_atol,
                              rtol=quaternion_rtol))


def random_sphere(rng: Optional[np.random.RandomState] = None) -> np.ndarray:
  """Generates points uniformly on a sphere."""
  if rng is None:
    rng = np.random
  z = rng.randn(3)
  z /= (np.linalg.norm(z) + 1e-20)
  return z


def random_half_sphere(
    half_elem: int = 1,
    rng: Optional[np.random.RandomState] = None) -> np.ndarray:
  """Generates points uniformly on a half-sphere."""
  z = random_sphere(rng)
  z[half_elem] = np.abs(z[half_elem])
  return z


def grid_sphere(num_slices: int) -> np.ndarray:
  """Generates points on a regular grid on a sphere.

  This places the poles at (0, +-1, 0).

  Args:
    num_slices: Number of slices. Should be even.

  Returns:
    The generated points. This will generate
    `2 + (num_slices // 2 - 1) * num_slices` points.
  """
  elevation = np.linspace(np.pi / 2, -np.pi / 2, num_slices // 2 + 1)
  azimuth = np.linspace(0.0, 2 * np.pi, num_slices + 1)[:num_slices]

  points = []
  for (
      band,
      el,
  ) in enumerate(elevation):
    if band == 0 or band == len(elevation) - 1:
      band_azimuth = [0.0]
    else:
      band_azimuth = azimuth

    for az in band_azimuth:
      r = np.cos(el)
      x = r * np.sin(az)
      z = r * np.cos(az)
      y = np.sin(el)
      points.append(np.array([x, y, z]))
  return np.array(points)


def get_mipnerf_camera_intrinsics(width: int,
                                  height: int,
                                  focal_length: float,
                                  sensor_width: float = 1.,
                                  sensor_height: float = 1.) -> np.ndarray:
  """Constructs the mipnerf-compatible intrinsics matrix."""
  # See https://en.wikipedia.org/wiki/Camera_resectioning#Intrinsic_parameters

  fx = focal_length / sensor_width * width
  fy = focal_length / sensor_height * height

  return np.array([
      [fx, 0., width / 2.],
      [0., fy, height / 2.],
      [0., 0., 1.],
  ], np.float32)


def get_camera_position(radius: float, inclination: float,
                        azimuth: float) -> np.ndarray:
  """Converts radius, inclination, azimuth to xyz.

  Uses this convention
  https://en.wikipedia.org/wiki/Spherical_coordinate_system#/media/File:3D_Spherical.svg

  Args:
    radius: float, how far is the camera from the center.
    inclination: float, in radians, theta in the above.
    azimuth: float, in radians, phi (LaTeX `varphi`) in the above.

  Returns:
    camera_position: Shape [3], xyz position.
  """
  return np.array([
      radius * np.cos(azimuth) * np.sin(inclination),
      radius * np.sin(azimuth) * np.sin(inclination),
      radius * np.cos(inclination)
  ])
