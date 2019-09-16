# Copyright 2019 The TensorFlow Probability Authors.
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
"""Utilities for hypothesis testing of psd_kernels."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from hypothesis.extra import numpy as hpnp
import hypothesis.strategies as hps
import numpy as np


@hps.composite
def kernel_input(
    draw,
    batch_shape,
    example_dim=None,
    example_ndims=None,
    feature_dim=None,
    feature_ndims=None):
  """Strategy for drawing arbitrary Kernel input.

  Args:
    draw: Hypothesis function supplied by `@hps.composite`.
    batch_shape: An optional `TensorShape`.  The batch shape of the resulting
      kernel input.  Hypothesis will pick a batch shape if omitted.
    example_dim: Optional Python int giving the size of each example dimension.
      If omitted, Hypothesis will choose one.
    example_ndims: Optional Python int giving the number of example dimensions
      of the input. If omitted, Hypothesis will choose one.
    feature_dim: Optional Python int giving the size of each feature dimension.
      If omitted, Hypothesis will choose one.
    feature_ndims: Optional Python int stating the number of feature dimensions
      inputs will have. If omitted, Hypothesis will choose one.
  Returns:
    kernel_input: A strategy for drawing kernel_input with the prescribed shape
      (or an arbitrary one if omitted).
  """
  if example_ndims is None:
    example_ndims = draw(hps.integers(min_value=1, max_value=4))
  if example_dim is None:
    example_dim = draw(hps.integers(min_value=2, max_value=6))

  if feature_ndims is None:
    feature_ndims = draw(hps.integers(min_value=1, max_value=4))
  if feature_dim is None:
    feature_dim = draw(hps.integers(min_value=2, max_value=6))

  input_shape = batch_shape
  input_shape += [example_dim] * example_ndims
  input_shape += [feature_dim] * feature_ndims
  # We would like kernel inputs to be unique. This is to avoid computing kernel
  # matrices that are semi-definite.
  return draw(hpnp.arrays(
      dtype=np.float32,
      shape=input_shape,
      elements=hps.floats(
          -100, 100, allow_nan=False, allow_infinity=False),
      unique=True))
