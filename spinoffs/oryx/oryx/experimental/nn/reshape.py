# Copyright 2020 The TensorFlow Probability Authors.
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
"""Contains layers that reshape arrays."""

import jax.numpy as np

from oryx.core import state
from oryx.experimental.nn import base

__all__ = [
    'Flatten',
    'Reshape',
]


class Flatten(base.Layer):
  """Flattens the inputs collapsing all ending dimensions."""

  @classmethod
  def initialize(cls, rng, in_spec):
    """Initializes Flatten Layer.

    Args:
      rng: Random key.
      in_spec: Input Spec.

    Returns:
      Tuple with the output shape and the LayerParams.
    """
    return base.LayerParams(info=in_spec.shape)

  @classmethod
  def spec(cls, in_spec):
    in_shape = in_spec.shape
    out_shape = (int(np.prod(in_shape)),)
    return state.Shape(out_shape, dtype=in_spec.dtype)

  def _call(self, x):
    """Applies Flatten Layer to the input x.

    Args:
      x: Input.

    Returns:
      Flattened input.
    """
    out_shape = Flatten.spec(x).shape
    return np.reshape(x, out_shape)


class Reshape(base.Layer):
  """Reshape the inputs to a new compatatible shape."""

  @classmethod
  def initialize(cls, rng, in_spec, dim_out):
    """Initializes Reshape Layer.

    Args:
      rng: Random key.
      in_spec: Input Spec.
      dim_out: Desired output dimensions.

    Returns:
      Tuple with the output shape and the LayerParams.
    """
    return base.LayerParams(info=(in_spec.shape, tuple(dim_out)))

  @classmethod
  def spec(cls, in_spec, dim_out):
    if isinstance(dim_out, int):
      dim_out = (dim_out,)
    else:
      dim_out = tuple(dim_out)
    return state.Shape(dim_out, dtype=in_spec.dtype)

  def _call(self, x):
    """Applies Reshape Layer to the input x.

    Args:
      x: Input.

    Returns:
      Flattened input.
    """
    _, dim_out = self.info
    out_shape = Reshape.spec(x, dim_out).shape
    return np.reshape(x, out_shape)
