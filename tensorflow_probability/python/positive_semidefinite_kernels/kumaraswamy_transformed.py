# Copyright 2018 The TensorFlow Probability Authors.
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
"""Kumaraswamy CDF transformed kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_probability.python import bijectors
from tensorflow_probability.python.positive_semidefinite_kernels import feature_transformed
from tensorflow_probability.python.positive_semidefinite_kernels.internal import util

__all__ = ['KumaraswamyTransformed']


class KumaraswamyTransformed(feature_transformed.FeatureTransformed):
  """Transform inputs by Kumaraswamy bijector.

  Uses Kumaraswamy bijector to warp features. The purpose of this is to is turn
  a stationary kernel in to a non-stationary one. Note that this kernel requires
  inputs to be in `[0, 1]` (the domain of the bijector).
  """

  def __init__(self,
               kernel,
               concentration1,
               concentration0,
               validate_args=False,
               name='KumaraswamyTransformedKernel'):
    """Construct a KumaraswamtTransformed kernel instance.

    This kernel wraps a kernel `k`, pre-transforming inputs with an
    Kumaraswamy bijector's `inverse` function before passign off to `k`.

    Args:
      kernel: The `PositiveSemidefiniteKernel` instance to transform.
      concentration1: forwarded to the concentration1 parameter of the
        Kumaraswamy bijector.
      concentration0: forwarded to the concentration0 parameter of the
        Kumaraswamy bijector.
      validate_args: forwarded to the `validate_args` parameter of the
        Kumaraswamy bijector.
      name: Python `str` name given to ops managed by this object.
    """
    self._concentration1 = concentration1
    self._concentration0 = concentration0

    def transform_by_kumaraswamy(x, feature_ndims, example_ndims):
      """Apply a Kumaraswamy bijector to features."""
      concentration1 = util.pad_shape_with_ones(
          self.concentration1,
          example_ndims,
          start=-(feature_ndims + 1))
      concentration0 = util.pad_shape_with_ones(
          self.concentration0,
          example_ndims,
          start=-(feature_ndims + 1))
      bij = bijectors.Kumaraswamy(
          concentration1, concentration0, validate_args=validate_args)
      # Apply the inverse as this is the Kumaraswamy CDF.
      return bij.inverse(x)

    super(KumaraswamyTransformed, self).__init__(
        kernel, transformation_fn=transform_by_kumaraswamy)

  @property
  def concentration1(self):
    return self._concentration1

  @property
  def concentration0(self):
    return self._concentration0
