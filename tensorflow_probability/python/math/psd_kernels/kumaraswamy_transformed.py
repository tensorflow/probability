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

import functools

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.psd_kernels import feature_transformed
from tensorflow_probability.python.math.psd_kernels import positive_semidefinite_kernel as psd_kernel
from tensorflow_probability.python.math.psd_kernels.internal import util


__all__ = ['KumaraswamyTransformed']


@psd_kernel.auto_composite_tensor_psd_kernel
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
    parameters = dict(locals())

    # Delayed import to avoid circular dependency between `tfp.bijectors` and
    # `tfp.math`
    from tensorflow_probability.python.bijectors import kumaraswamy_cdf  # pylint: disable=g-import-not-at-top

    with tf.name_scope(name):
      self._concentration1 = tensor_util.convert_nonref_to_tensor(
          concentration1, name='concentration1')
      self._concentration0 = tensor_util.convert_nonref_to_tensor(
          concentration0, name='concentration0')

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
        bij = kumaraswamy_cdf.KumaraswamyCDF(concentration1,
                                             concentration0,
                                             validate_args=validate_args)
        return bij.forward(x)

      super(KumaraswamyTransformed, self).__init__(
          kernel,
          transformation_fn=transform_by_kumaraswamy,
          validate_args=validate_args,
          parameters=parameters)

  @property
  def concentration1(self):
    return self._concentration1

  @property
  def concentration0(self):
    return self._concentration0

  def _batch_shape(self):
    return functools.reduce(
        tf.broadcast_static_shape,
        [self.kernel.batch_shape,
         self.concentration1.shape[:-self.kernel.feature_ndims],
         self.concentration0.shape[:-self.kernel.feature_ndims]])

  def _batch_shape_tensor(self):
    return functools.reduce(
        tf.broadcast_dynamic_shape,
        [self.kernel.batch_shape_tensor(),
         tf.shape(self.concentration1)[:-self.kernel.feature_ndims],
         tf.shape(self.concentration0)[:-self.kernel.feature_ndims]])

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    for arg_name, arg in dict(
        concentration0=self.concentration0,
        concentration1=self.concentration1).items():
      if arg is not None and is_init != tensor_util.is_ref(arg):
        assertions.append(assert_util.assert_positive(
            arg,
            message='{} must be positive.'.format(arg_name)))
    return assertions
