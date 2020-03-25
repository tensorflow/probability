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
"""Constant kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.psd_kernels.internal import util
from tensorflow_probability.python.math.psd_kernels.positive_semidefinite_kernel import PositiveSemidefiniteKernel

__all__ = ["Constant"]

class Constant(PositiveSemidefiniteKernel):
  """docs
  """

  def __init__(self, coef=None, feature_ndims=1, validate_args=False, name="Constant"):
    """docs
    """
    parameters = dict(locals())
    with tf.name_scope(name):
      dtype = util.maybe_get_common_dtype([coef])
      self._coef = tensor_util.convert_nonref_to_tensor(coef, dtype=dtype, name="coef")
    super(Constant, self).__init__(
      feature_ndims,
      dtype=dtype,
      name=name,
      validate_args=validate_args,
      parameters=parameters)

  def _apply(self, x1, x2, example_ndims=0):
    shape = tf.broadcast_dynamic_shape(x1[:-(example_ndims + self.feature_ndims)],
                                       x2[:-(example_ndims + self.feature_ndims)])
    expected = tf.ones(shape, dtype=self._dtype)
    if self.coef is not None:
      coef = tf.convert_to_tensor(self._coef)
      expected = coef * expected
    return expected

  @property
  def coef(self):
    return self._coef

  def _batch_shape(self):
    scalar_shape = tf.TensorShape([])
    return scalar_shape if self.coef is None else tf.shape(self.coef)
  
  def _batch_shape_tensor(self):
    return tf.TensorShape([]) if self.coef is None else self.coef.shape
  
  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    for arg_name, arg in dict(coef=self.coef).items():
      if arg is not None and is_init != tensor_util.is_ref(arg):
        assertions.append(assert_util.assert_positive(
          arg,
          message='{} must be positive.'.format(arg_name)))
    return assertions
