# Copyright 2021 The TensorFlow Probability Authors.
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
"""Exponential of another kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.math.psd_kernels import positive_semidefinite_kernel as psd_kernel

__all__ = ['PointwiseExponential']


class PointwiseExponential(psd_kernel.AutoCompositeTensorPsdKernel):
  """Pointwise exponential of a positive semi-definite kernel.

  Produces `exp(k(x, y))`, where `k(., .)` is the input positive semi-definite
  kernel.

  #### References
  [1]: John Shawe-Taylor and Nello Cristianini. Kernel Methods for Pattern
  Analysis. Section 3.2. 2004.
  https://people.eecs.berkeley.edu/~jordan/kernels/0521813972c03_p47-84.pdf
  """

  def __init__(self, kernel, validate_args=False, name='Exponential'):
    """Construct pointwise exponential of a input kernel.

    Args:
      kernel: Tensorflow Probability PSD kernel instance.
      validate_args: If `True`, parameters are checked for validity despite
        possibly degrading runtime performance
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())

    with tf.name_scope(name) as name:
      self._kernel = kernel
      super(PointwiseExponential, self).__init__(
          feature_ndims=kernel.feature_ndims,
          dtype=kernel.dtype,
          validate_args=validate_args,
          name=name,
          parameters=parameters)

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict(kernel=parameter_properties.BatchedComponentProperties())

  def _apply(self, x1, x2, example_ndims):
    kernel_output = self._kernel.apply(x1, x2, example_ndims)
    exponential_output = tf.exp(kernel_output)
    return exponential_output

  def _matrix(self, x1, x2):
    kernel_matrix = self._kernel.matrix(x1, x2)
    return tf.exp(kernel_matrix)

  @property
  def kernel(self):
    return self._kernel
