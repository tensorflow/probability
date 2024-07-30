# Copyright 2022 The TensorFlow Probability Authors.
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
"""Positive-Semidefinite Kernel test utilities."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.math.psd_kernels import positive_semidefinite_kernel


class MultipartTestKernel(
    positive_semidefinite_kernel.AutoCompositeTensorPsdKernel):
  """A kernel that takes nested structures as inputs.

  The inputs are flattened and concatenated, and then passed to the base kernel.
  """

  def __init__(self, kernel, feature_ndims=None):
    parameters = dict(locals())
    self._kernel = kernel
    if feature_ndims is None:
      feature_ndims = {'foo': kernel.feature_ndims, 'bar': kernel.feature_ndims}
    super(MultipartTestKernel, self).__init__(
        feature_ndims=feature_ndims,
        dtype={'foo': tf.float32, 'bar': tf.float32},
        parameters=parameters,
        validate_args=kernel.validate_args)

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict(kernel=parameter_properties.BatchedComponentProperties())

  @property
  def kernel(self):
    return self._kernel

  def _apply(self, x1, x2, example_ndims=0):
    def _flatten_and_concat(x):
      flat = tf.nest.flatten(
          tf.nest.map_structure(
              lambda t, nd: tf.reshape(  # pylint: disable=g-long-lambda
                  t, ps.concat([ps.shape(t)[:-nd], [-1]], axis=0)),
              x, self.feature_ndims))
      # Broadcast shapes of flattened components together.
      broadcasted_flat = [flat[0] + tf.zeros_like(flat[1][..., :1]),
                          flat[1] + tf.zeros_like(flat[0][..., :1])]
      return tf.concat(broadcasted_flat, axis=-1)
    x1 = _flatten_and_concat(x1)
    x2 = _flatten_and_concat(x2)
    return self.kernel._apply(x1, x2, example_ndims=example_ndims)  # pylint: disable=protected-access
