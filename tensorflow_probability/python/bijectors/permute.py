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
"""Permutation bijectors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util


__all__ = [
    'Permute',
]


class Permute(bijector.Bijector):
  """Permutes the rightmost dimension of a `Tensor`.

  ```python
  reverse = tfp.bijectors.Permute(permutation=[2, 1, 0])

  reverse.forward([-1., 0., 1.])
  # ==> [1., 0., -1]

  reverse.inverse([1., 0., -1])
  # ==> [-1., 0., 1.]

  reverse.forward_log_det_jacobian(any_value)
  # ==> 0.

  reverse.inverse_log_det_jacobian(any_value)
  # ==> 0.
  ```

  Warning: `tf.estimator` may repeatedly build the graph thus
  `Permute(np.random.permutation(event_size)).astype('int32'))` is not a
  reliable parameterization (nor would it be even if using `tf.constant`). A
  safe alternative is to use `tf.get_variable` to achieve "init once" behavior,
  i.e.,

  ```python
  def init_once(x, name):
    return tf.get_variable(name, initializer=x, trainable=False)

  Permute(permutation=init_once(
      np.random.permutation(event_size).astype('int32'),
      name='permutation'))
  ```

  """

  def __init__(self, permutation, axis=-1, validate_args=False, name=None):
    """Creates the `Permute` bijector.

    Args:
      permutation: An `int`-like vector-shaped `Tensor` representing the
        permutation to apply to the `axis` dimension of the transformed
        `Tensor`.
      axis: Scalar `int` `Tensor` representing the dimension over which to
        `tf.gather`. `axis` must be relative to the end (reading left to right)
        thus must be negative.
        Default value: `-1` (i.e., right-most).
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str`, name given to ops managed by this object.

    Raises:
      TypeError: if `not dtype_util.is_integer(permutation.dtype)`.
      ValueError: if `permutation` does not contain exactly one of each of
        `{0, 1, ..., d}`.
      NotImplementedError: if `axis` is not known prior to graph execution.
      NotImplementedError: if `axis` is not negative.
    """
    parameters = dict(locals())
    with tf.name_scope(name or 'permute') as name:
      axis = tensor_util.convert_nonref_to_tensor(
          axis, name='axis', as_shape_tensor=True)
      if not dtype_util.is_integer(axis.dtype):
        raise TypeError('axis.dtype ({}) should be `int`-like.'.format(
            dtype_util.name(axis.dtype)))
      axis_ = tf.get_static_value(axis)
      if axis_ is None:
        raise NotImplementedError('`axis` must be known prior to graph '
                                  'execution.')
      elif axis_ >= 0:
        raise NotImplementedError('`axis` must be relative the rightmost '
                                  'dimension, i.e., negative.')
      forward_min_event_ndims = int(np.abs(axis_))
      self._axis = axis

      permutation = tensor_util.convert_nonref_to_tensor(
          permutation, name='permutation')
      if not dtype_util.is_integer(permutation.dtype):
        raise TypeError('permutation.dtype ({}) should be `int`-like.'.format(
            dtype_util.name(permutation.dtype)))
      self._permutation = permutation

      super(Permute, self).__init__(
          forward_min_event_ndims=forward_min_event_ndims,
          is_constant_jacobian=True,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @property
  def permutation(self):
    return self._permutation

  @property
  def axis(self):
    return self._axis

  def _forward(self, x):
    y = tf.gather(x, self.permutation, axis=self.axis)
    tensorshape_util.set_shape(y, x.shape)
    return y

  def _inverse(self, y):
    x = tf.gather(
        y, tf.math.invert_permutation(self.permutation), axis=self.axis)
    tensorshape_util.set_shape(x, y.shape)
    return x

  def _inverse_log_det_jacobian(self, y):
    # is_constant_jacobian = True for this bijector, hence the
    # `log_det_jacobian` need only be specified for a single input, as this will
    # be tiled to match `event_ndims`.
    return tf.constant(0., dtype=dtype_util.base_dtype(y.dtype))

  def _forward_log_det_jacobian(self, x):
    return tf.constant(0., dtype=dtype_util.base_dtype(x.dtype))

  def _parameter_control_dependencies(self, is_init):
    assertions = []

    if is_init != tensor_util.is_ref(self.permutation):
      if not dtype_util.is_integer(self.permutation.dtype):
        raise TypeError('permutation.dtype ({}) should be `int`-like.'.format(
            dtype_util.name(self.permutation.dtype)))

      p = tf.get_static_value(self.permutation)
      if p is not None:
        if set(p) != set(np.arange(p.size)):
          raise ValueError('Permutation over `d` must contain exactly one of '
                           'each of `{0, 1, ..., d}`.')

      if self.validate_args:
        p = tf.sort(self.permutation, axis=-1)
        assertions.append(
            assert_util.assert_equal(
                p,
                tf.range(tf.shape(p)[-1]),
                message=('Permutation over `d` must contain exactly one of '
                         'each of `{0, 1, ..., d}`.')))

    return assertions
