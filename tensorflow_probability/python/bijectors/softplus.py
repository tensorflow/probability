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
"""Softplus bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math import generic as generic_math


__all__ = [
    'Softplus',
]


JAX_MODE = False  # Overwritten by rewrite script.


# TODO(b/155501444): Remove this when tf.nn.softplus is fixed.
if JAX_MODE:
  _stable_grad_softplus = tf.nn.softplus
else:

  @tf.custom_gradient
  def _stable_grad_softplus(x):
    """A (more) numerically stable softplus than `tf.nn.softplus`."""
    x = tf.convert_to_tensor(x)
    if x.dtype == tf.float64:
      cutoff = -20
    else:
      cutoff = -9

    y = tf.where(x < cutoff, tf.math.log1p(tf.exp(x)), tf.nn.softplus(x))

    def grad_fn(dy):
      return dy * tf.where(x < cutoff, tf.exp(x), tf.nn.sigmoid(x))

    return y, grad_fn


class Softplus(bijector.Bijector):
  """Bijector which computes `Y = g(X) = Log[1 + exp(X)]`.

  The softplus `Bijector` has the following two useful properties:

  * The domain is the positive real numbers
  * `softplus(x) approx x`, for large `x`, so it does not overflow as easily as
    the `Exp` `Bijector`.

  The optional nonzero `hinge_softness` parameter changes the transition at
  zero.  With `hinge_softness = c`, the bijector is:

    ```f_c(x) := c * g(x / c) = c * Log[1 + exp(x / c)].```

  For large `x >> 1`, `c * Log[1 + exp(x / c)] approx c * Log[exp(x / c)] = x`,
  so the behavior for large `x` is the same as the standard softplus.

  As `c > 0` approaches 0 from the right, `f_c(x)` becomes less and less soft,
  approaching `max(0, x)`.

  * `c = 1` is the default.
  * `c > 0` but small means `f(x) approx ReLu(x) = max(0, x)`.
  * `c < 0` flips sign and reflects around the `y-axis`: `f_{-c}(x) = -f_c(-x)`.
  * `c = 0` results in a non-bijective transformation and triggers an exception.

    Example Use:

    ```python
    # Create the Y=g(X)=softplus(X) transform which works only on Tensors with 1
    # batch ndim and 2 event ndims (i.e., vector of matrices).
    softplus = Softplus()
    x = [[[1., 2],
          [3, 4]],
         [[5, 6],
          [7, 8]]]
    log(1 + exp(x)) == softplus.forward(x)
    log(exp(x) - 1) == softplus.inverse(x)
    ```

    Note: log(.) and exp(.) are applied element-wise but the Jacobian is a
    reduction over the event space.
  """

  @distribution_util.AppendDocstring(
      kwargs_dict={
          'hinge_softness': (
              'Nonzero floating point `Tensor`.  Controls the softness of what '
              'would otherwise be a kink at the origin.  Default is 1.0'),
          'low': (
              'Nonzero floating point `Tensor` lower bound on output values. '
              'Implicitly zero if `None`. Otherwise, the '
              'transformation `y = softplus(x) + low` is implemented. This '
              'is equivalent to a `Chain([Shift(low), Softplus()])` bijector '
              'and is provided for convenience.')})
  def __init__(self,
               hinge_softness=None,
               low=None,
               validate_args=False,
               name='softplus'):
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      self._hinge_softness = tensor_util.convert_nonref_to_tensor(
          hinge_softness, name='hinge_softness')
      self._low = tensor_util.convert_nonref_to_tensor(low, name='low')
      super(Softplus, self).__init__(
          forward_min_event_ndims=0,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @classmethod
  def _is_increasing(cls):
    return True

  def _forward(self, x):
    if self.hinge_softness is None:
      y = _stable_grad_softplus(x)
    else:
      hinge_softness = tf.cast(self.hinge_softness, x.dtype)
      y = hinge_softness * _stable_grad_softplus(x / hinge_softness)
    return y + self.low if self.low is not None else y

  def _inverse(self, y):
    y = y - self.low if self.low is not None else y
    if self.hinge_softness is None:
      return generic_math.softplus_inverse(y)
    hinge_softness = tf.cast(self.hinge_softness, y.dtype)
    return hinge_softness * generic_math.softplus_inverse(y / hinge_softness)

  def _inverse_log_det_jacobian(self, y):
    # Could also do:
    #   ildj = tf.reduce_sum(y - tfp.math.softplus_inverse(y),
    #                              axis=event_dims)
    # but the following is more numerically stable. Ie,
    # Y = Log[1 + exp{X}] ==> X = Log[exp{Y} - 1]
    # ==> dX/dY = exp{Y} / (exp{Y} - 1)
    #           = 1 / (1 - exp{-Y}),
    # which is the most stable for large Y > 0. For small Y, we use
    # 1 - exp{-Y} approx Y.
    y = y - self.low if self.low is not None else y
    if self.hinge_softness is not None:
      y = y / tf.cast(self.hinge_softness, y.dtype)
    return -tf.math.log(-tf.math.expm1(-y))

  def _forward_log_det_jacobian(self, x):
    if self.hinge_softness is not None:
      x = x / tf.cast(self.hinge_softness, x.dtype)
    return -tf.math.softplus(-x)

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict(
        hinge_softness=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: Softplus(low=dtype_util.eps(dtype)))),
        low=parameter_properties.ParameterProperties())

  @property
  def hinge_softness(self):
    return self._hinge_softness

  @property
  def low(self):
    return self._low

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if (self.hinge_softness is not None and
        is_init != tensor_util.is_ref(self.hinge_softness)):
      assertions.append(assert_util.assert_none_equal(
          dtype_util.as_numpy_dtype(self._hinge_softness.dtype)(0),
          self.hinge_softness,
          message='Argument `hinge_softness` must be non-zero.'))
    return assertions
