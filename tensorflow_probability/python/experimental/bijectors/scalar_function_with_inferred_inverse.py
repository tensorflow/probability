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
"""Bijector to associate a numeric inverse with any invertible function."""

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import custom_gradient as tfp_custom_gradient

__all__ = ['ScalarFunctionWithInferredInverse']


class ScalarFunctionWithInferredInverse(bijector.Bijector):
  """Bijector to associate a numeric inverse with any invertible function."""

  def __init__(self,
               fn,
               domain_constraint_fn=None,
               root_search_fn=tfp_math.find_root_secant,
               max_iterations=50,
               require_convergence=True,
               validate_args=False,
               name='scalar_function_with_inferred_inverse'):
    """Initialize the ScalarFunctionWithInferredInverse bijector.

    Args:
      fn: Python `callable` taking a single Tensor argument `x`, and returning a
        Tensor `y` of the same shape. This is assumed to be an invertible
        (continuous and monotonic) function applied elementwise to `x`.
      domain_constraint_fn: optional Python `callable` that returns values
        within the domain of `fn`, used to constrain the root search. For any
        real-valued input `r`, the value `x = domain_constraint_fn(r)` should be
        a valid input to `fn`.
        Default value: `None`.
      root_search_fn: Optional Python `callable` used to search for roots of an
        objective function. This should have signature
        `root_search_fn(objective_fn, initial_x, max_iterations=None)`
        and return a tuple containing three `Tensor`s
        `(estimated_root, objective_at_estimated_root, num_iterations)`.
        Default value: `tfp.math.secant_root`.
      max_iterations: Optional Python integer maximum number of iterations to
        run the root search algorithm.
        Default value: `50`.
      require_convergence: Optional Python `bool` indicating whether to return
        inverse values when the root-finding algorithm may not have
        converged. If `True`, such values are replaced by `NaN`.
        Default value: `True`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
        Default value: `scalar_function_with_inferred_inverse`.
    """
    parameters = locals()
    with tf.name_scope(name):
      if domain_constraint_fn is None:
        domain_constraint_fn = lambda x: x
      self._fn = fn
      self._root_search_fn = root_search_fn
      self._domain_constraint_fn = domain_constraint_fn
      self._require_convergence = require_convergence
      self._max_iterations = max_iterations

      self._inverse = self._wrap_inverse_with_implicit_gradient()

      super(ScalarFunctionWithInferredInverse, self).__init__(
          parameters=parameters,
          forward_min_event_ndims=0,
          inverse_min_event_ndims=0,
          validate_args=validate_args,
          name=name)

  @property
  def domain_constraint_fn(self):
    return self._domain_constraint_fn

  @property
  def fn(self):
    return self._fn

  @property
  def max_iterations(self):
    return self._max_iterations

  @property
  def require_convergence(self):
    return self._require_convergence

  @property
  def root_search_fn(self):
    return self._root_search_fn

  def _forward(self, x):
    return self.fn(x)  # pylint: disable=not-callable

  def _inverse_no_gradient(self, y):
    # Search for a root in unconstrained space.
    unconstrained_root, _, num_iterations = self.root_search_fn(
        lambda ux: (self.fn(self.domain_constraint_fn(ux)) - y),  # pylint: disable=not-callable
        tf.ones_like(y),
        max_iterations=self.max_iterations)
    x = self.domain_constraint_fn(unconstrained_root)  # pylint: disable=not-callable
    if self.require_convergence:
      x = tf.where(
          num_iterations < self.max_iterations,
          x,
          tf.cast(np.nan, x.dtype))
    return x

  def _wrap_inverse_with_implicit_gradient(self):
    """Wraps the inverse to provide implicit reparameterization gradients."""

    def _vjp_fwd(y):
      # Prevent autodiff from trying to backprop through the root search.
      x = tf.stop_gradient(self._inverse_no_gradient(y))
      return x, (x, y)  # Auxiliary values for the backwards pass.

    # By the inverse function theorem, the derivative of an
    # inverse function is the reciprocal of the forward derivative. This has
    # been popularized in machine learning by [1].
    # [1] Michael Figurnov, Shakir Mohamed, Andriy Mnih (2018). Implicit
    #     Reparameterization Gradients. https://arxiv.org/abs/1805.08498.
    def _vjp_bwd(aux, dresult_dx):
      x, y = aux
      return [dresult_dx /
              _make_dy_dx_with_implicit_derivative_wrt_y(self.fn, x)(y)]

    def _inverse_jvp(primals, tangents):
      y, = primals
      dy, = tangents
      # Prevent autodiff from trying to backprop through the root search.
      x = tf.stop_gradient(self._inverse_no_gradient(y))
      return x, dy / _make_dy_dx_with_implicit_derivative_wrt_y(self.fn, x)(y)

    @tfp_custom_gradient.custom_gradient(
        vjp_fwd=_vjp_fwd,
        vjp_bwd=_vjp_bwd,
        jvp_fn=_inverse_jvp)
    def _inverse_with_gradient(y):
      return self._inverse_no_gradient(y)
    return _inverse_with_gradient


def _make_dy_dx_with_implicit_derivative_wrt_y(fn, x):
  """Given `y = fn(x)`, returns a function `dy_dx(y)` differentiable wrt y.

  The returned function `dy_dx(y)` computes the reciprocal of the derivative of
  `x = inverse_of_fn(y)`. By the inverse function theorem, this is just the
  derivative `dy / dx` of `y = fn(x)`. Even though we'll actually care about
  the derivative of the inverse, `dx / dy`, it's more efficient to return the
  reciprocal of that quantity from the forward derivative.

  Since `dy_dx(y)` is the first derivative of `fn(x)` evaluated at
  `x = inverse_of_fn(y)`, we define *its* derivative in terms of the
  second derivative of `fn`, via the chain rule:

  ```
  d / dy fn'(inverse_of_fn(y)) = fn''(inverse_of_fn(y)) * inverse_of_fn'(y)
                               = fn''(x) / fn'(x)
  ```

  When bijector log-det-jacobians are computed using autodiff, as in
  `ScalarFunctionWithInferredInverse`, the gradients of the log-det-jacobians
  make use of these second-derivative annotations.

  Args:
    fn: Python `callable` invertible scalar function of a scalar `x`. Must be
      twice differentiable.
    x: Float `Tensor` input at which `fn` and its derivatives are evaluated.
  Returns:
    dy_dx_fn: Python `callable` that takes an argument `y` and returns the
      derivative of `fn(x)`. The argument `y` is ignored (it is assumed to be
      `y = fn(x)`), but the derivative of `dy_dx_fn` wrt `y` is defined.
  """

  # To override first and second derivatives of the inverse
  # (second derivatives are needed for gradients of
  #  `inverse_log_det_jacobian`s), we'll need the first and second
  # derivatives from the forward direction.
  def _dy_dx_fwd(unused_y):
    first_order = lambda x: tfp_math.value_and_gradient(fn, x)[1]
    dy_dx, d2y_dx2 = tfp_math.value_and_gradient(first_order, x)
    return (dy_dx,
            (dy_dx, d2y_dx2))  # Auxiliary values for the second-order pass.

  # Chain rule for second derivative of an inverse function:
  # f''(inv_f(y)) = f''(x) * inv_f'(y)
  #               = f''(x) / f'(x).
  def _dy_dx_bwd(aux, dresult_d_dy_dx):
    dy_dx, d2y_dx2 = aux
    return [dresult_d_dy_dx * d2y_dx2 / dy_dx]

  def _dy_dx_jvp(primals, tangents):
    unused_y, = primals
    dy, = tangents
    first_order = lambda x: tfp_math.value_and_gradient(fn, x)[1]
    dy_dx, ddy_dx2 = tfp_math.value_and_gradient(first_order, x)
    return dy_dx, (dy / dy_dx) * ddy_dx2

  # Naively, autodiff of this derivative would attempt to backprop through
  # `x = root_search(fn, y)` when computing the second derivative with
  # respect to `y`. Since that's no good, we need to provide our own
  # custom gradient wrt `y`.
  @tfp_custom_gradient.custom_gradient(
      vjp_fwd=_dy_dx_fwd,
      vjp_bwd=_dy_dx_bwd,
      jvp_fn=_dy_dx_jvp)
  def _dy_dx_fn(y):
    del y  # Unused.
    _, dy_dx = tfp_math.value_and_gradient(fn, x)
    return dy_dx

  return _dy_dx_fn
