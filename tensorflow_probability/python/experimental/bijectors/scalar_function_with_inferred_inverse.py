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
               root_search_fn=tfp_math.secant_root,
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
      x = self._inverse_no_gradient(y)
      return x, x  # Keep `x` as an auxiliary value for the backwards pass.

    # By the inverse function theorem, the derivative of an
    # inverse function is the reciprocal of the forward derivative. This has
    # been popularized in machine learning by [1].
    # [1] Michael Figurnov, Shakir Mohamed, Andriy Mnih (2018). Implicit
    #     Reparameterization Gradients. https://arxiv.org/abs/1805.08498.
    def _vjp_bwd(x, grad_x):
      _, grads = tfp_math.value_and_gradient(self.fn, x)
      return (grad_x / grads,)

    @tfp_custom_gradient.custom_gradient(
        vjp_fwd=_vjp_fwd,
        vjp_bwd=_vjp_bwd)
    def _inverse_with_gradient(y):
      return self._inverse_no_gradient(y)
    return _inverse_with_gradient
