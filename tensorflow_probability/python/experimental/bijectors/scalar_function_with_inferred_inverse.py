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

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import callable_util
from tensorflow_probability.python.internal import custom_gradient as tfp_custom_gradient
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math import gradient
from tensorflow_probability.python.math import root_search

__all__ = ['ScalarFunctionWithInferredInverse']


class ScalarFunctionWithInferredInverse(bijector.Bijector):
  """Bijector to associate a numeric inverse with any invertible function."""

  def __init__(self,
               fn,
               domain_constraint_fn=None,
               root_search_fn=root_search.find_root_secant,
               max_iterations=50,
               require_convergence=True,
               additional_scalar_parameters_requiring_gradients=(),
               dtype=None,
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
      additional_scalar_parameters_requiring_gradients: Optional list of
        additional Tensor parameters with respect to which `fn` is
        differentiable. Each parameter is a (batch of) scalar(s) whose shape
        must broadcast with the shapes of `x` and `y`. Parameters are passed as
        `fn(x, *additional_scalar_parameters_requiring_gradients)`;
        explicitly passing a parameter ensures that calls to `inverse` and
        `inverse_log_det_jacobian` will generate correct gradients to that
        parameter. Parameters *not* passed here (for example,
        anything in the closure of `fn`) will not, in general, receive
        gradients.
        Default value: `()`.
      dtype: `tf.dtype` supported by this `Bijector`. `None` means dtype is not
        enforced.
        Default value: `None`.
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
      self._domain_constraint_fn = domain_constraint_fn
      self._root_search_fn = root_search_fn
      self._max_iterations = max_iterations
      self._require_convergence = require_convergence
      # TODO(davmre): for cases with lots of parameters, we might prefer to
      # support a single vector parameter in place of many scalars, so that
      # VJPs and JVPs can be computed efficiently using actual matrix ops.
      self._additional_scalar_parameters_requiring_gradients = (
          additional_scalar_parameters_requiring_gradients)

      self._bound_fn = (
          lambda x: fn(x, *additional_scalar_parameters_requiring_gradients))
      self._inverse = self._wrap_inverse_with_implicit_gradient()

      super(ScalarFunctionWithInferredInverse, self).__init__(
          parameters=parameters,
          dtype=dtype,
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

  @property
  def bound_fn(self):
    """Forward `fn` with any extra args bound, so that `y = bound_fn(x)`."""
    return self._bound_fn

  def _batch_shape(self, x_event_ndims):
    try:
      # Trace the function to extract its batch shape without executing it.
      fn_shape = callable_util.get_output_spec(
          lambda x: self.bound_fn(self.domain_constraint_fn(x)),  # pylint: disable=not-callable
          tf.TensorSpec([], dtype=self.dtype if self.dtype else tf.float32)
          ).shape
    except TypeError:  # `dtype` wasn't specified.
      return tf.TensorShape(None)

    fn_rank = tensorshape_util.rank(fn_shape)
    if fn_rank is not None:
      return fn_shape[:fn_rank - x_event_ndims]
    return fn_shape

  def _batch_shape_tensor(self, x_event_ndims):
    fn_shape = ps.shape(
        self.bound_fn(self.domain_constraint_fn(0.)))  # pylint: disable=not-callable
    return fn_shape[:ps.rank_from_shape(fn_shape) - x_event_ndims]

  def _forward(self, x):
    return self.bound_fn(x)

  def _inverse_no_gradient(self, y):
    # Search for a root in unconstrained space.
    unconstrained_root, _, num_iterations = self.root_search_fn(
        lambda ux: (self.bound_fn(self.domain_constraint_fn(ux)) - y),  # pylint: disable=not-callable
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

    def _vjp_fwd(y, *args_with_full_batch_shape):
      # Prevent autodiff from trying to backprop through the root search.
      x = tf.stop_gradient(self._inverse_no_gradient(y))
      return x, (
          # Auxiliary values for the backwards pass.
          x, y, args_with_full_batch_shape)

    # By the inverse function theorem, the derivative of an
    # inverse function is the reciprocal of the forward derivative. This has
    # been popularized in machine learning by [1].
    # [1] Michael Figurnov, Shakir Mohamed, Andriy Mnih (2018). Implicit
    #     Reparameterization Gradients. https://arxiv.org/abs/1805.08498.
    def _vjp_bwd(aux, dresult_d_x):
      x, y, args_with_full_batch_shape = aux
      dy_dx, *dy_dargs = _make_gradient_fn_of_y(
          self.fn, x)(y, *args_with_full_batch_shape)
      return [
          # Derivative of `y = f_inv(x)`.
          dresult_d_x / dy_dx
      ] + [
          # Derivatives to parameters by implicit differentiation. This
          # is a special case of eqn (6) in Figurnov et al. [1].
          - dresult_d_x * dy_darg / dy_dx
          for dy_darg in dy_dargs
          ]

    def _inverse_jvp(primals, tangents):
      y, *args_with_full_batch_shape = primals
      dy, *dargs = tangents

      # Prevent autodiff from trying to backprop through the root search.
      x = tf.stop_gradient(self._inverse_no_gradient(y))

      dy_dx, *dy_dargs = _make_gradient_fn_of_y(self.fn, x)(
          y, *args_with_full_batch_shape)
      dx = dy / dy_dx
      for (darg, dy_darg) in zip(dargs, dy_dargs):
        # Derivatives to parameters by implicit differentiation. This
        # is a special case of eqn (6) in Figurnov et al. [1].
        dx -= darg * dy_darg / dy_dx
      return x, dx

    @tfp_custom_gradient.custom_gradient(
        vjp_fwd=_vjp_fwd,
        vjp_bwd=_vjp_bwd,
        jvp_fn=_inverse_jvp)
    def _inverse_with_gradient(y, *args_with_full_batch_shape):
      del args_with_full_batch_shape  # Used by the backwards pass only.
      return self._inverse_no_gradient(y)

    def _arg_broadcasting_wrapped_inverse(y):
      args = self._additional_scalar_parameters_requiring_gradients
      if args:
        # Gradients of inverse wrt args will have full batch shape. If any arg
        # has a smaller shape, we need to deduplicate its gradient to have
        # that same smaller shape. An easy way to do that is to backprop
        # through `broadcast_to`. Note that this needs to occur outside of the
        # custom_gradient machinery, so that the backprop actually happens.
        # TODO(davmre): Do gradient reductions directly in the VJP using
        # `tf.raw_ops.BroadcastGradientArgs` so we can remove this wrapper
        # and avoid spurious broadcasting.
        full_batch_shape = ps.broadcast_shape(
            self.experimental_batch_shape_tensor(), ps.shape(y))
        args = [tf.broadcast_to(arg, full_batch_shape) for arg in args]
      return _inverse_with_gradient(y, *args)

    return _arg_broadcasting_wrapped_inverse


def _make_gradient_fn_of_y(fn, x):
  """Defines the gradient of `fn(x, *args)` as a function of `(y, *args)`.

  The returned function `grad_fn_of_y(y, *args)` computes the partial
  derivative(s) of the invertible function `fn(x, *args)` with respect to `x`
  and any other arguments `*args`. It is conceptually defined by

  ```
  def grad_fn_of_y(y, *args):
    x = inverse_of_fn(y, *args)
    return gradient(fn, x, *args)
  ```

  Unfortunately, we don't have direct access to `inverse_of_fn`, so instead we
  require that the appropriate `x` is given to us, and the function we return
  is valid *only* at `y = fn(x, *args)` for the provided value of `x`.

  Why would we bother to define a function that's valid only at a single point?
  The real goal is to annotate the *derivatives* of `grad_fn_of_y`
  (which are second-order derivatives of `fn`) so that it appears to the
  autodiff system *as if* we had written the 'conceptual' code above, using a
  differentiable inverse function. Without this trick, we could still use
  a first-order derivative of `fn` to, e.g., define a bijector
  log-det-jacobian, but we wouldn't be able to differentiate it.

  Args:
    fn: Python `callable` scalar function of a scalar `x` (and
      optionally, of additional scalar arguments `*args`), twice differentiable
      and invertible in its first argument `x`.
    x: Float `Tensor` input at which `fn` and its derivatives are evaluated.
  Returns:
    grad_fn_of_y: Python `callable` that takes arguments `(y, *args)` and
      returns a list of first-order partial derivatives of `y = fn(x, *args)`.
      If `fn` takes only one argument, the value returned is a
      single-element list `[dy_dx]`. The value of `y` is assumed to be
      `y = fn(x, *args)` and is ignored, but derivatives wrt `y` and the other
      `*args` are defined.

  """

  def _grad_fn(x, *args):
    _, grads = gradient.value_and_gradient(fn, x, *args)
    return grads if args else [grads]  # Always return a list.

  def _grad_fn_of_y_fwd(unused_y, *args):
    dy_dx, *dy_dargs = _grad_fn(x, *args)
    return [
        (dy_dx, *dy_dargs),  # Actual return values.
        (dy_dx, dy_dargs, args)  # Auxiliary values for the backward pass.
    ]

  def _second_order_terms(*args):
    """Computes entries of the (Hessian of `fn`) == (Jacobian of `_grad_fn`)."""
    # Partial derivatives of _grad_fn's first output (dy/dx) wrt `(x, *args)`.
    _, (d2y_dx2, *d2y_dx_dargs) = gradient.value_and_gradient(
        lambda x_and_args: _grad_fn(*x_and_args)[0], (x,) + args,
        auto_unpack_single_arg=False)

    # Partial derivatives of additional outputs (dy/da, etc) wrt the input
    # *args (if any). Note that we don't need derivatives of these outputs wrt
    # `x`, since these are equal to the values we computed above in
    # `d2y_dx_dargs`by the [symmetry of partial derivatives](
    #   https://en.wikipedia.org/wiki/Symmetry_of_second_derivatives). This
    # could also in principle be applied to optimize redundant partial
    # derivatives computed in this loop, although this would be incompatible
    # with parallelizing the loop (which is probably a bigger win).
    d2y_dargs2 = []
    for i in range(len(args)):
      # It may be possible to run this loop in parallel with `vectorized_map`,
      # although this would only matter in cases with >> 1 arguments.
      _, d2y_dargs2_row = gradient.value_and_gradient(
          lambda args, i=i: _grad_fn(x, *args)[1 + i],
          args,
          auto_unpack_single_arg=False)
      d2y_dargs2.append(d2y_dargs2_row)

    return d2y_dx2, d2y_dx_dargs, d2y_dargs2

  def _jacobian_of_grad_fn_wrt_y(dy_dx,
                                 dy_dargs,
                                 d2y_dx2,
                                 d2y_dx_dargs,
                                 d2y_dargs2):
    """Builds a Jacobian matrix for `_grad_fn_of_y(y, *args)`.

    Args:
      dy_dx: Tensor.
      dy_dargs: List of Tensors of length `len(args)`.
      d2y_dx2: Tensor.
      d2y_dx_dargs: List of Tensors of length `len(args)`.
      d2y_dargs2: Nested list of length `len(args)`, containing lists of length
        `len(args)`. Due to the symmetry of partial derivatives, the order of
        indices doesn't matter.
    Returns:
      jacobian: Nested list of length `len(args + 1)`, corresponding to the
        outputs of `_grad_fn_of_y` including the first output `dy/dx`. These
        contain inner lists, each also of `len(args + 1)`, corresponding
        to the inputs of `_grad_fn_of_y` including the first input `y`.
        The inner lists contain Tensor partial derivatives.

    ### Mathematical details

    Here we derive partial derivatives of the outputs of `grad_fn_of_y`:

    ```
    dy/dx, dy/da, dy/db, ... = grad_fn_of_y(y, a, b, ...)
                             = grad_fn(x(y, a, b, ...), a, b, ...)
    ```

    with respect to its inputs. Our goal will be to express these partial
    derivatives, which are second-order derivatives of the original `fn`, in
    terms of quantities that we can evaluate by differentiating the forward
    `y = fn(x, a, b, ...)`; that is, we always want `dy` in the numerator.

    We can divide these into two cases (1) and (2) in the Jacobian matrix:

    ```
                           inputs
                       y     a      b     ...
      outputs: dy/dx  (1)   (2)    (2)
               dy/da  (1)   (2)    (2)   ...
               dy/db  (1)   (2)    (2)
               ...
    ```

    The first case (1) contains the derivatives with respect to `y`, which fall
    out straightforwardly from the inverse function theorem
    `dx/dy = 1 / (dy/dx)`, and the chain rule:

    ```
    d (dy/dx) / dy = d (dy/dx) / dx  * dx/dy
                   = d2y/dx2 /  dy/dx
    d (dy/da) / dy = d (dy/da) / dx  * dx/dy
                   = d2y/dxa /  dy/dx
    # etc.
    ```

    The second case (2) includes derivatives with respect to the additional
    arguments `a, b, ...`. These are more involved, because these arguments
    enter `grad_fn_of_y` twice: in addition to their direct effect on
    `grad_fn(x, a, b, ...)`, we must also account for the indirect effect via
    `x = inverse_of_fn(y, a, b, ...)`. This requires total differentiation:

    ```
    d f(x(y, a, b, ...), a, b, ...) / db = partial(f, x) * partial(x, b)
                                          + partial(f, a) * partial(a, b)
                                          + partial(f, b) * partial(b, b)
                                          + ...
    ```

    where the cross terms between constants, like `partial(a, b)`, are zero and
    disappear. Considering a concrete Jacobian entry `d (dy/da) / db`, we find:

    ```
    d (dy/da) / db = d2y/dab + d2y/dxa * dx/db
                   = d2y/dab - d2y/dxa * (dx/dy) * dy/db
                     # by the implicit differentiation identity
                     # `dx/db = -(dx/dy) * (dy/db)` (eqn (6) of Figurnov et al.
                   = d2y/dab - (d2y/dxa * dy/db) / (dy/dx)
                     # by the inverse function theorem.
    ```

    Analagous derivations, using any output in place of `dy/da`,
    and any non-`y` input in place of `b`, are sufficient to compute all of the
    remaining Jacobian terms.

    ```

    """
    # Avoid errors if any second derivatives of `fn` are `None`. It is assumed
    # that all first derivatives are non-`None`.
    zero_if_none = lambda v: 0. if v is None else v

    # Case (1): derivative of first output wrt first input.
    jacobian_first_row = [zero_if_none(d2y_dx2) / dy_dx]
    for d2y_dx_darg, dy_darg in zip(d2y_dx_dargs, dy_dargs):
      # Case (2): derivative of first output wrt subsequent input.
      jacobian_first_row.append(
          zero_if_none(d2y_dx_darg) - zero_if_none(d2y_dx2) * dy_darg / dy_dx)

    jacobian_all_rows = [jacobian_first_row]
    for (d2y_dx_darg, d2y_dargs2_row) in zip(d2y_dx_dargs, d2y_dargs2):
      # Case (1): derivative of secondary return value wrt `y`.
      jacobian_current_row = [zero_if_none(d2y_dx_darg) / dy_dx]
      for arg_idx in range(len(dy_dargs)):
        # Case (2): derivative of a secondary return value wrt secondary input.
        jacobian_current_row.append(
            zero_if_none(d2y_dargs2_row[arg_idx])
            - zero_if_none(d2y_dx_darg) * dy_dargs[arg_idx] / dy_dx)
      jacobian_all_rows.append(jacobian_current_row)

    return jacobian_all_rows

  def _grad_fn_of_y_bwd(aux, dresult_d_grad_fn):
    dy_dx, dy_dargs, args = aux

    jacobian = _jacobian_of_grad_fn_wrt_y(dy_dx,
                                          dy_dargs,
                                          *_second_order_terms(*args))

    # We could concatenate the Jacobian components and do a real
    # `tf.linalg.matvec`, instead of coding a vector-matrix product manually,
    # but this is likely not a meaningful optimization in typical cases with
    # 1-2 args.
    gradients = []
    for j in range(1 + len(args)):
      g = 0.
      for i in range(1 + len(args)):
        g += dresult_d_grad_fn[i] * jacobian[i][j]
      gradients.append(g)
    return gradients

  def _grad_fn_of_y_jvp(primals, tangents):
    unused_y, *args = primals
    dy_dx, *dy_dargs = _grad_fn(x, *args)
    jacobian = _jacobian_of_grad_fn_wrt_y(dy_dx,
                                          dy_dargs,
                                          *_second_order_terms(*args))

    # We could concatenate the Jacobian components and do a real
    # `tf.linalg.matvec`, instead of coding a matrix-vector product manually,
    # but this is likely not a meaningful optimization in typical cases with
    # 1-2 args.
    tangents_out = []
    for i in range(len(args) + 1):
      tangent_out = 0.
      for j in range(len(args) + 1):
        tangent_out += tangents[j] * jacobian[i][j]
      tangents_out.append(tangent_out)

    return (dy_dx, *dy_dargs), tuple(tangents_out)

  @tfp_custom_gradient.custom_gradient(
      vjp_fwd=_grad_fn_of_y_fwd,
      vjp_bwd=_grad_fn_of_y_bwd,
      jvp_fn=_grad_fn_of_y_jvp)
  def _grad_fn_of_y_with_gradient(y, *args):
    del y  # Unused.
    return _grad_fn(x, *args)

  return _grad_fn_of_y_with_gradient
