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
"""Implements the Hager-Zhang inexact line search algorithm.

Line searches are a central component for many optimization algorithms (e.g.
BFGS, conjugate gradient etc). Most of the sophisticated line search methods
aim to find a step length in a given search direction so that the step length
satisfies the
[Wolfe conditions](https://en.wikipedia.org/wiki/Wolfe_conditions).
[Hager-Zhang 2006](http://users.clas.ufl.edu/hager/papers/CG/cg_compare.pdf)
algorithm is a refinement of the commonly used
[More-Thuente](https://dl.acm.org/citation.cfm?id=192132) algorithm.

This module implements the Hager-Zhang algorithm.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow as tf

__all__ = [
    'hager_zhang',
]

# Container to hold the function value and the derivative at a given point.
# Each entry is a scalar tensor of real dtype. Used for internal data passing.
_FnDFn = collections.namedtuple('_FnDFn', ['x', 'f', 'df'])


# Valuetype to hold information for the floating point precision data.
_FloatInfo = collections.namedtuple('_FloatInfo', ['eps', 'nextfloat'])


def _machine_eps(dtype):
  """Returns the machine epsilon for the supplied dtype."""
  if isinstance(dtype, tf.DType):
    dtype = dtype.as_numpy_dtype()
  return np.finfo(dtype).eps


def _next_after(x):
  """Computes the next larger floating point number.

  Tensorflow analogue of the function np.nextafter.
  Ideally, this function should have a native C++ kernel in TF core but until
  that is done, we are forced to use this implementation.

  Args:
    x: `Tensor` of real dtype and any shape. The value for which to compute
      the next floating point number.

  Returns:
    A tuple containing the following attributes:
      eps: The floating point resolution at `x`. A tensor of same shape and
        dtype as the input `x`.
      nextfloat: `Tensor` of the same dtype and shape as `x`. The smallest
        value `y` which is greater than `x` for that dtype.
  """
  cond = lambda epsilon: ~tf.equal(x + epsilon / 2, x)
  body = lambda epsilon: epsilon / 2
  epsilon = tf.while_loop(cond, body, (tf.ones_like(x),))
  return _FloatInfo(eps=epsilon, nextfloat=x + epsilon)


HagerZhangLineSearchResult = collections.namedtuple(
    'HagerZhangLineSearchResults', [
        'converged',  # Whether a pt satisfying Wolfe/Approx wolfe was found.
        'failed',  # Whether the line search failed. It can fail if the
                   # objective function or the gradient are not finite at
                   # an evaluation point.
        'func_evals',  # Number of function evaluations made.
        'iterations',  # Number of line search iterations made.
        'left_pt',  # The left end point of the final bracketing interval.
                    # If converged is True, it is equal to 'right_pt'.
                    # Otherwise, it corresponds to the last interval computed.
        'objective_at_left_pt',  # The function value at the left end point.
                                 # If converged is True, it is equal to
                                 # `fn_right_step`. Otherwise, it
                                 # corresponds to the last interval computed.
        'grad_objective_at_left_pt',  # The derivative of the function at the
                                      # left end point. If converged is True,
                                      # it is equal to
                                      # `grad_objective_at_right_pt`.
                                      # Otherwise, it corresponds to the last
                                      # interval computed.
        'right_pt',  # The right end point of the final bracketing interval.
                     # If converged is True, it is equal to 'left_pt'.
                     # Otherwise, it corresponds to the last interval computed.
        'objective_at_right_pt',  # The function value at the right end point.
                                  # If converged is True, it is equal to
                                  # `objective_at_left_pt`.
                                  # Otherwise, it corresponds to the last
                                  # interval computed.
        'grad_objective_at_right_pt'  # The derivative of the function at the
                                      # right end point. If converged is True,
                                      # it is equal to
                                      # `grad_objective_at_left_pt`.
                                      # Otherwise it corresponds to the last
                                      # interval computed.
    ])


def hager_zhang(value_and_gradients_function,
                initial_step_size=None,
                objective_at_zero=None,
                grad_objective_at_zero=None,
                objective_at_initial_step_size=None,
                grad_objective_at_initial_step_size=None,
                threshold_use_approximate_wolfe_condition=1e-6,
                shrinkage_param=0.66,
                expansion_param=5.0,
                sufficient_decrease_param=0.1,
                curvature_param=0.9,
                step_size_shrink_param=0.1,
                max_iterations=50,
                name=None):
  """The Hager Zhang line search algorithm.

  Performs an inexact line search based on the algorithm of
  [Hager and Zhang (2006)][2].
  The univariate objective function `value_and_gradients_function` is typically
  generated by projecting
  a multivariate objective function along a search direction. Suppose the
  multivariate function to be minimized is `g(x1,x2, .. xn)`. Let
  (d1, d2, ..., dn) be the direction along which we wish to perform a line
  search. Then the projected univariate function to be used for line search is

  ```None
    f(a) = g(x1 + d1 * a, x2 + d2 * a, ..., xn + dn * a)
  ```

  The directional derivative along (d1, d2, ..., dn) is needed for this
  procedure. This also corresponds to the derivative of the projected function
  `f(a)` with respect to `a`. Note that this derivative must be negative for
  `a = 0` if the direction is a descent direction.

  The usual stopping criteria for the line search is the satisfaction of the
  (weak) Wolfe conditions. For details of the Wolfe conditions, see
  ref. [3]. On a finite precision machine, the exact Wolfe conditions can
  be difficult to satisfy when one is very close to the minimum and as argued
  by [Hager and Zhang (2005)][1], one can only expect the minimum to be
  determined within square root of machine precision. To improve the situation,
  they propose to replace the Wolfe conditions with an approximate version
  depending on the derivative of the function which is applied only when one
  is very close to the minimum. The following algorithm implements this
  enhanced scheme.

  ### Usage:

  Primary use of line search methods is as an internal component of a class of
  optimization algorithms (called line search based methods as opposed to
  trust region methods). Hence, the end user will typically not want to access
  line search directly. In particular, inexact line search should not be
  confused with a univariate minimization method. The stopping criteria of line
  search is the satisfaction of Wolfe conditions and not the discovery of the
  minimum of the function.

  With this caveat in mind, the following example illustrates the standalone
  usage of the line search.

  ```python
    # Define a quadratic target with minimum at 1.3.
    value_and_gradients_function = lambda x: ((x - 1.3) ** 2, 2 * (x-1.3))
    # Set initial step size.
    step_size = tf.constant(0.1)
    ls_result = tfp.optimizer.linesearch.hager_zhang(
        value_and_gradients_function, initial_step_size=step_size)
    # Evaluate the results.
    with tf.Session() as session:
      results = session.run(ls_result)
      # Ensure convergence.
      assert(results.converged)
      # If the line search converged, the left and the right ends of the
      # bracketing interval are identical.
      assert(results.left_pt == result.right_pt)
      # Print the number of evaluations and the final step size.
      print ("Final Step Size: %f, Evaluation: %d" % (results.left_pt,
                                                      results.func_evals))
  ```

  ### References:
  [1]: William Hager, Hongchao Zhang. A new conjugate gradient method with
    guaranteed descent and an efficient line search. SIAM J. Optim., Vol 16. 1,
    pp. 170-172. 2005.
    https://www.math.lsu.edu/~hozhang/papers/cg_descent.pdf

  [2]: William Hager, Hongchao Zhang. Algorithm 851: CG_DESCENT, a conjugate
    gradient method with guaranteed descent. ACM Transactions on Mathematical
    Software, Vol 32., 1, pp. 113-137. 2006.
    http://users.clas.ufl.edu/hager/papers/CG/cg_compare.pdf

  [3]: Jorge Nocedal, Stephen Wright. Numerical Optimization. Springer Series in
    Operations Research. pp 33-36. 2006

  Args:
    value_and_gradients_function: A Python callable that accepts a real scalar
      tensor and returns a tuple of scalar tensors of real dtype containing
      the value of the function and its derivative at that point.
      In usual optimization application, this function would be generated by
      projecting the multivariate objective function along some specific
      direction. The direction is determined by some other procedure but should
      be a descent direction (i.e. the derivative of the projected univariate
      function must be negative at 0.).
    initial_step_size: (Optional) Scalar positive `Tensor` of real dtype. The
      initial value to try to bracket the minimum. Default is `1.` as a float32.
      Note that this point need not necessarily bracket the minimum for the line
      search to work correctly but the supplied value must be greater than
      0. A good initial value will make the search converge faster.
    objective_at_zero: (Optional) Scalar `Tensor` of real dtype. If supplied,
      the value of the function at `0.`. If not supplied, it will be computed.
    grad_objective_at_zero: (Optional) Scalar `Tensor` of real dtype. If
      supplied, the derivative of the  function at `0.`. If not supplied, it
      will be computed.
    objective_at_initial_step_size: (Optional) Scalar `Tensor` of real dtype.
      If supplied, the value of the function at `initial_step_size`.
      If not supplied, it will be computed.
    grad_objective_at_initial_step_size: (Optional) Scalar `Tensor` of real
      dtype. If supplied, the derivative of the  function at
      `initial_step_size`. If not supplied, it will be computed.
    threshold_use_approximate_wolfe_condition: Scalar positive `Tensor`
      of real dtype. Corresponds to the parameter 'epsilon' in
      [Hager and Zhang (2006)][2]. Used to estimate the
      threshold at which the line search switches to approximate Wolfe
      conditions.
    shrinkage_param: Scalar positive Tensor of real dtype. Must be less than
      `1.`. Corresponds to the parameter `gamma` in
      [Hager and Zhang (2006)][2].
      If the secant**2 step does not shrink the bracketing interval by this
      proportion, a bisection step is performed to reduce the interval width.
    expansion_param: Scalar positive `Tensor` of real dtype. Must be greater
      than `1.`. Used to expand the initial interval in case it does not bracket
      a minimum. Corresponds to `rho` in [Hager and Zhang (2006)][2].
    sufficient_decrease_param: Positive scalar `Tensor` of real dtype.
      Bounded above by the curvature param. Corresponds to `delta` in the
      terminology of [Hager and Zhang (2006)][2].
    curvature_param: Positive scalar `Tensor` of real dtype. Bounded above
      by `1.`. Corresponds to 'sigma' in the terminology of
      [Hager and Zhang (2006)][2].
    step_size_shrink_param: Positive scalar `Tensor` of real dtype. Bounded
      above by `1`. If the supplied step size is too big (i.e. either the
      objective value or the gradient at that point is infinite), this factor
      is used to shrink the step size until it is finite.
    max_iterations: Positive scalar `Tensor` of integral dtype or None. The
      maximum number of iterations to perform in the line search. The number of
      iterations used to bracket the minimum are also counted against this
      parameter.
    name: (Optional) Python str. The name prefixed to the ops created by this
      function. If not supplied, the default name 'hager_zhang' is used.

  Returns:
    results: A namedtuple containing the following attributes.
      converged: Boolean scalar `Tensor`. Whether a point satisfying
        Wolfe/Approx wolfe was found.
      func_evals: Scalar int32 `Tensor`. Number of function evaluations made.
      left_pt: Scalar `Tensor` of same dtype as `initial_step_size`. The
        left end point of the final bracketing interval. If converged is True,
        it is equal to `right_pt`. Otherwise, it corresponds to the last
        interval computed.
      objective_at_left_pt: Scalar `Tensor` of same dtype as
        `objective_at_initial_step_size`. The function value at the left
        end point. If converged is True, it is equal to `objective_at_right_pt`.
        Otherwise, it corresponds to the last interval computed.
      grad_objective_at_left_pt: Scalar `Tensor` of same dtype as
        `grad_objective_at_initial_step_size`. The derivative of the function
        at the left end point. If converged is True,
        it is equal to `grad_objective_at_right_pt`. Otherwise it
        corresponds to the last interval computed.
      right_pt: Scalar `Tensor` of same dtype as `initial_step_size`.
        The right end point of the final bracketing interval.
        If converged is True, it is equal to 'step'. Otherwise,
        it corresponds to the last interval computed.
      objective_at_right_pt: Scalar `Tensor` of same dtype as
        `objective_at_initial_step_size`.
        The function value at the right end point. If converged is True, it
        is equal to fn_step. Otherwise, it corresponds to the last
        interval computed.
      grad_objective_at_right_pt'  Scalar `Tensor` of same dtype as
        `grad_objective_at_initial_step_size`.
        The derivative of the function at the right end point.
        If converged is True, it is equal to the dfn_step.
        Otherwise it corresponds to the last interval computed.
  """
  with tf.name_scope(name, 'hager_zhang',
                     [initial_step_size,
                      objective_at_zero,
                      grad_objective_at_zero,
                      objective_at_initial_step_size,
                      grad_objective_at_initial_step_size,
                      threshold_use_approximate_wolfe_condition,
                      shrinkage_param,
                      expansion_param,
                      sufficient_decrease_param,
                      curvature_param]):
    val_0, val_c_input, f_lim, prepare_evals = _prepare_args(
        value_and_gradients_function,
        initial_step_size,
        objective_at_initial_step_size,
        grad_objective_at_initial_step_size,
        objective_at_zero,
        grad_objective_at_zero,
        threshold_use_approximate_wolfe_condition)

    valid_inputs = (_is_finite(val_0) & (val_0.df < 0) &
                    tf.is_finite(val_c_input.x) & (val_c_input.x > 0))

    def _invalid_inputs_fn():
      return HagerZhangLineSearchResult(
          converged=tf.convert_to_tensor(False, name='converged'),
          failed=tf.convert_to_tensor(True, name='failed'),
          func_evals=prepare_evals,
          iterations=tf.convert_to_tensor(0),
          left_pt=val_0.x,
          objective_at_left_pt=val_0.f,
          grad_objective_at_left_pt=val_0.df,
          right_pt=val_0.x,
          objective_at_right_pt=val_0.f,
          grad_objective_at_right_pt=val_0.df)

    def _valid_inputs_fn():
      """Performs bracketing and line search if inputs are valid."""
      # If the value or the gradient at the supplied step is not finite,
      # we attempt to repair it.
      step_size_too_large = ~(tf.is_finite(val_c_input.df) &
                              tf.is_finite(val_c_input.f))

      def _is_too_large_fn():
        return _fix_step_size(value_and_gradients_function,
                              val_c_input,
                              step_size_shrink_param)

      val_c, fix_evals = tf.contrib.framework.smart_cond(
          step_size_too_large,
          _is_too_large_fn,
          lambda: (val_c_input, 0))

      # Check if c is fixed now.
      valid_at_c = _is_finite(val_c) & (val_c.x > 0)
      def _failure_fn():
        # If c is still not good, just return 0.
        return HagerZhangLineSearchResult(
            converged=tf.convert_to_tensor(True, name='converged'),
            failed=tf.convert_to_tensor(False, name='failed'),
            func_evals=prepare_evals + fix_evals,
            iterations=tf.convert_to_tensor(0),
            left_pt=val_0.x,
            objective_at_left_pt=val_0.f,
            grad_objective_at_left_pt=val_0.df,
            right_pt=val_0.x,
            objective_at_right_pt=val_0.f,
            grad_objective_at_right_pt=val_0.df)
      def success_fn():
        """Bracketing and searching to do if all inputs are valid."""
        result = _bracket_and_search(
            value_and_gradients_function,
            val_0,
            val_c,
            f_lim,
            max_iterations,
            shrinkage_param=shrinkage_param,
            expansion_param=expansion_param,
            sufficient_decrease_param=sufficient_decrease_param,
            curvature_param=curvature_param)
        converged = tf.convert_to_tensor(result.found_wolfe, name='converged')
        return HagerZhangLineSearchResult(
            converged=converged,
            failed=tf.convert_to_tensor(result.failed, name='failed'),
            func_evals=result.num_evals + prepare_evals + fix_evals,
            iterations=result.iteration,
            left_pt=result.left.x,
            objective_at_left_pt=result.left.f,
            grad_objective_at_left_pt=result.left.df,
            right_pt=result.right.x,
            objective_at_right_pt=result.right.f,
            grad_objective_at_right_pt=result.right.df)
      return tf.contrib.framework.smart_cond(
          valid_at_c,
          true_fn=success_fn,
          false_fn=_failure_fn)

    return tf.contrib.framework.smart_cond(
        valid_inputs,
        true_fn=_valid_inputs_fn,
        false_fn=_invalid_inputs_fn)


def _fix_step_size(value_and_gradients_function,
                   val_c_input,
                   step_size_shrink_param):
  """Shrinks the input step size until the value and grad become finite."""
  # The maximum iterations permitted are determined as the number of halvings
  # it takes to reduce 1 to 0 in the given dtype.
  iter_max = np.ceil(-np.log2(_machine_eps(val_c_input.x.dtype)))
  def _cond(i, c, f_c, df_c):  # pylint: disable=unused-argument
    return (i < iter_max) & ~(tf.is_finite(f_c) & tf.is_finite(df_c))

  def _body(i, c, f_c, df_c):  # pylint: disable=unused-argument
    next_c = c * step_size_shrink_param
    return (i + 1, next_c) + value_and_gradients_function(next_c)

  evals, next_c, next_f, next_df = tf.while_loop(
      _cond,
      _body,
      (0, val_c_input.x, val_c_input.f, val_c_input.df))
  return _FnDFn(x=next_c, f=next_f, df=next_df), evals


_LineSearchInnerResult = collections.namedtuple('_LineSearchInnerResult', [
    'iteration',
    'found_wolfe',
    'failed',
    'num_evals',
    'left',
    'right'])


def _bracket_and_search(
    value_and_gradients_function,
    val_0,
    val_c,
    f_lim,
    max_iterations,
    shrinkage_param=None,
    expansion_param=None,
    sufficient_decrease_param=None,
    curvature_param=None):
  """Brackets the minimum and performs a line search.

  Args:
    value_and_gradients_function: A Python callable that accepts a real scalar
      tensor and returns a tuple of scalar tensors of real dtype containing
      the value of the function and its derivative at that point.
      In usual optimization application, this function would be generated by
      projecting the multivariate objective function along some specific
      direction. The direction is determined by some other procedure but should
      be a descent direction (i.e. the derivative of the projected univariate
      function must be negative at 0.).
    val_0: Instance of `_FnDFn` containing the value and gradient of the
      objective at 0. The gradient must be negative (i.e. must be a descent
      direction).
    val_c: Instance of `_FnDFn` containing the initial step size and the value
      and gradient of the objective at the initial step size. The step size
      must be positive and finite.
    f_lim: Scalar `Tensor` of float dtype.
    max_iterations: Positive scalar `Tensor` of integral dtype. The maximum
      number of iterations to perform in the line search. The number of
      iterations used to bracket the minimum are also counted against this
      parameter.
    shrinkage_param: Scalar positive Tensor of real dtype. Must be less than
      `1.`. Corresponds to the parameter `gamma` in [Hager and Zhang (2006)][2].
    expansion_param: Scalar positive `Tensor` of real dtype. Must be greater
      than `1.`. Used to expand the initial interval in case it does not bracket
      a minimum. Corresponds to `rho` in [Hager and Zhang (2006)][2].
    sufficient_decrease_param: Positive scalar `Tensor` of real dtype.
      Bounded above by the curvature param. Corresponds to `delta` in the
      terminology of [Hager and Zhang (2006)][2].
    curvature_param: Positive scalar `Tensor` of real dtype. Bounded above
      by `1.`. Corresponds to 'sigma' in the terminology of
      [Hager and Zhang (2006)][2].

  Returns:
    A namedtuple containing the following fields.
      iteration: A scalar int32 `Tensor`. The number of iterations consumed.
      found_wolfe: A scalar boolean `Tensor`. Indicates whether a point
        satisfying the Wolfe conditions has been found. If this is True, the
        interval will be degenerate (i.e. left and right below
        will be identical).
      failed: A scalar boolean `Tensor`. Indicates if invalid function or
        gradient values were encountered (i.e. infinity or NaNs).
      num_evals: A scalar int32 `Tensor`. The total number of function
        evaluations made.
      left: Instance of _FnDFn. The position and the associated value and
        derivative at the updated left end point of the interval.
      right: Instance of _FnDFn. The position and the associated value and
        derivative at the updated right end point of the interval.
  """
  bracket_result = _bracket(
      value_and_gradients_function,
      val_0,
      val_c,
      f_lim,
      max_iterations,
      expansion_param=expansion_param)

  # If the bracketing failed, or we have already exhausted all the allowed
  # iterations, we return an error.
  failed = (~tf.convert_to_tensor(bracket_result.bracketed) |
            tf.greater_equal(bracket_result.iteration, max_iterations))

  def _bracketing_failed_fn():
    return _LineSearchInnerResult(
        iteration=bracket_result.iteration,
        found_wolfe=False,
        failed=True,
        num_evals=bracket_result.num_evals,
        left=val_0,
        right=val_c)

  def _bracketing_success_fn():
    """Performs line search."""
    result = _line_search_after_bracketing(
        value_and_gradients_function,
        val_0,
        bracket_result.left,
        bracket_result.right,
        f_lim,
        bracket_result.iteration,
        max_iterations,
        sufficient_decrease_param=sufficient_decrease_param,
        curvature_param=curvature_param,
        shrinkage_param=shrinkage_param)

    return _LineSearchInnerResult(
        iteration=result.iteration,
        found_wolfe=result.found_wolfe,
        failed=result.failed,
        num_evals=bracket_result.num_evals + result.num_evals,
        left=result.left,
        right=result.right)

  return tf.contrib.framework.smart_cond(
      failed,
      true_fn=_bracketing_failed_fn,
      false_fn=_bracketing_success_fn)


def _line_search_after_bracketing(
    value_and_gradients_function,
    val_0,
    initial_left,
    initial_right,
    f_lim,
    starting_iteration,
    max_iterations,
    sufficient_decrease_param=None,
    curvature_param=None,
    shrinkage_param=None):
  """The main loop of line search after the minimum has been bracketed.

  Args:
    value_and_gradients_function: A Python callable that accepts a real scalar
      tensor and returns a tuple of scalar tensors of real dtype containing
      the value of the function and its derivative at that point.
      In usual optimization application, this function would be generated by
      projecting the multivariate objective function along some specific
      direction. The direction is determined by some other procedure but should
      be a descent direction (i.e. the derivative of the projected univariate
      function must be negative at 0.).
    val_0: Instance of `_FnDFn` containing the value and gradient of the
      objective at 0. The gradient must be negative (i.e. must be a descent
      direction).
    initial_left: Instance of _FnDFn. The value and derivative of the function
      evaluated at the left end point of the bracketing interval.
    initial_right: Instance of _FnDFn. The value and derivative of the function
      evaluated at the right end point of the bracketing interval.
    f_lim: Scalar `Tensor` of float dtype.
    starting_iteration: Scalar integer `Tensor` of the same dtype as
      `max_iterations`. The number of iterations that have already been
      consumed by the bracketing.
    max_iterations: Positive scalar `Tensor` of integral dtype. The maximum
      number of iterations to perform in the line search. The number of
      iterations used to bracket the minimum are also counted against this
      parameter.
    sufficient_decrease_param: Positive scalar `Tensor` of real dtype.
      Bounded above by the curvature param. Corresponds to `delta` in the
      terminology of [Hager and Zhang (2006)][2].
    curvature_param: Positive scalar `Tensor` of real dtype. Bounded above
      by `1.`. Corresponds to 'sigma' in the terminology of
      [Hager and Zhang (2006)][2].
    shrinkage_param: Scalar positive Tensor of real dtype. Must be less than
      `1.`. Corresponds to the parameter `gamma` in [Hager and Zhang (2006)][2].

  Returns:
    A namedtuple containing the following fields.
      iteration: A scalar int32 `Tensor`. The number of iterations consumed.
      found_wolfe: A scalar boolean `Tensor`. Indicates whether a point
        satisfying the Wolfe conditions has been found. If this is True, the
        interval will be degenerate (i.e. left and right below
        will be identical).
      failed: A scalar boolean `Tensor`. Indicates if invalid function or
        gradient values were encountered (i.e. infinity or NaNs).
      num_evals: A scalar int32 `Tensor`. The total number of function
        evaluations made.
      left: Instance of _FnDFn. The position and the associated value and
        derivative at the updated left end point of the interval.
      right: Instance of _FnDFn. The position and the associated value and
        derivative at the updated right end point of the interval.
  """

  def _loop_cond(iteration, found_wolfe, failed, evals, val_left, val_right):  # pylint:disable=unused-argument
    """Loop condition."""
    eps = _next_after(val_right.x).eps
    interval_shrunk = (val_right.x - val_left.x) <= eps
    found_wolfe = tf.convert_to_tensor(found_wolfe)
    return  ((iteration < max_iterations) &
             ~(found_wolfe | failed | interval_shrunk))

  def _loop_body(iteration, found_wolfe, failed, evals, val_left, val_right):  # pylint:disable=unused-argument
    """The loop body."""
    iteration += 1

    secant2_result = _secant2(
        value_and_gradients_function,
        val_0,
        val_left,
        val_right,
        f_lim,
        sufficient_decrease_param=sufficient_decrease_param,
        curvature_param=curvature_param)

    evals += secant2_result.num_evals

    def _failed_fn():
      return _LineSearchInnerResult(
          iteration=iteration,
          found_wolfe=False,
          failed=True,
          num_evals=evals,
          left=val_left,
          right=val_right)

    def _found_wolfe_fn():
      return _LineSearchInnerResult(
          iteration=iteration,
          found_wolfe=True,
          failed=False,
          num_evals=evals,
          left=secant2_result.left,
          right=secant2_result.right)

    def _default_fn():
      """Default action."""
      new_width = secant2_result.right.x - secant2_result.left.x
      old_width = val_right.x - val_left.x
      sufficient_shrinkage = new_width < shrinkage_param * old_width

      def _sufficient_shrinkage_fn():
        """Action to perform if secant2 shrank the interval sufficiently."""
        func_is_flat = (
            (_next_after(val_left.f).nextfloat >= val_right.f) &
            (_next_after(secant2_result.left.f).nextfloat >=
             secant2_result.right.f))
        is_flat_retval = _LineSearchInnerResult(
            iteration=iteration,
            found_wolfe=True,
            failed=False,
            num_evals=evals,
            left=secant2_result.left,
            right=secant2_result.left)
        not_is_flat_retval = _LineSearchInnerResult(
            iteration=iteration,
            found_wolfe=False,
            failed=False,
            num_evals=evals,
            left=secant2_result.left,
            right=secant2_result.right)

        return tf.contrib.framework.smart_cond(
            func_is_flat,
            true_fn=lambda: is_flat_retval,
            false_fn=lambda: not_is_flat_retval)

      def _insufficient_shrinkage_fn():
        """Action to perform if secant2 didn't shrink the interval enough."""
        update_result = _line_search_inner_bisection(
            value_and_gradients_function,
            secant2_result.left,
            secant2_result.right,
            f_lim)
        return _LineSearchInnerResult(
            iteration=iteration,
            found_wolfe=False,
            failed=update_result.failed,
            num_evals=evals + update_result.num_evals,
            left=update_result.left,
            right=update_result.right)

      return tf.contrib.framework.smart_cond(
          sufficient_shrinkage,
          true_fn=_sufficient_shrinkage_fn,
          false_fn=_insufficient_shrinkage_fn)

    return tf.contrib.framework.smart_case([
        (secant2_result.failed, _failed_fn),
        (secant2_result.found_wolfe, _found_wolfe_fn)
    ], default=_default_fn, exclusive=False)

  initial_args = _LineSearchInnerResult(
      iteration=starting_iteration,
      found_wolfe=False,
      failed=False,
      num_evals=0,
      left=initial_left,
      right=initial_right)

  raw_results = tf.while_loop(
      _loop_cond,
      _loop_body,
      loop_vars=initial_args,
      parallel_iterations=1)
  # Check if we terminated because of interval being shrunk in which case
  # we return success.
  effective_wolfe = (
      raw_results.found_wolfe |  # Found Wolfe, or
      (
          ~tf.convert_to_tensor(raw_results.failed, name='failed')
          &  # We didn't fail and didn't exceed the iterations.
          (raw_results.iteration < max_iterations)))
  return _LineSearchInnerResult(
      iteration=raw_results.iteration,
      found_wolfe=effective_wolfe,
      failed=raw_results.failed,
      num_evals=raw_results.num_evals,
      left=raw_results.left,
      right=raw_results.right)


def _line_search_inner_bisection(
    value_and_gradients_function,
    val_left,
    val_right,
    f_lim):
  """Performs bisection and updates the interval."""

  midpoint = (val_left.x + val_right.x) / 2
  f_m, df_m = value_and_gradients_function(midpoint)

  val_mid = _FnDFn(x=midpoint, f=f_m, df=df_m)
  val_mid_finite = _is_finite(val_mid)

  def _success_fn():
    """Action to take if the midpoint evaluation succeeded."""
    update_result = _update(
        value_and_gradients_function,
        val_left,
        val_right,
        val_mid,
        f_lim)
    return _UpdateResult(
        failed=update_result.failed,
        num_evals=update_result.num_evals + 1,
        left=update_result.left,
        right=update_result.right)

  def _failed_fn():
    return _UpdateResult(
        failed=True,
        num_evals=1,
        left=val_left,
        right=val_right)

  return tf.contrib.framework.smart_cond(
      val_mid_finite,
      true_fn=_success_fn,
      false_fn=_failed_fn)


def _satisfies_wolfe(val_0,
                     val_c,
                     f_lim,
                     sufficient_decrease_param,
                     curvature_param):
  """Checks whether the Wolfe or approx Wolfe conditions are satisfied.

  The Wolfe conditions are a set of stopping criteria for an inexact line search
  algorithm. Let f(a) be the function value along the search direction and
  df(a) the derivative along the search direction evaluated a distance 'a'.
  Here 'a' is the distance along the search direction. The Wolfe conditions are:

    ```None
      f(a) <= f(0) + delta * a * df(0)   (Armijo/Sufficient decrease condition)
      df(a) >= sigma * df(0)             (Weak curvature condition)
    ```
  `delta` and `sigma` are two user supplied parameters satisfying:
   `0 < delta < sigma <= 1.`. In the following, delta is called
   `sufficient_decrease_param` and sigma is called `curvature_param`.

  On a finite precision machine, the Wolfe conditions are difficult to satisfy
  when one is close to the minimum. Hence, Hager-Zhang propose replacing
  the sufficient decrease condition with the following condition on the
  derivative in the vicinity of a minimum.

    ```None
      df(a) <= (2 * delta - 1) * df(0)  (Approx Wolfe sufficient decrease)
    ```
  This condition is only used if one is near the minimum. This is tested using

    ```None
      f(a) <= f(0) + epsilon * |f(0)|
    ```
  The following function checks both the Wolfe and approx Wolfe conditions.
  Here, `epsilon` is a small positive constant. In the following, the argument
  `f_lim` corresponds to the product: epsilon * |f(0)|.

  Args:
    val_0: Instance of _FnDFn. The function and derivative value at 0.
    val_c: Instance of _FnDFn. The function and derivative value at the
      point to be tested.
    f_lim: Scalar `Tensor` of real dtype. The function value threshold for
      the approximate Wolfe conditions to be checked.
    sufficient_decrease_param: Positive scalar `Tensor` of real dtype.
      Bounded above by the curvature param. Corresponds to 'delta' in the
      terminology of [Hager and Zhang (2006)][2].
    curvature_param: Positive scalar `Tensor` of real dtype. Bounded above
      by `1.`. Corresponds to 'sigma' in the terminology of
      [Hager Zhang (2005)][1].

  Returns:
    is_satisfied: A scalar boolean `Tensor` which is True if either the
      Wolfe or approximate Wolfe conditions are satisfied.
  """
  exact_wolfe_suff_dec = (sufficient_decrease_param * val_0.df >=
                          (val_c.f - val_0.f) / val_c.x)
  wolfe_curvature = val_c.df >= curvature_param * val_0.df
  exact_wolfe = exact_wolfe_suff_dec & wolfe_curvature
  approx_wolfe_applies = val_c.f <= f_lim
  approx_wolfe_suff_dec = ((2 * sufficient_decrease_param - 1) * val_0.df
                           >= val_c.df)
  approx_wolfe = approx_wolfe_applies & approx_wolfe_suff_dec & wolfe_curvature
  is_satisfied = exact_wolfe | approx_wolfe
  return is_satisfied


def _secant(a, b, dfa, dfb):
  """Returns the secant interpolation for the minimum.

  The secant method is a technique for finding roots of nonlinear functions.
  When finding the minimum, one applies the secant method to the derivative
  of the function.
  For an arbitrary function and a bounding interval, the secant approximation
  can produce the next point which is outside the bounding interval. However,
  with the assumption of opposite slope condtion on the interval [a,b] the new
  point c is always bracketed by [a,b]. Note that by assumption,
  f'(a) < 0 and f'(b) > 0.
  Hence c is a weighted average of a and b and thus always in [a, b].

  Args:
    a: A scalar real `Tensor`. The left end point of the initial interval.
    b: A scalar real `Tensor`. The right end point of the initial interval.
    dfa: A scalar real `Tensor`. The derivative of the function at the
      left end point (i.e. a).
    dfb: A scalar real `Tensor`. The derivative of the function at the
      right end point (i.e. b).

  Returns:
    approx_minimum: A scalar real `Tensor`. An approximation to the point
      at which the derivative vanishes.
  """
  return (a * dfb - b * dfa) / (dfb - dfa)


_Secant2Result = collections.namedtuple('_Secant2Results', [
    'found_wolfe',
    'failed',
    'num_evals',
    'left',
    'right'])


def _secant2(value_and_gradients_function,
             val_0,
             val_left,
             val_right,
             f_lim,
             sufficient_decrease_param=0.1,
             curvature_param=0.9,
             name=None):
  """Performs the secant square procedure of Hager Zhang.

  Given an interval that brackets a root, this procedure performs an update of
  both end points using two intermediate points generated using the secant
  interpolation. For details see the steps S1-S4 in [Hager and Zhang (2006)][2].

  The interval [a, b] must satisfy the opposite slope conditions described in
  the documentation for '_update'.

  Args:
    value_and_gradients_function: A Python callable that accepts a real
      scalar tensor and returns a tuple
      containing the value of the function and its derivative at that point.
    val_0: Instance of _FnDFn. The function and derivative value at 0.
    val_left: Instance of _FnDFn. The value and derivative of the function
      evaluated at the left end point of the bracketing interval (labelled 'a'
      above).
    val_right: Instance of _FnDFn. The value and derivative of the function
      evaluated at the right end point of the bracketing interval (labelled 'b'
      above).
    f_lim: Scalar `Tensor` of real dtype. The function value threshold for
      the approximate Wolfe conditions to be checked.
    sufficient_decrease_param: Positive scalar `Tensor` of real dtype.
      Bounded above by the curvature param. Corresponds to 'delta' in the
      terminology of [Hager and Zhang (2006)][2].
    curvature_param: Positive scalar `Tensor` of real dtype. Bounded above
      by `1.`. Corresponds to 'sigma' in the terminology of
      [Hager and Zhang (2006)][2].
    name: (Optional) Python str. The name prefixed to the ops created by this
      function. If not supplied, the default name 'secant2' is used.

  Returns:
    A namedtuple containing the following fields.
      found_wolfe: A scalar boolean `Tensor`. Indicates whether a point
        satisfying the Wolfe conditions has been found. If this is True, the
        interval will be degenerate (i.e. val_left_bar and val_right_bar below
        will be identical).
      failed: A scalar boolean `Tensor`. Indicates if invalid function or
        gradient values were encountered (i.e. infinity or NaNs).
      num_evals: A scalar int32 `Tensor`. The total number of function
        evaluations made.
      left: Instance of _FnDFn. The position and the associated value and
        derivative at the updated left end point of the interval.
      right: Instance of _FnDFn. The position and the associated value and
        derivative at the updated right end point of the interval.
  """
  with tf.name_scope(name, 'secant2',
                     [val_0,
                      val_left,
                      val_right,
                      f_lim,
                      sufficient_decrease_param,
                      curvature_param]):
    a, dfa = val_left.x, val_left.df
    b, dfb = val_right.x, val_right.df

    c = _secant(a, b, dfa, dfb)  # This will always be s.t. a <= c <= b
    fc, dfc = value_and_gradients_function(c)
    val_c = _FnDFn(x=c, f=fc, df=dfc)

    secant_failed = ~_is_finite(val_c)

    def _secant_failed_fn():
      return _Secant2Result(
          found_wolfe=False,
          failed=True,
          num_evals=1,
          left=val_left,
          right=val_right)

    secant_failed_case = secant_failed, _secant_failed_fn

    found_wolfe = _satisfies_wolfe(
        val_0,
        val_c,
        f_lim,
        sufficient_decrease_param=sufficient_decrease_param,
        curvature_param=curvature_param)

    def _found_wolfe_fn():
      return _Secant2Result(
          found_wolfe=True,
          failed=False,
          num_evals=1,
          left=val_c,
          right=val_c)

    found_wolfe_case = found_wolfe, _found_wolfe_fn

    def _default_fn():
      """Default action."""
      inner_result = _secant2_inner(
          value_and_gradients_function,
          val_0,
          val_c,
          val_left,
          val_right,
          f_lim,
          sufficient_decrease_param=sufficient_decrease_param,
          curvature_param=curvature_param)
      return _Secant2Result(
          found_wolfe=inner_result.found_wolfe,
          failed=inner_result.failed,
          num_evals=inner_result.num_evals + 1,
          left=inner_result.left,
          right=inner_result.right)

    return tf.contrib.framework.smart_case([
        secant_failed_case,
        found_wolfe_case
    ], default=_default_fn, exclusive=False)


def _secant2_inner(value_and_gradients_function,
                   val_0,
                   val_c,
                   val_left,
                   val_right,
                   f_lim,
                   sufficient_decrease_param=None,
                   curvature_param=None):
  """Helper function for secant square."""
  update_result = _update(value_and_gradients_function,
                          val_left,
                          val_right,
                          val_c,
                          f_lim)

  update_worked = ~tf.convert_to_tensor(update_result.failed)

  def _failed_fn():
    return _Secant2Result(
        found_wolfe=False,
        failed=True,
        num_evals=update_result.num_evals,
        left=val_left,
        right=val_right)

  def _success_fn():
    """Graph to execute when the update above succeeded."""

    def _do_secant(val_1, val_2):
      return _secant(val_1.x, val_2.x, val_1.df, val_2.df), True

    next_c, is_new = tf.contrib.framework.smart_case([
        (tf.equal(update_result.right.x, val_c.x),
         lambda: _do_secant(val_right, update_result.right)),
        (tf.equal(update_result.left.x, val_c.x),
         lambda: _do_secant(val_left, update_result.left))
    ], default=lambda: (val_c.x, False))

    in_range = ((update_result.left.x <= next_c) &
                (next_c <= update_result.right.x))
    in_range_and_new = in_range & is_new

    def in_range_and_new_fn():
      """Action to take when a new trial point is generated."""
      f_c, df_c = value_and_gradients_function(next_c)
      val_c = _FnDFn(x=next_c, f=f_c, df=df_c)
      inner_result = _secant2_inner_update(
          value_and_gradients_function,
          val_0,
          val_c,
          update_result.left,
          update_result.right,
          f_lim,
          check_stopping_condition=True,
          sufficient_decrease_param=sufficient_decrease_param,
          curvature_param=curvature_param)

      return _Secant2Result(
          found_wolfe=inner_result.found_wolfe,
          failed=inner_result.failed,
          num_evals=update_result.num_evals + inner_result.num_evals + 1,
          left=inner_result.left,
          right=inner_result.right)

    in_range_and_new_case = in_range_and_new, in_range_and_new_fn

    def _in_range_not_new_fn():
      """Action to take when no new trial point is generated."""
      inner_result = _secant2_inner_update(
          value_and_gradients_function,
          val_0,
          val_c,
          update_result.left,
          update_result.right,
          f_lim,
          check_stopping_condition=True,
          sufficient_decrease_param=sufficient_decrease_param,
          curvature_param=curvature_param)
      return _Secant2Result(
          found_wolfe=inner_result.found_wolfe,
          failed=inner_result.failed,
          num_evals=update_result.num_evals + inner_result.num_evals,
          left=inner_result.left,
          right=inner_result.right)

    in_range_not_new_case = in_range, _in_range_not_new_fn

    def _default_fn():
      return _Secant2Result(
          found_wolfe=False,
          failed=False,
          num_evals=update_result.num_evals,
          left=update_result.left,
          right=update_result.right)

    return tf.contrib.framework.smart_case([
        in_range_and_new_case,
        in_range_not_new_case,
    ], default=_default_fn)

  return tf.contrib.framework.smart_cond(
      update_worked,
      true_fn=_success_fn,
      false_fn=_failed_fn)


def _secant2_inner_update(value_and_gradients_function,
                          val_0,
                          val_c,
                          val_left,
                          val_right,
                          f_lim,
                          check_stopping_condition=True,
                          sufficient_decrease_param=None,
                          curvature_param=None):
  """Helper function for secant-square step."""
  def _default_fn():
    """Default Action."""
    update_result = _update(value_and_gradients_function,
                            val_left,
                            val_right,
                            val_c,
                            f_lim)
    return _Secant2Result(
        found_wolfe=False,
        failed=update_result.failed,
        num_evals=update_result.num_evals,
        left=update_result.left,
        right=update_result.right)

  if not check_stopping_condition:
    return _default_fn()

  def _secant_failed_fn():
    return _Secant2Result(
        found_wolfe=False,
        failed=True,
        num_evals=0,
        left=val_left,
        right=val_right)

  secant_failed = ~_is_finite(val_c)

  secant_failed_case = secant_failed, _secant_failed_fn

  found_wolfe = _satisfies_wolfe(
      val_0,
      val_c,
      f_lim,
      sufficient_decrease_param=sufficient_decrease_param,
      curvature_param=curvature_param)

  def _found_wolfe_fn():
    return _Secant2Result(
        found_wolfe=True,
        failed=False,
        num_evals=0,
        left=val_c,
        right=val_c)

  # If we have found a point satisfying the Wolfe conditions,
  # we have converged.
  found_wolfe_case = found_wolfe, _found_wolfe_fn

  return tf.contrib.framework.smart_case([
      secant_failed_case,
      found_wolfe_case
  ], default=_default_fn, exclusive=False)


_UpdateResult = collections.namedtuple('_UpdateResult', [
    'failed',     # Boolean indicating whether the update failed.
    'num_evals',  # The total number of objective evaluations performed.
    'left',       # The left end point (instance of _FnDFn).
    'right'       # The right end point (instance of _FnDFn).
])


def _update(value_and_gradients_function,
            val_left,
            val_right,
            val_trial,
            f_lim):
  """Squeezes a bracketing interval containing the minimum.

  Given an interval which brackets a minimum and a point in that interval,
  finds a smaller nested interval which also brackets the minimum. If the
  supplied point does not lie in the bracketing interval, the current interval
  is returned.

  The requirement of the interval bracketing a minimum is expressed through the
  opposite slope conditions. Assume the left end point is 'a', the right
  end point is 'b', the function to be minimized is 'f' and the derivative is
  'df'. The update procedure relies on the following conditions being satisfied:

  '''
    f(a) <= f(0) + epsilon   (1)
    df(a) < 0                (2)
    df(b) > 0                (3)
  '''

  In the first condition, epsilon is a small positive constant. The condition
  demands that the function at the left end point be not much bigger than the
  starting point (i.e. 0). This is an easy to satisfy condition because by
  assumption, we are in a direction where the function value is decreasing.
  The second and third conditions together demand that there is at least one
  zero of the derivative in between a and b.

  In addition to the interval, the update algorithm requires a third point to
  be supplied. Usually, this point would lie within the interval [a, b]. If the
  point is outside this interval, the current interval is returned. If the
  point lies within the interval, the behaviour of the function and derivative
  value at this point is used to squeeze the original interval in a manner that
  preserves the opposite slope conditions.

  For further details of this component, see the procedure U0-U3 on page 123 of
  the [Hager and Zhang (2006)][2] article.

  Note that this function does not explicitly verify whether the opposite slope
  conditions are satisfied for the supplied interval. It is assumed that this
  is so.

  Args:
    value_and_gradients_function: A Python callable that accepts a real scalar
      tensor and returns a tuple containing the value of the function and its
      derivative at that point.
    val_left: Instance of _FnDFn. The value and derivative of the function
      evaluated at the left end point of the bracketing interval (labelled 'a'
      above).
    val_right: Instance of _FnDFn. The value and derivative of the function
      evaluated at the right end point of the bracketing interval (labelled 'b'
      above).
    val_trial: Instance of _FnDFn. The value and derivative of the function
      evaluated at the trial point to be used to shrink the interval (labelled
      'c' above).
    f_lim: Scalar `Tensor` of real dtype. The function value threshold for
      the approximate Wolfe conditions to be checked.

  Returns:
    A namedtuple containing the following fields
    failed: A boolean scalar `Tensor` indicating whether the objective
      function failed to yield a finite value at the trial points.
    num_evals: A scalar int32 `Tensor`. The total number of function evaluations
      made.
    left: Instance of _FnDFn. The position and the associated value and
      derivative at the updated left end point of the interval.
    right: Instance of _FnDFn. The position and the associated value and
      derivative at the updated right end point of the interval.
  """
  left, right = val_left.x, val_right.x
  trial, f_trial, df_trial = val_trial.x, val_trial.f, val_trial.df

  # If the intermediate point is not in the interval, do nothing.
  def _out_of_range_fn():
    return _UpdateResult(
        failed=False,
        num_evals=0,
        left=val_left,
        right=val_right)

  is_out_of_range = (trial < left) | (trial > right)

  out_of_range_case = is_out_of_range, _out_of_range_fn

  # The new point is a valid right end point (has positive derivative).
  def _can_update_right_fn():
    return _UpdateResult(
        failed=False,
        num_evals=0,
        left=val_left,
        right=val_trial)

  can_update_right = (df_trial >= 0), _can_update_right_fn

  # The new point is a valid left end point because it has negative slope
  # and the value at the point is not too large.
  def _can_update_left_fn():
    return _UpdateResult(
        failed=False,
        num_evals=0,
        left=val_trial,
        right=val_right)

  can_update_left = (f_trial <= f_lim), _can_update_left_fn

  def _default_fn():
    """Default Action."""
    bisection_result = _bisect(value_and_gradients_function,
                               val_left,
                               val_trial,
                               f_lim)
    return _UpdateResult(
        failed=bisection_result.failed,
        num_evals=bisection_result.num_evals,
        left=bisection_result.left,
        right=bisection_result.right)

  return tf.contrib.framework.smart_case(
      [
          out_of_range_case,
          can_update_right,
          can_update_left
      ],
      default=_default_fn,
      exclusive=False)


_BisectionResult = collections.namedtuple('_BisectionResult', [
    'stopped',    # Boolean indicating whether bisection terminated gracefully.
    'failed',     # Boolean indicating whether objective evaluation failed.
    'num_evals',  # The total number of objective evaluations performed.
    'left',       # The left end point (instance of _FnDFn).
    'right'       # The right end point (instance of _FnDFn).
])


def _bisect(value_and_gradients_function,
            initial_left,
            initial_right,
            f_lim):
  """Bisects an interval and updates to satisfy opposite slope conditions.

  Corresponds to the step U3 in [Hager and Zhang (2006)][2].

  Args:
    value_and_gradients_function: A Python callable that accepts a real scalar
      tensor and returns a tuple of scalar tensors of real dtype containing
      the value of the function and its derivative at that point.
      In usual optimization application, this function would be generated by
      projecting the multivariate objective function along some specific
      direction. The direction is determined by some other procedure but should
      be a descent direction (i.e. the derivative of the projected univariate
      function must be negative at `0.`).
    initial_left: Instance of _FnDFn. The value and derivative of the function
      evaluated at the left end point of the current bracketing interval.
    initial_right: Instance of _FnDFn. The value and derivative of the function
      evaluated at the right end point of the current bracketing interval.
    f_lim: Scalar `Tensor` of real dtype. The function value threshold for
      the approximate Wolfe conditions to be checked.

  Returns:
    A namedtuple containing the following fields:
      stopped: A scalar boolean tensor. Indicates whether the bisection
        loop terminated normally (i.e. by finding an interval that satisfies
        the opposite slope conditions).
      failed: A scalar boolean tensor. Indicates whether the objective function
        failed to produce a finite value.
      num_evals: A scalar int32 tensor. The number of value and gradients
        function evaluations.
      final_left: Instance of _FnDFn. The value and derivative of the function
        evaluated at the left end point of the bracketing interval found.
      final_right: Instance of _FnDFn. The value and derivative of the function
        evaluated at the right end point of the bracketing interval found.
  """

  def _loop_cond(stopped, failed, evals, val_left, val_right):  # pylint:disable=unused-argument
    """Loop conditional."""
    stopped = tf.convert_to_tensor(stopped)  # Ensure it is a tensor so | works.
    eps = _next_after(val_right.x).eps
    # Factor of 2 needed because we are computing the midpoint in the loop
    # body and if the interval width falls belows twice the epsilon,
    # the mid point will be indistinguishable from the endpoints and we will
    # have an infinite loop.
    interval_too_small = (val_right.x - val_left.x) <= 2 * eps
    return ~(stopped | failed | interval_too_small)

  # The case where the proposed point has negative slope but the value is
  # too high. It is not a valid left end point but along with the current left
  # end point, it encloses another minima. The following loop tries to narrow
  # the interval so that it satisfies the opposite slope conditions.
  def _loop_body(stopped, failed, evals, val_left, val_right):  # pylint:disable=unused-argument
    """Updates the right end point to satisfy the opposite slope conditions."""
    val_left_x = val_left.x
    mid_pt = (val_left_x + val_right.x) / 2
    f_mid, df_mid = value_and_gradients_function(mid_pt)
    # The case conditions.
    val_mid = _FnDFn(x=mid_pt, f=f_mid, df=df_mid)
    evals += 1
    mid_failed = ~_is_finite(val_mid)

    def _failed_fn():
      return _BisectionResult(
          stopped=False,
          failed=True,
          num_evals=evals,
          left=val_left,
          right=val_right)

    failed_case = (mid_failed, _failed_fn)

    def _valid_right_fn():
      return _BisectionResult(
          stopped=True,
          failed=False,
          num_evals=evals,
          left=val_left,
          right=val_mid)

    # The new point can be a valid right end point.
    valid_right_case = (df_mid >= 0), _valid_right_fn

    # It is a valid left end pt.
    valid_left = (df_mid < 0) & (f_mid <= f_lim)

    # Note that we must return found = False in this case because our target
    # is to find a good right end point and improvements to the left end point
    # are coincidental. Hence the loop must continue until we exit via
    # the valid_right case.
    def _valid_left_fn():
      return _BisectionResult(
          stopped=False,
          failed=False,
          num_evals=evals,
          left=val_mid,
          right=val_right)

    valid_left_case = valid_left, _valid_left_fn

    # To be explicit, this action applies when the new point has a positive
    # slope but the function value at that point is too high. This is the
    # same situation with which we started the loop in the first place. Hence
    # we should just replace the old right end point and continue to loop.
    def _default_fn():
      return _BisectionResult(
          stopped=False,
          failed=False,
          num_evals=evals,
          left=val_left,
          right=val_mid)

    return tf.contrib.framework.smart_case([
        failed_case,
        valid_right_case,
        valid_left_case
    ], default=_default_fn, exclusive=False)

  initial_args = _BisectionResult(
      stopped=tf.convert_to_tensor(False),
      failed=False,
      num_evals=0,
      left=initial_left,
      right=initial_right)

  raw_results = tf.while_loop(_loop_cond,
                              _loop_body,
                              initial_args,
                              parallel_iterations=1)

  return _BisectionResult(
      stopped=(raw_results.stopped | ~raw_results.failed),
      failed=raw_results.failed,
      num_evals=raw_results.num_evals,
      left=raw_results.left,
      right=raw_results.right)


_BracketResult = collections.namedtuple('_BracketResult', [
    'iteration',  # Number of iterations taken to bracket.
    'bracketed',  # Boolean indicating whether bracketing succeeded.
    'failed',     # Boolean indicating whether objective evaluation failed.
    'num_evals',  # The total number of objective evaluations performed.
    'left',       # The left end point (instance of _FnDFn).
    'right'       # The right end point (instance of _FnDFn).
])


def _bracket(value_and_gradients_function,
             val_0,
             val_c,
             f_lim,
             max_iterations,
             expansion_param=5.0):
  """Brackets the minimum given an initial starting point.

  Applies the Hager Zhang bracketing algorithm to find an interval containing
  a region with points satisfying Wolfe conditions. Uses the supplied initial
  step size 'c' to find such an interval. The only condition on 'c' is that
  it should be positive. For more details see steps B0-B3 in
  [Hager and Zhang (2006)][2].

  Args:
    value_and_gradients_function: A Python callable that accepts a real scalar
      tensor and returns a tuple containing the value of the function and
      its derivative at that point.
    val_0: Instance of _FnDFn. The function and derivative value at 0.
    val_c: Instance of _FnDFn. The value and derivative of the function
      evaluated at the initial trial point (labelled 'c' above).
    f_lim: Scalar `Tensor` of real dtype. The function value threshold for
      the approximate Wolfe conditions to be checked.
    max_iterations: Int32 scalar `Tensor`. The maximum number of iterations
      permitted.
    expansion_param: Scalar positive `Tensor` of real dtype. Must be greater
      than `1.`. Used to expand the initial interval in case it does not bracket
      a minimum.

  Returns:
    A namedtuple with the following fields.
    iterations: An int32 scalar `Tensor`. The number of iterations performed.
      Bounded above by `max_iterations` parameter.
    bracketed: A boolean scalar `Tensor`. True if the minimum has been
      bracketed by the returned interval.
    failed: A boolean scalar `Tensor`. True if the an error was encountered
      while bracketing.
    num_evals: An int32 scalar `Tensor`. The number of times the objective was
      evaluated.
    left: Instance of _FnDFn. The position and the associated value and
      derivative at the updated left end point of the interval found.
    right: Instance of _FnDFn. The position and the associated value and
      derivative at the updated right end point of the interval found.
  """
  def _cond(iteration, bracketed, failed, *ignored_args):  # pylint:disable=unused-argument
    """Loop cond."""
    retval = tf.logical_not(bracketed | failed | (iteration >= max_iterations))
    return retval

  def _body(iteration, bracketed, failed, evals, val_left, val_right):  # pylint:disable=unused-argument
    """Loop body to find the bracketing interval."""
    iteration += 1

    def _not_finite_fn():
      return _BracketResult(iteration=iteration,
                            bracketed=False,
                            failed=True,
                            num_evals=evals,
                            left=val_left,
                            right=val_right)

    # Check that the function or gradient are finite and quit if they aren't.
    not_finite = ~_is_finite(val_left, val_right)

    case0 = (not_finite, _not_finite_fn)

    def _right_pt_found_fn():
      return _BracketResult(iteration=iteration,
                            bracketed=True,
                            failed=False,
                            num_evals=evals,
                            left=val_left,
                            right=val_right)

    # If the right point has an increasing derivative, then [left, right]
    # encloses a minimum and we are done.
    case1 = ((val_right.df >= 0), _right_pt_found_fn)

    # This case applies if the point has negative derivative (i.e. it is almost
    # suitable as a left endpoint.
    def _case2_fn():
      """Case 2."""
      bisection_result = _bisect(
          value_and_gradients_function,
          val_0,
          val_right,
          f_lim)
      return _BracketResult(
          iteration=iteration,
          bracketed=True,
          failed=bisection_result.failed,
          num_evals=evals + bisection_result.num_evals,
          left=bisection_result.left,
          right=bisection_result.right)

    case2 = (val_right.f > f_lim), _case2_fn

    def _default_fn():
      """Expansion."""
      next_right = expansion_param * val_right.x
      f_next_right, df_next_right = value_and_gradients_function(next_right)
      val_next_right = _FnDFn(x=next_right, f=f_next_right, df=df_next_right)
      failed = ~_is_finite(val_next_right)
      return _BracketResult(
          iteration=iteration,
          bracketed=False,
          failed=failed,
          num_evals=evals + 1,
          left=val_right,
          right=val_next_right)

    return tf.contrib.framework.smart_case(
        [
            case0,
            case1,
            case2
        ],
        default=_default_fn,
        exclusive=False)

  initial_vars = _BracketResult(
      iteration=tf.convert_to_tensor(0),
      bracketed=False,
      failed=False,
      num_evals=0,
      left=val_0,
      right=val_c)

  return tf.while_loop(
      _cond,
      _body,
      initial_vars,
      parallel_iterations=1)


def _is_finite(val_1, val_2=None):
  """Checks if the supplied values are finite.

  Args:
    val_1: Instance of _FnDFn. The function and derivative value.
    val_2: (Optional) Instance of _FnDFn. The function and derivative value.

  Returns:
    is_finite: Scalar boolean `Tensor` indicating whether the function value
      and the gradient in `val_1` (and optionally) in `val_2` are all finite.
  """
  val_1_finite = tf.is_finite(val_1.f) & tf.is_finite(val_1.df)
  if val_2 is not None:
    return val_1_finite & tf.is_finite(val_2.f) & tf.is_finite(val_2.df)
  return val_1_finite


def _prepare_args(value_and_gradients_function,
                  initial_step_size,
                  objective_at_initial_step_size,
                  grad_objective_at_initial_step_size,
                  objective_at_zero,
                  grad_objective_at_zero,
                  threshold_use_approximate_wolfe_condition):
  """Prepares the arguments for the line search initialization.

  Args:
    value_and_gradients_function: A Python callable that accepts a real
      scalar tensor and returns a tuple containing the value of the
      function and its derivative at that point.
    initial_step_size: Scalar positive `Tensor` of real dtype. The
      initial value to try to bracket the minimum. Default is `1.`.
      Note that this point need not necessarily bracket the minimum for the
      line search to work correctly. However, a good value for it will make the
      search converge faster.
    objective_at_initial_step_size: Scalar `Tensor` of real dtype.
      If supplied, the value of the function at `initial_step_size`. If None,
      it will be computed.
    grad_objective_at_initial_step_size: Scalar `Tensor` of real dtype.
      If supplied, the
      derivative of the  function at `initial_step_size`. If None, it
      will be computed.
    objective_at_zero: Scalar `Tensor` of real dtype. If supplied, the value
      of the function at `0.`. If it is None, it will be computed.
    grad_objective_at_zero: Scalar `Tensor` of real dtype. If supplied, the
      derivative of the  function at `0.`. If None, it will be computed.
    threshold_use_approximate_wolfe_condition: Scalar positive `Tensor` of
      real dtype. Corresponds to the parameter 'epsilon' in
      [Hager and Zhang (2006)][2]. Used to estimate the
      threshold at which the line search switches to approximate Wolfe
      conditions.

  Returns:
    val_0: An instance of `_FnDFn` containing the value and derivative of the
      function at `0.`.
    val_initial: An instance of `_FnDFn` containing the value and derivative of
      the function at `initial_step_size`.
    f_lim: Scalar `Tensor` of real dtype. The function value threshold for
      the approximate Wolfe conditions to be checked.
    eval_count: Scalar int32 `Tensor`. The number of target function
      evaluations made by this function.
  """
  eval_count = tf.convert_to_tensor(0)
  if initial_step_size is not None:
    initial_step_size = tf.convert_to_tensor(initial_step_size)
  else:
    initial_step_size = tf.convert_to_tensor(1.0, dtype=tf.float32)
  if (objective_at_initial_step_size is None or
      grad_objective_at_initial_step_size is None):
    (
        objective_at_initial_step_size,
        grad_objective_at_initial_step_size
    ) = value_and_gradients_function(initial_step_size)
    eval_count += 1
  val_initial = _FnDFn(x=initial_step_size,
                       f=objective_at_initial_step_size,
                       df=grad_objective_at_initial_step_size)
  x_0 = tf.zeros_like(initial_step_size)
  if objective_at_zero is not None:
    objective_at_zero = tf.convert_to_tensor(objective_at_zero)

  if grad_objective_at_zero is not None:
    grad_objective_at_zero = tf.convert_to_tensor(grad_objective_at_zero)

  if objective_at_zero is None or grad_objective_at_zero is None:
    (
        objective_at_zero,
        grad_objective_at_zero
    ) = value_and_gradients_function(x_0)
    eval_count += 1

  val_0 = _FnDFn(x=x_0, f=objective_at_zero, df=grad_objective_at_zero)
  f_lim = objective_at_zero + (threshold_use_approximate_wolfe_condition *
                               tf.abs(objective_at_zero))
  return val_0, val_initial, f_lim, eval_count


def _to_str(x):
  """Converts a bool tensor to a string with True/False values."""
  x = tf.convert_to_tensor(x)
  if x.dtype == tf.bool:
    return tf.where(x, tf.fill(x.shape, 'True'), tf.fill(x.shape, 'False'))
  return x


# A convenience function useful while debugging in the graph mode.
def _print(pass_through_tensor, values):
  """Wrapper for tf.Print which supports lists and namedtuples for printing."""
  flat_values = []
  for value in values:
    # Checks if it is a namedtuple.
    if hasattr(value, '_fields'):
      for field in value._fields:
        flat_values.extend([field, _to_str(getattr(value, field))])
      continue
    if isinstance(value, (list, tuple)):
      for v in value:
        flat_values.append(_to_str(v))
      continue
    flat_values.append(_to_str(value))
  return tf.Print(pass_through_tensor, flat_values)
