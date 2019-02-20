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
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.optimizer.linesearch.internal import hager_zhang_lib as hzl

__all__ = [
    'hager_zhang',
]

# Container to hold the function value and the derivative at a given point.
# Each entry is a scalar tensor of real dtype. Used for internal data passing.
_FnDFn = hzl.FnDFn


def _machine_eps(dtype):
  """Returns the machine epsilon for the supplied dtype."""
  if isinstance(dtype, tf.DType):
    dtype = dtype.as_numpy_dtype()
  return np.finfo(dtype).eps


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
        'grad_objective_at_right_pt',  # The derivative of the function at the
                                       # right end point. If converged is True,
                                       # it is equal to
                                       # `grad_objective_at_left_pt`.
                                       # Otherwise it corresponds to the last
                                       # interval computed.
        'full_result'  # Return value of the `value_and_gradients_function`
                       # at the left end point `left_pt`.
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
    # Define value and gradient namedtuple
    ValueAndGradient = namedtuple('ValueAndGradient', ['f', 'df'])
    # Define a quadratic target with minimum at 1.3.
    def value_and_gradients_function(x):
      return ValueAndGradient(f=(x - 1.3) ** 2, df=2 * (x-1.3))
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
      tensor and returns an object that can be converted to a namedtuple.
      The namedtuple should have fields 'f' and 'df' that correspond to scalar
      tensors of real dtype containing the value of the function and its
      derivative at that point. The other namedtuple fields, if present,
      should be tensors or sequences (possibly nested) of tensors.
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

    valid_inputs = (
        hzl.is_finite(val_0) & (val_0.df < 0) & tf.math.is_finite(val_c_input.x)
        & (val_c_input.x > 0))

    def _invalid_inputs_fn():
      return HagerZhangLineSearchResult(
          converged=tf.convert_to_tensor(value=False, name='converged'),
          failed=tf.convert_to_tensor(value=True, name='failed'),
          func_evals=prepare_evals,
          iterations=tf.convert_to_tensor(value=0),
          left_pt=val_0.x,
          objective_at_left_pt=val_0.f,
          grad_objective_at_left_pt=val_0.df,
          right_pt=val_0.x,
          objective_at_right_pt=val_0.f,
          grad_objective_at_right_pt=val_0.df,
          full_result=val_0.full_result)

    def _valid_inputs_fn():
      """Performs bracketing and line search if inputs are valid."""
      # If the value or the gradient at the supplied step is not finite,
      # we attempt to repair it.
      step_size_too_large = ~(
          tf.math.is_finite(val_c_input.df) & tf.math.is_finite(val_c_input.f))

      def _is_too_large_fn():
        return _fix_step_size(value_and_gradients_function,
                              val_c_input,
                              step_size_shrink_param)

      val_c, fix_evals = prefer_static.cond(step_size_too_large,
                                            _is_too_large_fn,
                                            lambda: (val_c_input, 0))

      # Check if c is fixed now.
      valid_at_c = hzl.is_finite(val_c) & (val_c.x > 0)
      def _failure_fn():
        # If c is still not good, just return 0.
        return HagerZhangLineSearchResult(
            converged=tf.convert_to_tensor(value=True, name='converged'),
            failed=tf.convert_to_tensor(value=False, name='failed'),
            func_evals=prepare_evals + fix_evals,
            iterations=tf.convert_to_tensor(value=0),
            left_pt=val_0.x,
            objective_at_left_pt=val_0.f,
            grad_objective_at_left_pt=val_0.df,
            right_pt=val_0.x,
            objective_at_right_pt=val_0.f,
            grad_objective_at_right_pt=val_0.df,
            full_result=val_0.full_result)
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
        converged = tf.convert_to_tensor(
            value=result.found_wolfe, name='converged')
        return HagerZhangLineSearchResult(
            converged=converged,
            failed=tf.convert_to_tensor(value=result.failed, name='failed'),
            func_evals=result.num_evals + prepare_evals + fix_evals,
            iterations=result.iteration,
            left_pt=result.left.x,
            objective_at_left_pt=result.left.f,
            grad_objective_at_left_pt=result.left.df,
            right_pt=result.right.x,
            objective_at_right_pt=result.right.f,
            grad_objective_at_right_pt=result.right.df,
            full_result=result.left.full_result)
      return prefer_static.cond(valid_at_c,
                                true_fn=success_fn,
                                false_fn=_failure_fn)

    return prefer_static.cond(valid_inputs,
                              true_fn=_valid_inputs_fn,
                              false_fn=_invalid_inputs_fn)


def _fix_step_size(value_and_gradients_function,
                   val_c_input,
                   step_size_shrink_param):
  """Shrinks the input step size until the value and grad become finite."""
  # The maximum iterations permitted are determined as the number of halvings
  # it takes to reduce 1 to 0 in the given dtype.
  iter_max = np.ceil(-np.log2(_machine_eps(val_c_input.x.dtype)))
  def _cond(i, c, f_c, df_c, fdf_c):  # pylint: disable=unused-argument
    return (i < iter_max) & ~(tf.math.is_finite(f_c) & tf.math.is_finite(df_c))

  def _body(i, c, f_c, df_c, fdf_c):  # pylint: disable=unused-argument
    next_c = c * step_size_shrink_param
    res_at_c = value_and_gradients_function(next_c)
    next_f, next_df = res_at_c.f, res_at_c.df  # Extract value and gradient
    return (i + 1, next_c, next_f, next_df, res_at_c)

  evals, next_c, next_f, next_df, next_full_result = tf.while_loop(
      cond=_cond,
      body=_body,
      loop_vars=((0, val_c_input.x, val_c_input.f, val_c_input.df,
                  val_c_input.full_result)))
  return (_FnDFn(x=next_c, f=next_f, df=next_df, full_result=next_full_result),
          evals)


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
      tensor and returns an object that can be converted to a namedtuple.
      The namedtuple should have fields 'f' and 'df' that correspond to scalar
      tensors of real dtype containing the value of the function and its
      derivative at that point. The other namedtuple fields, if present,
      should be tensors or sequences (possibly nested) of tensors.
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
  bracket_result = hzl.bracket(
      value_and_gradients_function,
      val_0,
      val_c,
      f_lim,
      max_iterations,
      expansion_param=expansion_param)

  # If the bracketing failed, or we have already exhausted all the allowed
  # iterations, we return an error.
  failed = (bracket_result.failed |
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

  return prefer_static.cond(failed,
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
      tensor and returns an object that can be converted to a namedtuple.
      The namedtuple should have fields 'f' and 'df' that correspond to scalar
      tensors of real dtype containing the value of the function and its
      derivative at that point. The other namedtuple fields, if present,
      should be tensors or sequences (possibly nested) of tensors.
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
    interval_shrunk = tf.math.nextafter(val_left.x, val_right.x) >= val_right.x
    found_wolfe = tf.convert_to_tensor(value=found_wolfe)
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
            (tf.math.nextafter(val_left.f, val_right.f) >= val_right.f) &
            (tf.math.nextafter(secant2_result.left.f, secant2_result.right.f) >=
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

        return prefer_static.cond(func_is_flat,
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

      return prefer_static.cond(sufficient_shrinkage,
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
      cond=_loop_cond,
      body=_loop_body,
      loop_vars=initial_args,
      parallel_iterations=1)
  # Check if we terminated because of interval being shrunk in which case
  # we return success.
  effective_wolfe = (
      raw_results.found_wolfe |  # Found Wolfe, or
      (
          ~tf.convert_to_tensor(value=raw_results.failed, name='failed')
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
  result_mid = value_and_gradients_function(midpoint)
  f_mid, df_mid = result_mid.f, result_mid.df
  val_mid = _FnDFn(x=midpoint, f=f_mid, df=df_mid, full_result=result_mid)
  val_mid_finite = hzl.is_finite(val_mid)

  def _success_fn():
    """Action to take if the midpoint evaluation succeeded."""
    update_result = hzl.update(
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

  return prefer_static.cond(val_mid_finite,
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
    value_and_gradients_function: A Python callable that accepts a real scalar
      tensor and returns a namedtuple instance. The namedtuple should have
      fields 'f' and 'df' that correspond to scalar tensors of real dtype
      containing the value of the function and its derivative at that point. The
      rest namedtuple fields, if present, should be tensors or sequences
      (possibly nested) of tensors. In usual optimization application, this
      function would be generated by projecting the multivariate objective
      function along some specific direction. The direction is determined by
      some other procedure but should be a descent direction (i.e. the
      derivative of the projected univariate function must be negative at 0.).
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
    result_at_c = value_and_gradients_function(c)
    f_c, df_c = result_at_c.f, result_at_c.df
    val_c = _FnDFn(x=c, f=f_c, df=df_c, full_result=result_at_c)

    secant_failed = ~hzl.is_finite(val_c)

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
  update_result = hzl.update(value_and_gradients_function,
                             val_left,
                             val_right,
                             val_c,
                             f_lim)

  update_worked = ~update_result.failed

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
      result_next_c = value_and_gradients_function(next_c)
      f_next_c, df_next_c = result_next_c.f, result_next_c.df
      val_c = _FnDFn(x=next_c,
                     f=f_next_c,
                     df=df_next_c,
                     full_result=result_next_c)
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

  return prefer_static.cond(update_worked,
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
    update_result = hzl.update(value_and_gradients_function,
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

  secant_failed = ~hzl.is_finite(val_c)

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


def _prepare_args(value_and_gradients_function,
                  initial_step_size,
                  objective_at_initial_step_size,
                  grad_objective_at_initial_step_size,
                  objective_at_zero,
                  grad_objective_at_zero,
                  threshold_use_approximate_wolfe_condition):
  """Prepares the arguments for the line search initialization.

  Args:
    value_and_gradients_function: A Python callable that accepts a real scalar
      tensor and returns an object that can be converted to a namedtuple.
      The namedtuple should have fields 'f' and 'df' that correspond to scalar
      tensors of real dtype containing the value of the function and its
      derivative at that point. The other namedtuple fields, if present,
      should be tensors or sequences (possibly nested) of tensors.
      In usual optimization application, this function would be generated by
      projecting the multivariate objective function along some specific
      direction. The direction is determined by some other procedure but should
      be a descent direction (i.e. the derivative of the projected univariate
      function must be negative at 0.).
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
  eval_count = tf.convert_to_tensor(value=0)
  if initial_step_size is not None:
    initial_step_size = tf.convert_to_tensor(value=initial_step_size)
  else:
    initial_step_size = tf.convert_to_tensor(value=1.0, dtype=tf.float32)
  if (objective_at_initial_step_size is None or
      grad_objective_at_initial_step_size is None):
    res_at_initial_step_size = value_and_gradients_function(initial_step_size)
    (
        objective_at_initial_step_size,
        grad_objective_at_initial_step_size
    ) = (res_at_initial_step_size.f,
         res_at_initial_step_size.df)
    eval_count += 1
  val_initial = _FnDFn(x=initial_step_size,
                       f=objective_at_initial_step_size,
                       df=grad_objective_at_initial_step_size,
                       full_result=res_at_initial_step_size)

  x_0 = tf.zeros_like(initial_step_size)
  res_at_x_0 = value_and_gradients_function(x_0)
  if objective_at_zero is not None:
    objective_at_zero = tf.convert_to_tensor(value=objective_at_zero)

  if grad_objective_at_zero is not None:
    grad_objective_at_zero = tf.convert_to_tensor(value=grad_objective_at_zero)

  if objective_at_zero is None or grad_objective_at_zero is None:
    (
        objective_at_zero,
        grad_objective_at_zero
    ) = (res_at_x_0.f,
         res_at_x_0.df)
    eval_count += 1

  val_0 = _FnDFn(x=x_0, f=objective_at_zero,
                 df=grad_objective_at_zero,
                 full_result=res_at_x_0)

  f_lim = objective_at_zero + (threshold_use_approximate_wolfe_condition *
                               tf.abs(objective_at_zero))
  return val_0, val_initial, f_lim, eval_count


def _to_str(x):
  """Converts a bool tensor to a string with True/False values."""
  x = tf.convert_to_tensor(value=x)
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
  return tf.compat.v1.Print(pass_through_tensor, flat_values)
