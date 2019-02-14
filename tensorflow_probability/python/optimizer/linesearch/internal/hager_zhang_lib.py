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

import tensorflow as tf


FnDFn = collections.namedtuple(
    'FnDFn', [
        'x',  # A scalar tensor of real dtype, the function point to be
              # evaluated.
        'f',  # A scalar tensor of real dtype, the value of the function at the
              # supplied point.
        'df',  # A scalar tensor of real dypt, the derivative of the function at
               # the supplied point.
        'full_result'  # Return value of the `value_and_gradients_function`
                       # at the left end point `left_pt`.
    ])


def _apply(value_and_gradients_function, x):
  """Evaluate the objective function and returns an `FnDFn` instance."""
  x = tf.convert_to_tensor(value=x)
  full_result = value_and_gradients_function(x)
  f, df = full_result.f, full_result.df
  return FnDFn(x=x, f=f, df=df, full_result=full_result)


_IntermediateResult = collections.namedtuple('_IntermediateResult', [
    'iteration',  # Number of iterations taken to bracket.
    'stopped',    # Boolean indicating whether bracketing/bisection terminated.
    'failed',     # Boolean indicating whether objective evaluation failed.
    'num_evals',  # The total number of objective evaluations performed.
    'left',       # The left end point (instance of FnDFn).
    'right'       # The right end point (instance of FnDFn).
])


def _val_where(cond, tval, fval):
  """Like tf.where but works on namedtuples."""
  if isinstance(tval, tf.Tensor):
    return tf.where(cond, tval, fval)
  elif isinstance(tval, tuple):
    cls = type(tval)
    return cls(*(_val_where(cond, t, f) for t, f in zip(tval, fval)))
  else:
    raise Exception(TypeError)


def update(value_and_gradients_function, val_left, val_right, val_trial, f_lim):
  """Squeezes a bracketing interval containing the minimum.

  Given an interval which brackets a minimum and a point in that interval,
  finds a smaller nested interval which also brackets the minimum. If the
  supplied point does not lie in the bracketing interval, the current interval
  is returned.

  The following description is given in terms of individual points evaluated on
  a line function to be minimized. Note, however, the implementation also
  accepts batches of points allowing to minimize multiple line functions at
  once. See details on the docstring of `value_and_gradients_function` below.

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
      Alternatively, the function may represent the batching of `n` such line
      functions (e.g. projecting a single multivariate objective function along
      `n` distinct directions at once) accepting n points as input, i.e. a
      tensor of shape [n], and the fields 'f' and 'df' in the returned
      namedtuple should each be a tensor of shape [n], with the corresponding
      function values and derivatives at the input points.
    val_left: Instance of _FnDFn. The value and derivative of the function
      evaluated at the left end point of the bracketing interval (labelled 'a'
      above).
    val_right: Instance of _FnDFn. The value and derivative of the function
      evaluated at the right end point of the bracketing interval (labelled 'b'
      above).
    val_trial: Instance of _FnDFn. The value and derivative of the function
      evaluated at the trial point to be used to shrink the interval (labelled
      'c' above).
    f_lim: real `Tensor` of shape [n]. The function value threshold for
      the approximate Wolfe conditions to be checked for each batch member.

  Returns:
    A namedtuple containing the following fields:
      iteration: An int32 scalar `Tensor`. The number of iterations performed
        by the bisect algorithm.
      stopped: A boolean `Tensor` of shape [n]. True for those batch members
        where the bisection algorithm terminated.
      failed: A boolean `Tensor` of shape [n]. True for those batch members
        where an error was encountered.
      num_evals: An int32 scalar `Tensor`. The number of times the objective
        function was evaluated.
      left: Instance of `FnDFn`. The position and the associated value and
        derivative at the updated left end point of the interval found.
      right: Instance of `FnDFn`. The position and the associated value and
        derivative at the updated right end point of the interval found.
  """
  # We should only update if the trial point is within the interval.
  within_range = (val_left.x < val_trial.x) & (val_trial.x < val_right.x)

  # The new point is a valid left end point if it has negative slope
  # and the value at the point is not too large.
  valid_left = (val_trial.df < 0) & (val_trial.f <= f_lim)

  # If the trial point has a negative slope but the value at that point
  # is too high, bisect can narrow down an interval between the current left
  # and the trial point.
  needs_bisect = within_range & (val_trial.df < 0) & (val_trial.f > f_lim)

  # Note that if `~valid_left` it is because either:
  # - the slope at the trial point is positive, so it is a valid right
  #   point, or
  # - the needs_bisect condition is true.
  # In both cases we want to keep the current left and replace right
  # with the trial point.
  left = _val_where(within_range & valid_left, val_trial, val_left)
  right = _val_where(within_range & ~valid_left, val_trial, val_right)

  bisect_args = _IntermediateResult(
      iteration=tf.convert_to_tensor(value=0),
      stopped=~needs_bisect,
      failed=tf.zeros_like(within_range),  # i.e. all false.
      num_evals=tf.convert_to_tensor(value=0),
      left=left,
      right=right)
  return _bisect(value_and_gradients_function, bisect_args, f_lim)


def bracket(value_and_gradients_function,
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
      tensor and returns a namedtuple containing the value filed `f` of the
      function and its derivative value field `df` at that point.
      Alternatively, the function may representthe batching of `n` such line
      functions (e.g. projecting a single multivariate objective function along
      `n` distinct directions at once) accepting n points as input, i.e. a
      tensor of shape [n], and return a tuple of two tensors of shape [n], the
      function values and the corresponding derivatives at the input points.
    val_0: Instance of `FnDFn`. The function and derivative value at 0.
    val_c: Instance of `FnDFn`. The value and derivative of the function
      evaluated at the initial trial point (labelled 'c' above).
    f_lim: real `Tensor` of shape [n]. The function value threshold for
      the approximate Wolfe conditions to be checked for each batch member.
    max_iterations: Int32 scalar `Tensor`. The maximum number of iterations
      permitted. The limit applies equally to all batch members.
    expansion_param: Scalar positive `Tensor` of real dtype. Must be greater
      than `1.`. Used to expand the initial interval in case it does not bracket
      a minimum.

  Returns:
    A namedtuple with the following fields.
      iteration: An int32 scalar `Tensor`. The number of iterations performed.
        Bounded above by `max_iterations` parameter.
      stopped: A boolean `Tensor` of shape [n]. True for those batch members
        where the algorithm terminated before reaching `max_iterations`.
      failed: A boolean `Tensor` of shape [n]. True for those batch members
        where an error was encountered during bracketing.
      num_evals: An int32 scalar `Tensor`. The number of times the objective
        function was evaluated.
      left: Instance of `FnDFn`. The position and the associated value and
        derivative at the updated left end point of the interval found.
      right: Instance of `FnDFn`. The position and the associated value and
        derivative at the updated right end point of the interval found.
  """
  # Fail if either of the initial points are not finite.
  failed = ~is_finite(val_0, val_c)

  # If the slope at `c` is positive, step B1 in [2], then the given initial
  # points already bracket a minimum.
  bracketed = val_c.df >= 0

  # Bisection is needed, step B2, if `c` almost works as left point but the
  # objective value is too high.
  needs_bisect = (val_c.df < 0) & (val_c.f > f_lim)

  # In these three cases bracketing is already `stopped` and there is no need
  # to perform further evaluations. Otherwise the bracketing loop is needed to
  # expand the interval, step B3, until the conditions are met.
  initial_args = _IntermediateResult(
      iteration=tf.convert_to_tensor(value=0),
      stopped=failed | bracketed | needs_bisect,
      failed=failed,
      num_evals=tf.convert_to_tensor(value=0),
      left=val_0,
      right=val_c)

  def _loop_cond(curr):
    return (curr.iteration <
            max_iterations) & ~tf.reduce_all(input_tensor=curr.stopped)

  def _loop_body(curr):
    """Main body of bracketing loop."""
    # The loop maintains the invariant that curr.stopped is true if we have
    # either: failed, successfully bracketed, or not yet bracketed but needs
    # bisect. On the only remaining case, step B3 in [2]. case we need to
    # expand and update the left/right values appropriately.
    new_right = _apply(value_and_gradients_function,
                       expansion_param * curr.right.x)
    left = _val_where(curr.stopped, curr.left, curr.right)
    right = _val_where(curr.stopped, curr.right, new_right)

    # Updated the failed, bracketed, and needs_bisect conditions.
    failed = curr.failed | ~is_finite(right)
    bracketed = right.df >= 0
    needs_bisect = (right.df < 0) & (right.f > f_lim)
    return [_IntermediateResult(
        iteration=curr.iteration + 1,
        stopped=failed | bracketed | needs_bisect,
        failed=failed,
        num_evals=curr.num_evals + 1,
        left=left,
        right=right)]

  bracket_result = tf.while_loop(
      cond=_loop_cond, body=_loop_body, loop_vars=[initial_args])[0]

  # For entries where bisect is still needed, mark them as not yet stopped,
  # reset the left point to 0, and run `_bisect` on them.
  needs_bisect = (
      (bracket_result.right.df < 0) & (bracket_result.right.f > f_lim))
  stopped = bracket_result.failed | ~needs_bisect
  left = _val_where(stopped, bracket_result.left, val_0)
  bisect_args = bracket_result._replace(stopped=stopped, left=left)
  return _bisect(value_and_gradients_function, bisect_args, f_lim)


def bisect(value_and_gradients_function,
           initial_left,
           initial_right,
           f_lim):
  """Bisects an interval and updates to satisfy opposite slope conditions.

  Corresponds to the step U3 in [Hager and Zhang (2006)][2].

  Args:
    value_and_gradients_function: A Python callable that accepts a real scalar
      tensor and returns a namedtuple containing the value filed `f` of the
      function and its derivative value field `df` at that point.
      Alternatively, the function may representthe batching of `n` such line
      functions (e.g. projecting a single multivariate objective function along
      `n` distinct directions at once) accepting n points as input, i.e. a
      tensor of shape [n], and return a tuple of two tensors of shape [n], the
      function values and the corresponding derivatives at the input points.
    initial_left: Instance of _FnDFn. The value and derivative of the function
      evaluated at the left end point of the current bracketing interval.
    initial_right: Instance of _FnDFn. The value and derivative of the function
      evaluated at the right end point of the current bracketing interval.
    f_lim: real `Tensor` of shape [n]. The function value threshold for
      the approximate Wolfe conditions to be checked for each batch member.

  Returns:
    A namedtuple containing the following fields:
      iteration: An int32 scalar `Tensor`. The number of iterations performed.
        Bounded above by `max_iterations` parameter.
      stopped: A boolean scalar `Tensor`. True if the bisect algorithm
        terminated.
      failed: A scalar boolean tensor. Indicates whether the objective function
        failed to produce a finite value.
      num_evals: A scalar int32 tensor. The number of value and gradients
        function evaluations.
      final_left: Instance of _FnDFn. The value and derivative of the function
        evaluated at the left end point of the bracketing interval found.
      final_right: Instance of _FnDFn. The value and derivative of the function
        evaluated at the right end point of the bracketing interval found.
  """
  failed = ~is_finite(initial_left, initial_right)
  needs_bisect = (initial_right.df < 0) & (initial_right.f > f_lim)
  bisect_args = _IntermediateResult(
      iteration=tf.convert_to_tensor(value=0),
      stopped=failed | ~needs_bisect,
      failed=failed,
      num_evals=tf.convert_to_tensor(value=0),
      left=initial_left,
      right=initial_right)
  return _bisect(value_and_gradients_function, bisect_args, f_lim)


def _bisect(value_and_gradients_function, initial_args, f_lim):
  """Actual implementation of bisect given initial_args in a _BracketResult."""
  def _loop_cond(curr):
    # TODO(b/112524024): Also take into account max_iterations.
    return ~tf.reduce_all(input_tensor=curr.stopped)

  def _loop_body(curr):
    """Narrow down interval to satisfy opposite slope conditions."""
    mid = _apply(value_and_gradients_function, (curr.left.x + curr.right.x) / 2)

    # Fail if function values at mid point are no longer finite; or left/right
    # points are so close to it that we can't distinguish them any more.
    failed = (curr.failed | ~is_finite(mid) |
              (mid.x == curr.left.x) | (mid.x == curr.right.x))

    # If mid point has a negative slope and the function value at that point is
    # small enough, we can use it as a new left end point to narrow down the
    # interval. If mid point has a positive slope, then we have found a suitable
    # right end point to bracket a minima within opposite slopes. Otherwise, the
    # mid point has a negative slope but the function value at that point is too
    # high to work as left end point, we are in the same situation in which we
    # started the loop so we just update the right end point and continue.
    to_update = ~(curr.stopped | failed)
    update_left = (mid.df < 0) & (mid.f <= f_lim)
    left = _val_where(to_update & update_left, mid, curr.left)
    right = _val_where(to_update & ~update_left, mid, curr.right)

    # We're done when the right end point has a positive slope.
    stopped = curr.stopped | failed | (right.df >= 0)

    return [_IntermediateResult(
        iteration=curr.iteration,
        stopped=stopped,
        failed=failed,
        num_evals=curr.num_evals + 1,
        left=left,
        right=right)]

  # The interval needs updating if the right end point has a negative slope and
  # the value of the function at that point is too high. It is not a valid left
  # end point but along with the current left end point, it encloses another
  # minima. The loop above tries to narrow the interval so that it satisfies the
  # opposite slope conditions.
  return tf.while_loop(
      cond=_loop_cond, body=_loop_body, loop_vars=[initial_args])[0]


def is_finite(val_1, val_2=None):
  """Checks if the supplied values are finite.

  Args:
    val_1: Instance of _FnDFn. The function and derivative value.
    val_2: (Optional) Instance of _FnDFn. The function and derivative value.

  Returns:
    is_finite: Scalar boolean `Tensor` indicating whether the function value
      and the gradient in `val_1` (and optionally) in `val_2` are all finite.
  """
  val_1_finite = tf.math.is_finite(val_1.f) & tf.math.is_finite(val_1.df)
  if val_2 is not None:
    return val_1_finite & tf.math.is_finite(val_2.f) & tf.math.is_finite(
        val_2.df)
  return val_1_finite
