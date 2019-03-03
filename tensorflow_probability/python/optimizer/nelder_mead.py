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
"""The Nelder Mead derivative free minimization algorithm.

Nelder Mead method is one of the most popular derivative free minimization
methods. For an optimization problem in `n`-dimensions it maintains a set of
`n+1` candidate solutions that span a non-degenerate simplex. It successively
modifies the simplex based on a set of moves (reflection, expansion, shrinkage
and contraction) using the function values at each of the vertices.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
import tensorflow as tf

from tensorflow_probability.python.internal import prefer_static

# Tolerance to check for floating point zeros.
_EPSILON = 1e-10


NelderMeadOptimizerResults = collections.namedtuple(
    'NelderMeadOptimizerResults', [
        'converged',  # Scalar boolean tensor indicating whether the minimum
                      # was found within tolerance.
        'num_objective_evaluations',  # The total number of objective
                                      # evaluations performed.
        'position',  # A tensor containing the last argument value found
                     # during the search. If the search converged, then
                     # this value is the argmin of the objective function.
        'objective_value',  # A tensor containing the value of the objective
                            # function at the `position`. If the search
                            # converged, then this is the (local) minimum of
                            # the objective function.
        'final_simplex',  # The last simplex constructed before stopping.
        'final_objective_values',  # The objective function evaluated at the
                                   # vertices of the final simplex.
        'initial_simplex',  # The initial simplex.
        'initial_objective_values',  # The values of the objective function
                                     # at the vertices of the initial simplex.
        'num_iterations'  # The number of iterations of the algorithm performed.
    ])


def minimize(objective_function,
             initial_simplex=None,
             initial_vertex=None,
             step_sizes=None,
             objective_at_initial_simplex=None,
             objective_at_initial_vertex=None,
             batch_evaluate_objective=False,
             func_tolerance=1e-8,
             position_tolerance=1e-8,
             parallel_iterations=1,
             max_iterations=None,
             reflection=None,
             expansion=None,
             contraction=None,
             shrinkage=None,
             name=None):
  """Minimum of the objective function using the Nelder Mead simplex algorithm.

  Performs an unconstrained minimization of a (possibly non-smooth) function
  using the Nelder Mead simplex method. Nelder Mead method does not support
  univariate functions. Hence the dimensions of the domain must be 2 or greater.
  For details of the algorithm, see
  [Press, Teukolsky, Vetterling and Flannery(2007)][1].

  Points in the domain of the objective function may be represented as a
  `Tensor` of general shape but with rank at least 1. The algorithm proceeds
  by modifying a full rank simplex in the domain. The initial simplex may
  either be specified by the user or can be constructed using a single vertex
  supplied by the user. In the latter case, if `v0` is the supplied vertex,
  the simplex is the convex hull of the set:

  ```None
  S = {v0} + {v0 + step_i * e_i}
  ```

  Here `e_i` is a vector which is `1` along the `i`-th axis and zero elsewhere
  and `step_i` is a characteristic length scale along the `i`-th axis. If the
  step size is not supplied by the user, a unit step size is used in every axis.
  Alternately, a single step size may be specified which is used for every
  axis. The most flexible option is to supply a bespoke step size for every
  axis.

  ### Usage:

  The following example demonstrates the usage of the Nelder Mead minimzation
  on a two dimensional problem with the minimum located at a non-differentiable
  point.

  ```python
    # The objective function
    def sqrt_quadratic(x):
      return tf.sqrt(tf.reduce_sum(x ** 2, axis=-1))

    start = tf.constant([6.0, -21.0])  # Starting point for the search.
    optim_results = tfp.optimizer.nelder_mead_minimize(
        sqrt_quadratic, initial_vertex=start, func_tolerance=1e-8,
        batch_evaluate_objective=True)

    with tf.Session() as session:
      results = session.run(optim_results)
      # Check that the search converged
      assert(results.converged)
      # Check that the argmin is close to the actual value.
      np.testing.assert_allclose(results.position, np.array([0.0, 0.0]),
                                 atol=1e-7)
      # Print out the total number of function evaluations it took.
      print ("Function evaluations: %d" % results.num_objective_evaluations)
  ```

  ### References:
  [1]: William Press, Saul Teukolsky, William Vetterling and Brian Flannery.
    Numerical Recipes in C++, third edition. pp. 502-507. (2007).
    http://numerical.recipes/cpppages/chap0sel.pdf

  [2]: Jeffrey Lagarias, James Reeds, Margaret Wright and Paul Wright.
    Convergence properties of the Nelder-Mead simplex method in low dimensions,
    Siam J. Optim., Vol 9, No. 1, pp. 112-147. (1998).
    http://www.math.kent.edu/~reichel/courses/Opt/reading.material.2/nelder.mead.pdf

  [3]: Fuchang Gao and Lixing Han. Implementing the Nelder-Mead simplex
    algorithm with adaptive parameters. Computational Optimization and
    Applications, Vol 51, Issue 1, pp 259-277. (2012).
    https://pdfs.semanticscholar.org/15b4/c4aa7437df4d032c6ee6ce98d6030dd627be.pdf

  Args:
    objective_function:  A Python callable that accepts a point as a
      real `Tensor` and returns a `Tensor` of real dtype containing
      the value of the function at that point. The function
      to be minimized. If `batch_evaluate_objective` is `True`, the callable
      may be evaluated on a `Tensor` of shape `[n+1] + s ` where `n` is
      the dimension of the problem and `s` is the shape of a single point
      in the domain (so `n` is the size of a `Tensor` representing a
      single point).
      In this case, the expected return value is a `Tensor` of shape `[n+1]`.
      Note that this method does not support univariate functions so the problem
      dimension `n` must be strictly greater than 1.
    initial_simplex: (Optional) `Tensor` of real dtype. The initial simplex to
      start the search. If supplied, should be a `Tensor` of shape `[n+1] + s`
      where `n` is the dimension of the problem and `s` is the shape of a
      single point in the domain. Each row (i.e. the `Tensor` with a given
      value of the first index) is interpreted as a vertex of a simplex and
      hence the rows must be affinely independent. If not supplied, an axes
      aligned simplex is constructed using the `initial_vertex` and
      `step_sizes`. Only one and at least one of `initial_simplex` and
      `initial_vertex` must be supplied.
    initial_vertex: (Optional) `Tensor` of real dtype and any shape that can
      be consumed by the `objective_function`. A single point in the domain that
      will be used to construct an axes aligned initial simplex.
    step_sizes: (Optional) `Tensor` of real dtype and shape broadcasting
      compatible with `initial_vertex`. Supplies the simplex scale along each
      axes. Only used if `initial_simplex` is not supplied. See description
      above for details on how step sizes and initial vertex are used to
      construct the initial simplex.
    objective_at_initial_simplex: (Optional) Rank `1` `Tensor` of real dtype
      of a rank `1` `Tensor`. The value of the objective function at the
      initial simplex. May be supplied only if `initial_simplex` is
      supplied. If not supplied, it will be computed.
    objective_at_initial_vertex: (Optional) Scalar `Tensor` of real dtype. The
      value of the objective function at the initial vertex. May be supplied
      only if the `initial_vertex` is also supplied.
    batch_evaluate_objective: (Optional) Python `bool`. If True, the objective
      function will be evaluated on all the vertices of the simplex packed
      into a single tensor. If False, the objective will be mapped across each
      vertex separately. Evaluating the objective function in a batch allows
      use of vectorization and should be preferred if the objective function
      allows it.
    func_tolerance: (Optional) Scalar `Tensor` of real dtype. The algorithm
      stops if the absolute difference between the largest and the smallest
      function value on the vertices of the simplex is below this number.
    position_tolerance: (Optional) Scalar `Tensor` of real dtype. The
      algorithm stops if the largest absolute difference between the
      coordinates of the vertices is below this threshold.
    parallel_iterations: (Optional) Positive integer. The number of iterations
      allowed to run in parallel.
    max_iterations: (Optional) Scalar positive `Tensor` of dtype `int32`.
      The maximum number of iterations allowed. If `None` then no limit is
      applied.
    reflection: (Optional) Positive Scalar `Tensor` of same dtype as
      `initial_vertex`. This parameter controls the scaling of the reflected
      vertex. See, [Press et al(2007)][1] for details. If not specified,
      uses the dimension dependent prescription of [Gao and Han(2012)][3].
    expansion: (Optional) Positive Scalar `Tensor` of same dtype as
      `initial_vertex`. Should be greater than `1` and `reflection`. This
      parameter controls the expanded scaling of a reflected vertex.
      See, [Press et al(2007)][1] for details. If not specified, uses the
      dimension dependent prescription of [Gao and Han(2012)][3].
    contraction: (Optional) Positive scalar `Tensor` of same dtype as
      `initial_vertex`. Must be between `0` and `1`. This parameter controls
      the contraction of the reflected vertex when the objective function at
      the reflected point fails to show sufficient decrease.
      See, [Press et al(2007)][1] for more details. If not specified, uses
      the dimension dependent prescription of [Gao and Han(2012][3].
    shrinkage: (Optional) Positive scalar `Tensor` of same dtype as
      `initial_vertex`. Must be between `0` and `1`. This parameter is the scale
      by which the simplex is shrunk around the best point when the other
      steps fail to produce improvements.
      See, [Press et al(2007)][1] for more details. If not specified, uses
      the dimension dependent prescription of [Gao and Han(2012][3].
    name: (Optional) Python str. The name prefixed to the ops created by this
      function. If not supplied, the default name 'minimize' is used.

  Returns:
    optimizer_results: A namedtuple containing the following items:
      converged: Scalar boolean tensor indicating whether the minimum was
        found within tolerance.
      num_objective_evaluations: The total number of objective
        evaluations performed.
      position: A `Tensor` containing the last argument value found
        during the search. If the search converged, then
        this value is the argmin of the objective function.
      objective_value: A tensor containing the value of the objective
        function at the `position`. If the search
        converged, then this is the (local) minimum of
        the objective function.
      final_simplex: The last simplex constructed before stopping.
      final_objective_values: The objective function evaluated at the
        vertices of the final simplex.
      initial_simplex: The starting simplex.
      initial_objective_values: The objective function evaluated at the
        vertices of the initial simplex.
      num_iterations: The number of iterations of the main algorithm body.

  Raises:
    ValueError: If any of the following conditions hold
      1. If none or more than one of `initial_simplex` and `initial_vertex` are
        supplied.
      2. If `initial_simplex` and `step_sizes` are both specified.
  """
  with tf.compat.v1.name_scope(name, 'minimize', [
      initial_simplex, initial_vertex, step_sizes, objective_at_initial_simplex,
      objective_at_initial_vertex, func_tolerance, position_tolerance
  ]):
    (
        dim,
        _,
        simplex,
        objective_at_simplex,
        num_evaluations
    ) = _prepare_args(objective_function,
                      initial_simplex,
                      initial_vertex,
                      step_sizes,
                      objective_at_initial_simplex,
                      objective_at_initial_vertex,
                      batch_evaluate_objective)
    domain_dtype = simplex.dtype
    (
        reflection,
        expansion,
        contraction,
        shrinkage
    ) = _resolve_parameters(dim,
                            reflection,
                            expansion,
                            contraction,
                            shrinkage,
                            domain_dtype)

    closure_kwargs = dict(
        objective_function=objective_function,
        dim=dim,
        func_tolerance=func_tolerance,
        position_tolerance=position_tolerance,
        batch_evaluate_objective=batch_evaluate_objective,
        reflection=reflection,
        expansion=expansion,
        contraction=contraction,
        shrinkage=shrinkage)

    def _loop_body(_, iterations, simplex, objective_at_simplex,
                   num_evaluations):
      (
          converged,
          next_simplex,
          next_objective,
          evaluations
      ) = nelder_mead_one_step(simplex, objective_at_simplex, **closure_kwargs)

      return (converged, iterations + 1, next_simplex, next_objective,
              num_evaluations + evaluations)

    initial_args = (False, 0, simplex, objective_at_simplex,
                    num_evaluations)
    # Loop until either we have converged or if the max iterations are supplied
    # then until we have converged or exhausted the available iteration budget.
    def _is_converged(converged, num_iterations, *ignored_args):  # pylint:disable=unused-argument
      # It is important to ensure that not_converged is a tensor. If
      # converged is not a tensor but a Python bool, then the overloaded
      # op '~' acts as bitwise complement so ~True = -2 and ~False = -1.
      # In that case, the loop will never terminate.
      not_converged = tf.logical_not(converged)
      return (not_converged if max_iterations is None
              else (not_converged & (num_iterations < max_iterations)))

    (converged, num_iterations, final_simplex, final_objective_values,
     final_evaluations) = tf.while_loop(
         cond=_is_converged,
         body=_loop_body,
         loop_vars=initial_args,
         parallel_iterations=parallel_iterations)
    order = tf.argsort(
        final_objective_values, direction='ASCENDING', stable=True)
    best_index = order[0]
    # The explicit cast to Tensor below is done to avoid returning a mixture
    # of Python types and Tensors which cause problems with session.run.
    # In the eager mode, converged may remain a Python bool. Trying to evaluate
    # the whole tuple in one evaluate call will raise an exception because
    # of the presence of non-tensors. This is very annoying so we explicitly
    # cast those arguments to Tensors.
    return NelderMeadOptimizerResults(
        converged=tf.convert_to_tensor(value=converged),
        num_objective_evaluations=final_evaluations,
        position=final_simplex[best_index],
        objective_value=final_objective_values[best_index],
        final_simplex=final_simplex,
        final_objective_values=final_objective_values,
        num_iterations=tf.convert_to_tensor(value=num_iterations),
        initial_simplex=simplex,
        initial_objective_values=objective_at_simplex)


def nelder_mead_one_step(current_simplex,
                         current_objective_values,
                         objective_function=None,
                         dim=None,
                         func_tolerance=None,
                         position_tolerance=None,
                         batch_evaluate_objective=False,
                         reflection=None,
                         expansion=None,
                         contraction=None,
                         shrinkage=None,
                         name=None):
  """A single iteration of the Nelder Mead algorithm."""
  with tf.compat.v1.name_scope(name, 'nelder_mead_one_step'):
    domain_dtype = current_simplex.dtype.base_dtype
    order = tf.argsort(
        current_objective_values, direction='ASCENDING', stable=True)
    (
        best_index,
        worst_index,
        second_worst_index
    ) = order[0], order[-1], order[-2]

    worst_vertex = current_simplex[worst_index]

    (
        best_objective_value,
        worst_objective_value,
        second_worst_objective_value
    ) = (
        current_objective_values[best_index],
        current_objective_values[worst_index],
        current_objective_values[second_worst_index]
    )

    # Compute the centroid of the face opposite the worst vertex.
    face_centroid = tf.reduce_sum(
        input_tensor=current_simplex, axis=0) - worst_vertex
    face_centroid /= tf.cast(dim, domain_dtype)

    # Reflect the worst vertex through the opposite face.
    reflected = face_centroid + reflection * (face_centroid - worst_vertex)
    objective_at_reflected = objective_function(reflected)

    num_evaluations = 1
    has_converged = _check_convergence(current_simplex,
                                       current_simplex[best_index],
                                       best_objective_value,
                                       worst_objective_value,
                                       func_tolerance,
                                       position_tolerance)
    def _converged_fn():
      return (True, current_simplex, current_objective_values, 0)
    case0 = has_converged, _converged_fn
    accept_reflected = (
        (objective_at_reflected < second_worst_objective_value) &
        (objective_at_reflected >= best_objective_value))
    accept_reflected_fn = _accept_reflected_fn(current_simplex,
                                               current_objective_values,
                                               worst_index,
                                               reflected,
                                               objective_at_reflected)
    case1 = accept_reflected, accept_reflected_fn
    do_expansion = objective_at_reflected < best_objective_value
    expansion_fn = _expansion_fn(objective_function,
                                 current_simplex,
                                 current_objective_values,
                                 worst_index,
                                 reflected,
                                 objective_at_reflected,
                                 face_centroid,
                                 expansion)
    case2 = do_expansion, expansion_fn
    do_outside_contraction = (
        (objective_at_reflected < worst_objective_value) &
        (objective_at_reflected >= second_worst_objective_value)
    )
    outside_contraction_fn = _outside_contraction_fn(
        objective_function,
        current_simplex,
        current_objective_values,
        face_centroid,
        best_index,
        worst_index,
        reflected,
        objective_at_reflected,
        contraction,
        shrinkage,
        batch_evaluate_objective)
    case3 = do_outside_contraction, outside_contraction_fn
    default_fn = _inside_contraction_fn(objective_function,
                                        current_simplex,
                                        current_objective_values,
                                        face_centroid,
                                        best_index,
                                        worst_index,
                                        worst_objective_value,
                                        contraction,
                                        shrinkage,
                                        batch_evaluate_objective)
    (
        converged,
        next_simplex,
        next_objective_at_simplex,
        case_evals) = prefer_static.case([case0, case1, case2, case3],
                                         default=default_fn, exclusive=False)
    next_simplex.set_shape(current_simplex.shape)
    next_objective_at_simplex.set_shape(current_objective_values.shape)
    return (
        converged,
        next_simplex,
        next_objective_at_simplex,
        num_evaluations + case_evals
    )


def _accept_reflected_fn(simplex,
                         objective_values,
                         worst_index,
                         reflected,
                         objective_at_reflected):
  """Creates the condition function pair for a reflection to be accepted."""
  def _replace_worst_with_reflected():
    next_simplex = _replace_at_index(simplex, worst_index, reflected)
    next_objective_values = _replace_at_index(objective_values, worst_index,
                                              objective_at_reflected)
    return False, next_simplex, next_objective_values, 0
  return _replace_worst_with_reflected


def _expansion_fn(objective_function,
                  simplex,
                  objective_values,
                  worst_index,
                  reflected,
                  objective_at_reflected,
                  face_centroid,
                  expansion):
  """Creates the condition function pair for an expansion."""
  def _expand_and_maybe_replace():
    """Performs the expansion step."""
    expanded = face_centroid + expansion * (reflected - face_centroid)
    expanded_objective_value = objective_function(expanded)
    expanded_is_better = (expanded_objective_value <
                          objective_at_reflected)
    accept_expanded_fn = lambda: (expanded, expanded_objective_value)
    accept_reflected_fn = lambda: (reflected, objective_at_reflected)
    next_pt, next_objective_value = prefer_static.cond(
        expanded_is_better, accept_expanded_fn, accept_reflected_fn)
    next_simplex = _replace_at_index(simplex, worst_index, next_pt)
    next_objective_at_simplex = _replace_at_index(objective_values,
                                                  worst_index,
                                                  next_objective_value)
    return False, next_simplex, next_objective_at_simplex, 1
  return _expand_and_maybe_replace


def _outside_contraction_fn(objective_function,
                            simplex,
                            objective_values,
                            face_centroid,
                            best_index,
                            worst_index,
                            reflected,
                            objective_at_reflected,
                            contraction,
                            shrinkage,
                            batch_evaluate_objective):
  """Creates the condition function pair for an outside contraction."""
  def _contraction():
    """Performs a contraction."""
    contracted = face_centroid + contraction * (reflected - face_centroid)
    objective_at_contracted = objective_function(contracted)
    is_contracted_acceptable = objective_at_contracted <= objective_at_reflected
    def _accept_contraction():
      next_simplex = _replace_at_index(simplex, worst_index, contracted)
      objective_at_next_simplex = _replace_at_index(
          objective_values,
          worst_index,
          objective_at_contracted)
      return (False,
              next_simplex,
              objective_at_next_simplex,
              1)

    def _reject_contraction():
      return _shrink_towards_best(objective_function,
                                  simplex,
                                  best_index,
                                  shrinkage,
                                  batch_evaluate_objective)

    return prefer_static.cond(is_contracted_acceptable,
                              _accept_contraction,
                              _reject_contraction)
  return _contraction


def _inside_contraction_fn(objective_function,
                           simplex,
                           objective_values,
                           face_centroid,
                           best_index,
                           worst_index,
                           worst_objective_value,
                           contraction,
                           shrinkage,
                           batch_evaluate_objective):
  """Creates the condition function pair for an inside contraction."""
  def _contraction():
    """Performs a contraction."""
    contracted = face_centroid - contraction * (face_centroid -
                                                simplex[worst_index])
    objective_at_contracted = objective_function(contracted)
    is_contracted_acceptable = objective_at_contracted <= worst_objective_value
    def _accept_contraction():
      next_simplex = _replace_at_index(simplex, worst_index, contracted)
      objective_at_next_simplex = _replace_at_index(
          objective_values,
          worst_index,
          objective_at_contracted)
      return (
          False,
          next_simplex,
          objective_at_next_simplex,
          1
      )

    def _reject_contraction():
      return _shrink_towards_best(objective_function, simplex, best_index,
                                  shrinkage, batch_evaluate_objective)

    return prefer_static.cond(is_contracted_acceptable,
                              _accept_contraction,
                              _reject_contraction)
  return _contraction


def _shrink_towards_best(objective_function,
                         simplex,
                         best_index,
                         shrinkage,
                         batch_evaluate_objective):
  """Shrinks the simplex around the best vertex."""

  # If the contraction step fails to improve the average objective enough,
  # the simplex is shrunk towards the best vertex.
  best_vertex = simplex[best_index]
  shrunk_simplex = best_vertex + shrinkage * (simplex - best_vertex)
  objective_at_shrunk_simplex, evals = _evaluate_objective_multiple(
      objective_function,
      shrunk_simplex,
      batch_evaluate_objective)
  return (False,
          shrunk_simplex,
          objective_at_shrunk_simplex,
          evals)


def _replace_at_index(x, index, replacement):
  """Replaces an element at supplied index."""
  x_new = tf.concat([x[:index], tf.expand_dims(replacement, axis=0),
                     x[(index + 1):]], axis=0)
  return x_new


def _check_convergence(simplex,
                       best_vertex,
                       best_objective,
                       worst_objective,
                       func_tolerance,
                       position_tolerance):
  """Returns True if the simplex has converged.

  If the simplex size is smaller than the `position_tolerance` or the variation
  of the function value over the vertices of the simplex is smaller than the
  `func_tolerance` return True else False.

  Args:
    simplex: `Tensor` of real dtype. The simplex to test for convergence. For
      more details, see the docstring for `initial_simplex` argument
      of `minimize`.
    best_vertex: `Tensor` of real dtype and rank one less than `simplex`. The
      vertex with the best (i.e. smallest) objective value.
    best_objective: Scalar `Tensor` of real dtype. The best (i.e. smallest)
      value of the objective function at a vertex.
    worst_objective: Scalar `Tensor` of same dtype as `best_objective`. The
      worst (i.e. largest) value of the objective function at a vertex.
    func_tolerance: Scalar positive `Tensor`. The tolerance for the variation
      of the objective function value over the simplex. If the variation over
      the simplex vertices is below this threshold, convergence is True.
    position_tolerance: Scalar positive `Tensor`. The algorithm stops if the
      lengths (under the supremum norm) of edges connecting to the best vertex
      are below this threshold.

  Returns:
    has_converged: A scalar boolean `Tensor` indicating whether the algorithm
      is deemed to have converged.
  """
  objective_convergence = tf.abs(worst_objective -
                                 best_objective) < func_tolerance
  simplex_degeneracy = tf.reduce_max(
      input_tensor=tf.abs(simplex - best_vertex)) < position_tolerance
  return objective_convergence | simplex_degeneracy


def _prepare_args(objective_function,
                  initial_simplex,
                  initial_vertex,
                  step_sizes,
                  objective_at_initial_simplex,
                  objective_at_initial_vertex,
                  batch_evaluate_objective):
  """Computes the initial simplex and the objective values at the simplex.

  Args:
    objective_function:  A Python callable that accepts a point as a
      real `Tensor` and returns a `Tensor` of real dtype containing
      the value of the function at that point. The function
      to be evaluated at the simplex. If `batch_evaluate_objective` is `True`,
      the callable may be evaluated on a `Tensor` of shape `[n+1] + s `
      where `n` is the dimension of the problem and `s` is the shape of a
      single point in the domain (so `n` is the size of a `Tensor`
      representing a single point).
      In this case, the expected return value is a `Tensor` of shape `[n+1]`.
    initial_simplex: None or `Tensor` of real dtype. The initial simplex to
      start the search. If supplied, should be a `Tensor` of shape `[n+1] + s`
      where `n` is the dimension of the problem and `s` is the shape of a
      single point in the domain. Each row (i.e. the `Tensor` with a given
      value of the first index) is interpreted as a vertex of a simplex and
      hence the rows must be affinely independent. If not supplied, an axes
      aligned simplex is constructed using the `initial_vertex` and
      `step_sizes`. Only one and at least one of `initial_simplex` and
      `initial_vertex` must be supplied.
    initial_vertex: None or `Tensor` of real dtype and any shape that can
      be consumed by the `objective_function`. A single point in the domain that
      will be used to construct an axes aligned initial simplex.
    step_sizes: None or `Tensor` of real dtype and shape broadcasting
      compatible with `initial_vertex`. Supplies the simplex scale along each
      axes. Only used if `initial_simplex` is not supplied. See the docstring
      of `minimize` for more details.
    objective_at_initial_simplex: None or rank `1` `Tensor` of real dtype.
      The value of the objective function at the initial simplex.
      May be supplied only if `initial_simplex` is
      supplied. If not supplied, it will be computed.
    objective_at_initial_vertex: None or scalar `Tensor` of real dtype. The
      value of the objective function at the initial vertex. May be supplied
      only if the `initial_vertex` is also supplied.
    batch_evaluate_objective: Python `bool`. If True, the objective function
      will be evaluated on all the vertices of the simplex packed into a
      single tensor. If False, the objective will be mapped across each
      vertex separately.

  Returns:
    prepared_args: A tuple containing the following elements:
      dimension: Scalar `Tensor` of `int32` dtype. The dimension of the problem
        as inferred from the supplied arguments.
      num_vertices: Scalar `Tensor` of `int32` dtype. The number of vertices
        in the simplex.
      simplex: A `Tensor` of same dtype as `initial_simplex`
        (or `initial_vertex`). The first component of the shape of the
        `Tensor` is `num_vertices` and each element represents a vertex of
        the simplex.
      objective_at_simplex: A `Tensor` of same dtype as the dtype of the
        return value of objective_function. The shape is a vector of size
        `num_vertices`. The objective function evaluated at the simplex.
      num_evaluations: An `int32` scalar `Tensor`. The number of points on
        which the objective function was evaluated.

  Raises:
    ValueError: If any of the following conditions hold
      1. If none or more than one of `initial_simplex` and `initial_vertex` are
        supplied.
      2. If `initial_simplex` and `step_sizes` are both specified.
  """
  if objective_at_initial_simplex is not None and initial_simplex is None:
    raise ValueError('`objective_at_initial_simplex` specified but the'
                     '`initial_simplex` was not.')

  if objective_at_initial_vertex is not None and initial_vertex is None:
    raise ValueError('`objective_at_initial_vertex` specified but the'
                     '`initial_vertex` was not.')

  # The full simplex was specified.
  if initial_simplex is not None:
    if initial_vertex is not None:
      raise ValueError('Both `initial_simplex` and `initial_vertex` specified.'
                       ' Only one of the two should be specified.')

    if step_sizes is not None:
      raise ValueError('`step_sizes` must not be specified when an'
                       ' `initial_simplex` has been specified.')
    return _prepare_args_with_initial_simplex(objective_function,
                                              initial_simplex,
                                              objective_at_initial_simplex,
                                              batch_evaluate_objective)

  if initial_vertex is None:
    raise ValueError('One of `initial_simplex` or `initial_vertex`'
                     ' must be supplied')

  if step_sizes is None:
    step_sizes = _default_step_sizes(initial_vertex)

  return _prepare_args_with_initial_vertex(objective_function,
                                           initial_vertex,
                                           step_sizes,
                                           objective_at_initial_vertex,
                                           batch_evaluate_objective)


def _default_step_sizes(reference_vertex):
  """Chooses default step sizes according to [Gao and Han(2010)][3]."""
  # Step size to choose when the coordinate is zero.
  small_sizes = tf.ones_like(reference_vertex) * 0.00025
  # Step size to choose when the coordinate is non-zero.
  large_sizes = reference_vertex * 0.05
  return tf.where(tf.abs(reference_vertex) < _EPSILON,
                  small_sizes,
                  large_sizes)


def _prepare_args_with_initial_simplex(objective_function,
                                       initial_simplex,
                                       objective_at_initial_simplex,
                                       batch_evaluate_objective):
  """Evaluates the objective function at the specified initial simplex."""
  initial_simplex = tf.convert_to_tensor(value=initial_simplex)

  # If d is the dimension of the problem, the number of vertices in the
  # simplex should be d+1. From this, we can infer the number of dimensions
  # as n - 1 where n is the number of vertices specified.
  num_vertices = tf.shape(input=initial_simplex)[0]
  dim = num_vertices - 1
  num_evaluations = 0

  if objective_at_initial_simplex is None:
    objective_at_initial_simplex, n_evals = _evaluate_objective_multiple(
        objective_function, initial_simplex, batch_evaluate_objective)
    num_evaluations += n_evals
  objective_at_initial_simplex = tf.convert_to_tensor(
      value=objective_at_initial_simplex)
  return (dim,
          num_vertices,
          initial_simplex,
          objective_at_initial_simplex,
          num_evaluations)


def _prepare_args_with_initial_vertex(objective_function,
                                      initial_vertex,
                                      step_sizes,
                                      objective_at_initial_vertex,
                                      batch_evaluate_objective):
  """Constructs a standard axes aligned simplex."""
  dim = tf.size(input=initial_vertex)
  num_vertices = dim + 1
  unit_vectors_along_axes = tf.reshape(
      tf.eye(dim, dim, dtype=initial_vertex.dtype.base_dtype),
      tf.concat([[dim], tf.shape(input=initial_vertex)], axis=0))

  # If step_sizes does not broadcast to initial_vertex, the multiplication
  # in the second term will fail.
  simplex_face = initial_vertex + step_sizes * unit_vectors_along_axes
  simplex = tf.concat([tf.expand_dims(initial_vertex, axis=0),
                       simplex_face], axis=0)
  num_evaluations = 0
  # Evaluate the objective function at the simplex vertices.
  if objective_at_initial_vertex is None:
    objective_at_initial_vertex = objective_function(initial_vertex)
    num_evaluations += 1

  objective_at_simplex_face, num_evals = _evaluate_objective_multiple(
      objective_function, simplex_face, batch_evaluate_objective)
  num_evaluations += num_evals

  objective_at_simplex = tf.concat(
      [
          tf.expand_dims(objective_at_initial_vertex, axis=0),
          objective_at_simplex_face
      ], axis=0)

  return (dim,
          num_vertices,
          simplex,
          objective_at_simplex,
          num_evaluations)


def _resolve_parameters(dim,
                        reflection,
                        expansion,
                        contraction,
                        shrinkage,
                        dtype):
  """Applies the [Gao and Han][3] presciption to the unspecified parameters."""
  dim = tf.cast(dim, dtype=dtype)
  reflection = 1. if reflection is None else reflection
  expansion = (1. + 2. / dim) if expansion is None else expansion
  contraction = (0.75 - 1. / (2 * dim)) if contraction is None else contraction
  shrinkage = (1. - 1. / dim) if shrinkage is None else shrinkage
  return reflection, expansion, contraction, shrinkage


def _evaluate_objective_multiple(objective_function, arg_batch,
                                 batch_evaluate_objective):
  """Evaluates the objective function on a batch of points.

  If `batch_evaluate_objective` is True, returns
  `objective function(arg_batch)` else it maps the `objective_function`
  across the `arg_batch`.

  Args:
    objective_function: A Python callable that accepts a single `Tensor` of
      rank 'R > 1' and any shape 's' and returns a scalar `Tensor` of real dtype
      containing the value of the function at that point. If
      `batch a `Tensor` of shape `[batch_size] + s ` where `batch_size` is the
      size of the batch of args. In this case, the expected return value is a
      `Tensor` of shape `[batch_size]`.
    arg_batch: A `Tensor` of real dtype. The batch of arguments at which to
      evaluate the `objective_function`. If `batch_evaluate_objective` is False,
      `arg_batch` will be unpacked along the zeroth axis and the
      `objective_function` will be applied to each element.
    batch_evaluate_objective: `bool`. Whether the `objective_function` can
      evaluate a batch of arguments at once.

  Returns:
    A tuple containing:
      objective_values: A `Tensor` of real dtype and shape `[batch_size]`.
        The value of the objective function evaluated at the supplied
        `arg_batch`.
      num_evaluations: An `int32` scalar `Tensor`containing the number of
        points on which the objective function was evaluated (i.e `batch_size`).
  """
  n_points = tf.shape(input=arg_batch)[0]
  if batch_evaluate_objective:
    return objective_function(arg_batch), n_points
  return tf.map_fn(objective_function, arg_batch), n_points
