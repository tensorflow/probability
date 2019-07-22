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
"""The differential evolution global optimization algorithm.

Differential evolution (DE) is a population-based global optimization scheme.
It is applicable to problems with a continuous parameter space. Because it
does not require computing gradients, it is also applicable to
non-differentiable functions. For more details see:
https://en.wikipedia.org/wiki/Differential_evolution

DE starts with a population of candidate solutions (represented as vectors).
It generates new trial solutions by a combination of
- "mutation", namely adding the weighted difference between
  two population vectors to a target vector, and
- "crossover", namely mixing the target with the content of a third.

If the trial vector thus constructed yields a lower cost value it
replaces the target vector it was made from. This process is repeated for
each member of the population to complete one generation.

There are a number of different schemes that fall under the DE umbrella.
The established notation for representing these schemes
is as `DE/x/y/z` where `x` specifies how the population member to be mutated is
selected (may be `rand` if it is chosen randomly or `best` if the best
member is chosen), `y` specifies the number of difference vectors used and `z`
denotes the crossover scheme employed. This may be `bin` if binary
recombination is used or `exp` if exponential crossover is used.

The most commonly employed scheme is `DE/rand/1/bin`. This is the one
implemented in this module.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tensorflow as tf

from tensorflow_probability.python import distributions


_DifferentialEvolutionOptimizerResults = collections.namedtuple(
    '_DifferentialEvolutionOptimizerResults', [
        'converged',  # Scalar boolean tensor indicating whether the minimum
                      # was found within tolerance.
        'failed',  # Scalar boolean tensor indicating whether the search failed.
                   # This may happen if the objective values become NaN for
                   # the entire population.
        'position',  # A list of tensors containing the best argument value
                     # found during the search. If the search converged, then
                     # this value is the argmin of the objective function.
        'objective_value',  # A tensor containing the value of the objective
                            # function at the `position`. If the search
                            # converged, then this is the (local) minimum of
                            # the objective function.
        'final_population',  # The final state of the population.
        'final_objective_values',  # The objective function evaluated at the
                                   # final population.
        'initial_population',  # The starting population.
        'initial_objective_values',  # The objective function evaluated at the
                                     # initial population.
        'num_iterations'  # The number of generations the population was
                          # evolved.
    ])


class DifferentialEvolutionOptimizerResults(
    _DifferentialEvolutionOptimizerResults):
  """Results of a differential evolution optimization run.

  The object has the following attributes:
    converged: Scalar boolean `Tensor` indicating whether the minimum was
      found within the specified tolerances.
    failed: Scalar boolean tensor indicating whether the search failed.
      This may happen if the objective values become NaN for the entire
      population.
    position: A `Tensor` containing the best point found during the search.
      If the search converged, then this value is the argmin of the
      objective function within the specified tolerances.
    objective_value: A tensor containing the value of the objective
      function at the `position`. If the search
      converged, then this is the (local) minimum of
      the objective function.
    final_population: The final state of the population.
    final_objective_values: The objective function evaluated at the
      final population.
    initial_population: The starting population.
    initial_objective_values: The objective function evaluated at the
      initial population.
    num_iterations: The number of generations the population was evolved.
  """


# Class to keep track of the loop variables in the minimize method.
_MinimizeLoopVars = collections.namedtuple(
    '_MinimizeLoopVars',
    [
        'converged',
        'failed',
        'num_iterations',
        'population',
        'population_values'
    ])


def one_step(
    objective_function,
    population,
    population_values=None,
    differential_weight=0.5,
    crossover_prob=0.9,
    seed=None,
    name=None):
  """Performs one step of the differential evolution algorithm.

  Args:
    objective_function:  A Python callable that accepts a batch of possible
      solutions and returns the values of the objective function at those
      arguments as a rank 1 real `Tensor`. This specifies the function to be
      minimized. The input to this callable may be either a single `Tensor`
      or a Python `list` of `Tensor`s. The signature must match the format of
      the argument `population`. (i.e., objective_function(*population) must
      return the value of the function to be minimized).
    population:  `Tensor` or Python `list` of `Tensor`s representing the
      current population vectors. Each `Tensor` must be of the same real dtype.
      The first dimension indexes individual population members while the
      rest of the dimensions are consumed by the value function. For example,
      if the population is a single `Tensor` of shape [n, m1, m2], then `n` is
      the population size and the output of `objective_function` applied to the
      population is a `Tensor` of shape [n]. If the population is a python
      list of `Tensor`s then each `Tensor` in the list should have the first
      axis of a common size, say `n` and `objective_function(*population)`
      should return a `Tensor of shape [n]. The population must have at least
      4 members for the algorithm to work correctly.
    population_values: A `Tensor` of rank 1 and real dtype. The result of
      applying `objective_function` to the `population`. If not supplied it is
      computed using the `objective_function`.
      Default value: None.
    differential_weight: Real scalar `Tensor`. Must be positive and less than
      2.0. The parameter controlling the strength of mutation.
      Default value: 0.5
    crossover_prob: Real scalar `Tensor`. Must be between 0 and 1. The
      probability of recombination per site.
      Default value: 0.9
    seed: `int` or None. The random seed for this `Op`. If `None`, no seed is
      applied.
      Default value: None.
    name: (Optional) Python str. The name prefixed to the ops created by this
      function. If not supplied, the default name 'one_step' is
      used.
      Default value: None

  Returns:
    A sequence containing the following elements (in order):
    next_population: A `Tensor` or Python `list` of `Tensor`s of the same
      structure as the input population. The population at the next generation.
    next_population_values: A `Tensor` of same shape and dtype as input
      `population_values`. The function values for the `next_population`.
  """
  with tf.compat.v1.name_scope(
      name, 'one_step',
      [population, population_values, differential_weight, crossover_prob]):
    population, _ = _ensure_list(population)
    if population_values is None:
      population_values = objective_function(*population)
    population_size = tf.shape(input=population[0])[0]
    seed_stream = distributions.SeedStream(seed, salt='one_step')
    mixing_indices = _get_mixing_indices(population_size, seed=seed_stream())
    # Construct the mutated solution vectors. There is one for each member of
    # the population.
    mutants = _get_mutants(population,
                           population_size,
                           mixing_indices,
                           differential_weight)
    # Perform recombination between the parents and the mutants.
    candidates = _binary_crossover(population,
                                   population_size,
                                   mutants,
                                   crossover_prob,
                                   seed=seed_stream())
    candidate_values = objective_function(*candidates)
    if population_values is None:
      population_values = objective_function(*population)

    infinity = tf.zeros_like(population_values) + np.inf

    population_values = tf.compat.v1.where(
        tf.math.is_nan(population_values), x=infinity, y=population_values)

    to_replace = candidate_values < population_values
    next_population = [
        tf.compat.v1.where(to_replace, x=candidates_part, y=population_part)
        for candidates_part, population_part in zip(candidates, population)
    ]
    next_values = tf.compat.v1.where(
        to_replace, x=candidate_values, y=population_values)

  return next_population, next_values


def minimize(objective_function,
             initial_population=None,
             initial_position=None,
             population_size=50,
             population_stddev=1.,
             max_iterations=100,
             func_tolerance=0,
             position_tolerance=1e-8,
             differential_weight=0.5,
             crossover_prob=0.9,
             seed=None,
             name=None):
  """Applies the Differential evolution algorithm to minimize a function.

  Differential Evolution is an evolutionary optimization algorithm which works
  on a set of candidate solutions called the population. It iteratively
  improves the population by applying genetic operators of mutation and
  recombination. The objective function `f` supplies the fitness of each
  candidate. A candidate `s_1` is considered better than `s_2` if
  `f(s_1) < f(s_2)`.

  This method allows the user to either specify an initial population or a
  single candidate solution. If a single solution is specified, a population
  of the specified size is initialized by adding independent normal noise
  to the candidate solution.

  The implementation also supports a multi-part specification of the state. For
  example, consider the objective function:

  ```python
  # x is a tensor of shape [n, m] while y is of shape [n].
  def objective(x, y):
    return tf.math.reduce_sum(x ** 2, axis=-1) + y ** 2
  ```
  The state in this case is specified by two input tensors `x` and `y`. To
  apply the algorithm to this objective function, one would need to specify
  either an initial population as a list of two tensors of shapes
  `[population_size, k]` and `[population_size]`. The following code shows the
  complete example:

  ```python
    population_size = 40
    # With an initial population and a multi-part state.
    initial_population = (tf.random.normal([population_size]),
                          tf.random.normal([population_size]))
    def easom_fn(x, y):
      return -(tf.math.cos(x) * tf.math.cos(y) *
               tf.math.exp(-(x-np.pi)**2 - (y-np.pi)**2))

    optim_results = tfp.optimizers.differential_evolution_minimize(
        easom_fn,
        initial_population=initial_population,
        seed=43210)

    print (optim_results.converged)
    print (optim_results.position)  # Should be (close to) [pi, pi].
    print (optim_results.objective_value)    # Should be -1.


    # With a single starting point
    initial_position = (tf.constant(1.0), tf.constant(1.0))

    optim_results = tfp.optimizers.differential_evolution_minimize(
        easom_fn,
        initial_position=initial_position,
        population_size=40,
        population_stddev=2.0,
        seed=43210)
  ```

  Args:
    objective_function: A Python callable that accepts a batch of possible
      solutions and returns the values of the objective function at those
      arguments as a rank 1 real `Tensor`. This specifies the function to be
      minimized. The input to this callable may be either a single `Tensor`
      or a Python `list` of `Tensor`s. The signature must match the format of
      the argument `population`. (i.e. objective_function(*population) must
      return the value of the function to be minimized).
    initial_population: A real `Tensor` or Python list of `Tensor`s.
      If a list, each `Tensor` must be of rank at least 1 and with a common
      first dimension. The first dimension indexes into the candidate solutions
      while the rest of the dimensions (if any) index into an individual
      solution. The size of the population must be at least 4. This is a
      requirement of the DE algorithm.
    initial_position: A real `Tensor` of any shape. The seed solution used
      to initialize the population of solutions. If this parameter is specified
      then `initial_population` must not be specified.
    population_size: A positive scalar int32 `Tensor` greater than 4. The
      size of the population to evolve. This parameter is ignored if
      `initial_population` is specified.
      Default value: 50.
    population_stddev: A positive scalar real `Tensor` of the same dtype
      as `initial_position`. This parameter is ignored if `initial_population`
      is specified. Used to generate the population from the `initial_position`
      by adding random normal noise with zero mean and the specified standard
      deviation.
      Default value: 1.0
    max_iterations: Positive scalar int32 `Tensor`. The maximum number of
      generations to evolve the population for.
      Default value: 100
    func_tolerance: Scalar `Tensor` of the same dtype as the output of the
      `objective_function`. The algorithm stops if the absolute difference
      between the largest and the smallest objective function value in the
      population is below this number.
      Default value: 0
    position_tolerance: Scalar `Tensor` of the same real dtype as
      `initial_position` or `initial_population`. The algorithm terminates if
      the largest absolute difference between the coordinates of the population
      members is below this threshold.
      Default value: 1e-8
    differential_weight: Real scalar `Tensor`. Must be positive and less than
      2.0. The parameter controlling the strength of mutation in the algorithm.
      Default value: 0.5
    crossover_prob: Real scalar `Tensor`. Must be between 0 and 1. The
      probability of recombination per site.
      Default value: 0.9
    seed: `int` or None. The random seed for this `Op`. If `None`, no seed is
      applied.
      Default value: None.
    name: (Optional) Python str. The name prefixed to the ops created by this
      function. If not supplied, the default name
      'differential_evolution_minimize' is used.
      Default value: None

  Returns:
    optimizer_results: An object containing the following attributes:
      converged: Scalar boolean `Tensor` indicating whether the minimum was
        found within the specified tolerances.
      num_objective_evaluations: The total number of objective
        evaluations performed.
      position: A `Tensor` containing the best point found during the search.
        If the search converged, then this value is the argmin of the
        objective function within the specified tolerances.
      objective_value: A `Tensor` containing the value of the objective
        function at the `position`. If the search
        converged, then this is the (local) minimum of
        the objective function.
      final_population: The final state of the population.
      final_objective_values: The objective function evaluated at the
        final population.
      initial_population: The starting population.
      initial_objective_values: The objective function evaluated at the
        initial population.
      num_iterations: The number of iterations of the main algorithm body.

  Raises:
    ValueError: If neither the initial population, nor the initial position
      are specified or if both are specified.
  """

  if initial_population is None and initial_position is None:
    raise ValueError('Either the initial population or the initial position '
                     'must be specified.')
  if initial_population is not None and initial_position is not None:
    raise ValueError('Only one of initial population or initial position '
                     'should be specified')

  with tf.compat.v1.name_scope(
      name,
      default_name='minimize',
      values=[
          initial_population, initial_position, population_size,
          population_stddev, max_iterations, func_tolerance, position_tolerance,
          differential_weight, crossover_prob
      ]):
    (
        was_iterable,
        population,
        population_values,
        max_iterations,
        func_tolerance,
        position_tolerance,
        differential_weight,
        crossover_prob
    ) = _get_initial_args(objective_function,
                          initial_population,
                          initial_position,
                          population_size,
                          population_stddev,
                          max_iterations,
                          func_tolerance,
                          position_tolerance,
                          differential_weight,
                          crossover_prob,
                          seed)

    def evolve_body(loop_vars):
      """Performs one step of the evolution."""
      next_population, next_population_values = one_step(
          objective_function,
          loop_vars.population,
          population_values=loop_vars.population_values,
          differential_weight=differential_weight,
          crossover_prob=crossover_prob,
          seed=seed)
      converged = _check_convergence(next_population,
                                     next_population_values,
                                     func_tolerance,
                                     position_tolerance)

      failed = _check_failure(next_population_values)

      return [_MinimizeLoopVars(
          converged=converged,
          failed=failed,
          num_iterations=loop_vars.num_iterations+1,
          population=next_population,
          population_values=next_population_values)]

    def evolve_cond(loop_vars):
      should_stop = (
          loop_vars.failed |
          loop_vars.converged |
          (max_iterations is not None and
           loop_vars.num_iterations >= max_iterations))
      return ~should_stop

    initial_vars = _MinimizeLoopVars(
        converged=tf.convert_to_tensor(value=False),
        failed=tf.convert_to_tensor(value=False),
        num_iterations=tf.convert_to_tensor(value=0),
        population=population,
        population_values=population_values)
    final_state = tf.while_loop(
        cond=evolve_cond, body=evolve_body, loop_vars=(initial_vars,))[0]
    best_position, best_values = _find_best_in_population(
        final_state.population,
        final_state.population_values)
    # Ensure we return a similar structure to what the user supplied.
    final_population = final_state.population
    if not was_iterable:
      final_population = final_population[0]
      best_position = best_position[0]
    return DifferentialEvolutionOptimizerResults(
        converged=final_state.converged,
        failed=final_state.failed,
        position=best_position,
        objective_value=best_values,
        final_population=final_population,
        final_objective_values=final_state.population_values,
        initial_population=population,
        initial_objective_values=population_values,
        num_iterations=final_state.num_iterations)


def _get_initial_args(objective_function,
                      initial_population,
                      initial_position,
                      population_size,
                      population_stddev,
                      max_iterations,
                      func_tolerance,
                      position_tolerance,
                      differential_weight,
                      crossover_prob,
                      seed):
  """Processes initial args."""
  was_iterable = False
  if initial_position is not None:
    initial_position, was_iterable = _ensure_list(initial_position)

  if initial_population is not None:
    initial_population, was_iterable = _ensure_list(initial_population)

  population = _get_starting_population(initial_population,
                                        initial_position,
                                        population_size,
                                        population_stddev,
                                        seed=seed)

  differential_weight = tf.convert_to_tensor(
      value=differential_weight, dtype=population[0].dtype.base_dtype)

  crossover_prob = tf.convert_to_tensor(value=crossover_prob)
  population_values = objective_function(*population)
  if max_iterations is not None:
    max_iterations = tf.convert_to_tensor(value=max_iterations)
  func_tolerance = tf.convert_to_tensor(
      value=func_tolerance, dtype=population_values.dtype.base_dtype)
  position_tolerance = tf.convert_to_tensor(
      value=position_tolerance, dtype=population[0].dtype.base_dtype)
  return (was_iterable,
          population,
          population_values,
          max_iterations,
          func_tolerance,
          position_tolerance,
          differential_weight,
          crossover_prob)


def _check_failure(population_values):
  """Checks if all the population values are NaN/infinite."""
  return tf.math.reduce_all(input_tensor=tf.math.is_inf(population_values))


def _find_best_in_population(population, values):
  """Finds the population member with the lowest value."""
  best_value = tf.math.reduce_min(input_tensor=values)
  best_index = tf.compat.v1.where(tf.math.equal(values, best_value))[0, 0]

  return ([population_part[best_index] for population_part in population],
          best_value)


def _check_convergence(population,
                       population_values,
                       func_tolerance,
                       position_tolerance):
  """Checks whether the convergence criteria have been met."""
  # Check func tolerance
  value_range = tf.math.abs(
      tf.math.reduce_max(input_tensor=population_values) -
      tf.math.reduce_min(input_tensor=population_values))
  value_converged = value_range <= func_tolerance
  # Ideally, we would compute the position convergence by computing the
  # pairwise distance between every member of the population and checking if
  # the maximum of those is less than the supplied tolerance. However, this is
  # completely infeasible in terms of performance. We adopt a more conservative
  # approach which checks the distance between the first population member
  # with the rest of the population. If the largest such distance is less than
  # half the supplied tolerance, we stop. The reason why this is sufficient is
  # as follows. For any pair of distinct points (a, b) in the population, we
  # have the relation:  |a - b| <= |x0 - a| + |x0 - b|, where x0 is any
  # other point. In particular, let x0 be the first element of the population
  # and suppose that the largest distance between this point and any other
  # member is epsilon. Then, for any pair of points (a, b),
  # |a - b| <= 2 * epsilon and hence, the maximum distance between any pair of
  # points in the population is bounded above by twice the distance between
  # the first point and other points.
  half_tol = position_tolerance / 2
  def part_converged(part):
    return tf.math.reduce_max(input_tensor=tf.math.abs(part -
                                                       part[0])) <= half_tol

  x_converged = tf.math.reduce_all(
      input_tensor=[part_converged(part) for part in population])
  return value_converged | x_converged


def _get_starting_population(initial_population,
                             initial_position,
                             population_size,
                             population_stddev,
                             seed):
  """Constructs the initial population.

  If an initial population is not already provided, this function constructs
  a population by adding random normal noise to the initial position.

  Args:
    initial_population: None or a list of `Tensor`s. The initial population.
    initial_position: None or a list of `Tensor`s. The initial position.
      If initial_population is None, this argument must not be None.
    population_size: Scalar integer `Tensor`. The number of members in the
      population. If the initial population is not None, this parameter is
      ignored.
    population_stddev: A positive scalar real `Tensor` of the same dtype
      as `initial_position` or `initial_population` (whichever is not None).
      This parameter is ignored if `initial_population`
      is specified. Used to generate the population from the
      `initial_position` by adding random normal noise with zero mean and
      the specified standard deviation.
    seed: Seed for random number generation.

  Returns:
    A list of `Tensor`s. The initial population.
  """
  if initial_population is not None:
    return [tf.convert_to_tensor(value=part) for part in initial_population]
  # Constructs the population by adding normal noise to the initial position.
  seed_stream = distributions.SeedStream(seed, salt='get_starting_population')
  population = []
  for part in initial_position:
    part = tf.convert_to_tensor(value=part)
    part_event_shape = tf.shape(input=part)
    # We only draw population_size-1 random vectors because we want to ensure
    # that the supplied position is part of the population. The first member
    # is set to be the initial_position.
    population_part_shape = tf.concat([[population_size-1],
                                       part_event_shape], axis=0)
    population_part = tf.random.normal(population_part_shape,
                                       stddev=population_stddev,
                                       dtype=part.dtype.base_dtype,
                                       seed=seed_stream())
    population_part += part
    population_part = tf.concat([[part], population_part], axis=0)
    population.append(population_part)
  return population


def _binary_crossover(population,
                      population_size,
                      mutants,
                      crossover_prob,
                      seed):
  """Performs recombination by binary crossover for the current population.

  Let v_i denote the i'th component of the member v and m_i the corresponding
  component of the mutant vector corresponding to v. Then the crossed over
  vector w_i is determined by setting w_i =
  (m_i with probability=crossover_prob else v_i). In addition, DE requires that
  at least one of the components is crossed over (otherwise we end
  up with no change). This is done by choosing on index say k randomly where
  a force crossover is performed (i.e. w_k = m_k). This is the scheme
  implemented in this function.

  Args:
    population: A Python list of `Tensor`s where each `Tensor` in the list
      must be of rank at least 1 and all the elements must have a common
      first dimension. The base population to cross over.
    population_size: A scalar integer `Tensor`. The number of elements in the
      population (i.e. size of the first dimension of any member of
      `population`).
    mutants: A Python list of `Tensor`s with the same structure as `population`.
      The mutated population.
    crossover_prob: A positive real scalar `Tensor` bounded above by 1.0. The
      probability of a crossover being performed for each axis.
    seed: `int` or None. The random seed for this `Op`. If `None`, no seed is
      applied.

  Returns:
    A list of `Tensor`s of the same structure, dtype and shape as `population`.
    The recombined population.
  """
  sizes = [tf.cast(tf.size(input=x), dtype=tf.float64) for x in population]
  seed_stream = distributions.SeedStream(seed, salt='binary_crossover')
  force_crossover_group = distributions.Categorical(sizes).sample(
      [population_size, 1], seed=seed_stream())
  recombinants = []
  for i, population_part in enumerate(population):
    pop_part_flat = tf.reshape(population_part, [population_size, -1])
    mutant_part_flat = tf.reshape(mutants[i], [population_size, -1])
    part_size = tf.size(input=population_part) // population_size
    force_crossovers = tf.one_hot(
        tf.random.uniform([population_size],
                          minval=0,
                          maxval=part_size,
                          dtype=tf.int32,
                          seed=seed_stream()),
        part_size,
        on_value=True,
        off_value=False,
        dtype=tf.bool)  # Tensor of shape [population_size, size]
    group_mask = tf.math.equal(force_crossover_group, i)
    force_crossovers &= group_mask
    do_binary_crossover = tf.random.uniform(
        [population_size, part_size],
        dtype=crossover_prob.dtype.base_dtype,
        seed=seed_stream()) < crossover_prob
    do_binary_crossover |= force_crossovers
    recombinant_flat = tf.compat.v1.where(
        do_binary_crossover, x=mutant_part_flat, y=pop_part_flat)
    recombinant = tf.reshape(recombinant_flat, tf.shape(input=population_part))
    recombinants.append(recombinant)
  return recombinants


def _get_mutants(population,
                 population_size,
                 mixing_indices,
                 differential_weight):
  """Computes the mutatated vectors for each population member.

  Args:
    population:  Python `list` of `Tensor`s representing the
      current population vectors. Each `Tensor` must be of the same real dtype.
      The first dimension of each `Tensor` indexes individual
      population members. For example, if the population is a list with a
      single `Tensor` of shape [n, m1, m2], then `n` is the population size and
      the shape of an individual solution is [m1, m2].
      If there is more than one element in the population, then each `Tensor`
      in the list should have the first axis of the same size.
    population_size: Scalar integer `Tensor`. The size of the population.
    mixing_indices: `Tensor` of integral dtype and shape [n, 3] where `n` is the
      number of members in the population. Each element of the `Tensor` must be
      a valid index into the first dimension of the population (i.e range
      between `0` and `n-1` inclusive).
    differential_weight: Real scalar `Tensor`. Must be positive and less than
      2.0. The parameter controlling the strength of mutation.

  Returns:
    mutants: `Tensor` or Python `list` of `Tensor`s of the same shape and dtype
      as the input population. The mutated vectors.
  """
  mixing_indices = tf.reshape(mixing_indices, [-1])
  weights = tf.stack([1.0, differential_weight, -differential_weight])
  def _mutant_part(population_part):
    donors = tf.gather(population_part, mixing_indices)
    donors = tf.transpose(
        a=tf.reshape(donors, [population_size, 3, -1]), perm=[0, 2, 1])
    return tf.math.reduce_sum(input_tensor=donors * weights, axis=-1)

  return [_mutant_part(population_part) for population_part in population]


def _get_mixing_indices(size, seed=None, name=None):
  """Generates an array of indices suitable for mutation operation.

  The mutation operation in differential evolution requires that for every
  element of the population, three distinct other elements be chosen to produce
  a trial candidate. This function generates an array of shape [size, 3]
  satisfying the properties that:
    (a). array[i, :] does not contain the index 'i'.
    (b). array[i, :] does not contain any overlapping indices.
    (c). All elements in the array are between 0 and size - 1 inclusive.

  Args:
    size: Scalar integer `Tensor`. The number of samples as well as a the range
      of the indices to sample from.
    seed: `int` or None. The random seed for this `Op`. If `None`, no seed is
      applied.
      Default value: `None`.
    name:  Python `str` name prefixed to Ops created by this function.
      Default value: 'get_mixing_indices'.

  Returns:
    sample: A `Tensor` of shape [size, 3] and same dtype as `size` containing
    samples without replacement between 0 and size - 1 (inclusive) with the
    `i`th row not including the number `i`.
  """
  with tf.compat.v1.name_scope(
      name, default_name='get_mixing_indices', values=[size]):
    size = tf.convert_to_tensor(value=size)
    dtype = size.dtype
    seed_stream = distributions.SeedStream(seed, salt='get_mixing_indices')
    first = tf.random.uniform([size],
                              maxval=size-1,
                              dtype=dtype,
                              seed=seed_stream())
    second = tf.random.uniform([size],
                               maxval=size-2,
                               dtype=dtype,
                               seed=seed_stream())
    third = tf.random.uniform([size],
                              maxval=size-3,
                              dtype=dtype,
                              seed=seed_stream())

    # Shift second if it is on top of or to the right of first
    second = tf.compat.v1.where(first < second, x=second, y=second + 1)
    smaller = tf.math.minimum(first, second)
    larger = tf.math.maximum(first, second)
    # Shift the third one so it does not coincide with either the first or the
    # second number. Assuming first < second, shift by 1 if the number is in
    # [first, second) and by 2 if the number is greater than or equal to the
    # second.
    third = tf.compat.v1.where(third < smaller, x=third, y=third + 1)
    third = tf.compat.v1.where(third < larger, x=third, y=third + 1)
    sample = tf.stack([first, second, third], axis=1)
    to_avoid = tf.expand_dims(tf.range(size), axis=-1)
    sample = tf.compat.v1.where(sample < to_avoid, x=sample, y=sample + 1)
    return sample


def _ensure_list(tensor_or_list):
  """Converts the input arg to a list if it is not a list already.

  Args:
    tensor_or_list: A `Tensor` or a Python list of `Tensor`s. The argument to
      convert to a list of `Tensor`s.

  Returns:
    A tuple of two elements. The first is a Python list of `Tensor`s containing
    the original arguments. The second is a boolean indicating whether
    the original argument was a list or tuple already.
  """
  if isinstance(tensor_or_list, (list, tuple)):
    return list(tensor_or_list), True
  return [tensor_or_list], False
