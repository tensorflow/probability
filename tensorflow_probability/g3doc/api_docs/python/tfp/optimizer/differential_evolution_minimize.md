<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.optimizer.differential_evolution_minimize" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.optimizer.differential_evolution_minimize


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/optimizer/differential_evolution.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Applies the Differential evolution algorithm to minimize a function.

``` python
tfp.optimizer.differential_evolution_minimize(
    objective_function,
    initial_population=None,
    initial_position=None,
    population_size=50,
    population_stddev=1.0,
    max_iterations=100,
    func_tolerance=0,
    position_tolerance=1e-08,
    differential_weight=0.5,
    crossover_prob=0.9,
    seed=None,
    name=None
)
```



<!-- Placeholder for "Used in" -->

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

#### Args:


* <b>`objective_function`</b>: A Python callable that accepts a batch of possible
  solutions and returns the values of the objective function at those
  arguments as a rank 1 real `Tensor`. This specifies the function to be
  minimized. The input to this callable may be either a single `Tensor`
  or a Python `list` of `Tensor`s. The signature must match the format of
  the argument `population`. (i.e. objective_function(*population) must
  return the value of the function to be minimized).
* <b>`initial_population`</b>: A real `Tensor` or Python list of `Tensor`s.
  If a list, each `Tensor` must be of rank at least 1 and with a common
  first dimension. The first dimension indexes into the candidate solutions
  while the rest of the dimensions (if any) index into an individual
  solution. The size of the population must be at least 4. This is a
  requirement of the DE algorithm.
* <b>`initial_position`</b>: A real `Tensor` of any shape. The seed solution used
  to initialize the population of solutions. If this parameter is specified
  then `initial_population` must not be specified.
* <b>`population_size`</b>: A positive scalar int32 `Tensor` greater than 4. The
  size of the population to evolve. This parameter is ignored if
  `initial_population` is specified.
  Default value: 50.
* <b>`population_stddev`</b>: A positive scalar real `Tensor` of the same dtype
  as `initial_position`. This parameter is ignored if `initial_population`
  is specified. Used to generate the population from the `initial_position`
  by adding random normal noise with zero mean and the specified standard
  deviation.
  Default value: 1.0
* <b>`max_iterations`</b>: Positive scalar int32 `Tensor`. The maximum number of
  generations to evolve the population for.
  Default value: 100
* <b>`func_tolerance`</b>: Scalar `Tensor` of the same dtype as the output of the
  `objective_function`. The algorithm stops if the absolute difference
  between the largest and the smallest objective function value in the
  population is below this number.
  Default value: 0
* <b>`position_tolerance`</b>: Scalar `Tensor` of the same real dtype as
  `initial_position` or `initial_population`. The algorithm terminates if
  the largest absolute difference between the coordinates of the population
  members is below this threshold.
  Default value: 1e-8
* <b>`differential_weight`</b>: Real scalar `Tensor`. Must be positive and less than
  2.0. The parameter controlling the strength of mutation in the algorithm.
  Default value: 0.5
* <b>`crossover_prob`</b>: Real scalar `Tensor`. Must be between 0 and 1. The
  probability of recombination per site.
  Default value: 0.9
* <b>`seed`</b>: `int` or None. The random seed for this `Op`. If `None`, no seed is
  applied.
  Default value: None.
* <b>`name`</b>: (Optional) Python str. The name prefixed to the ops created by this
  function. If not supplied, the default name
  'differential_evolution_minimize' is used.
  Default value: None


#### Returns:


* <b>`optimizer_results`</b>: An object containing the following attributes:
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


#### Raises:


* <b>`ValueError`</b>: If neither the initial population, nor the initial position
  are specified or if both are specified.