<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.optimizer.differential_evolution_one_step" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.optimizer.differential_evolution_one_step


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/optimizer/differential_evolution.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Performs one step of the differential evolution algorithm.

``` python
tfp.optimizer.differential_evolution_one_step(
    objective_function,
    population,
    population_values=None,
    differential_weight=0.5,
    crossover_prob=0.9,
    seed=None,
    name=None
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`objective_function`</b>:  A Python callable that accepts a batch of possible
  solutions and returns the values of the objective function at those
  arguments as a rank 1 real `Tensor`. This specifies the function to be
  minimized. The input to this callable may be either a single `Tensor`
  or a Python `list` of `Tensor`s. The signature must match the format of
  the argument `population`. (i.e., objective_function(*population) must
  return the value of the function to be minimized).
* <b>`population`</b>:  `Tensor` or Python `list` of `Tensor`s representing the
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
* <b>`population_values`</b>: A `Tensor` of rank 1 and real dtype. The result of
  applying `objective_function` to the `population`. If not supplied it is
  computed using the `objective_function`.
  Default value: None.
* <b>`differential_weight`</b>: Real scalar `Tensor`. Must be positive and less than
  2.0. The parameter controlling the strength of mutation.
  Default value: 0.5
* <b>`crossover_prob`</b>: Real scalar `Tensor`. Must be between 0 and 1. The
  probability of recombination per site.
  Default value: 0.9
* <b>`seed`</b>: `int` or None. The random seed for this `Op`. If `None`, no seed is
  applied.
  Default value: None.
* <b>`name`</b>: (Optional) Python str. The name prefixed to the ops created by this
  function. If not supplied, the default name 'one_step' is
  used.
  Default value: None


#### Returns:

A sequence containing the following elements (in order):

* <b>`next_population`</b>: A `Tensor` or Python `list` of `Tensor`s of the same
  structure as the input population. The population at the next generation.
* <b>`next_population_values`</b>: A `Tensor` of same shape and dtype as input
  `population_values`. The function values for the `next_population`.