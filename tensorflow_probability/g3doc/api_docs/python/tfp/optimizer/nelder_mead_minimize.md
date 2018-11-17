<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.optimizer.nelder_mead_minimize" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.optimizer.nelder_mead_minimize

``` python
tfp.optimizer.nelder_mead_minimize(
    objective_function,
    initial_simplex=None,
    initial_vertex=None,
    step_sizes=None,
    objective_at_initial_simplex=None,
    objective_at_initial_vertex=None,
    batch_evaluate_objective=False,
    func_tolerance=1e-08,
    position_tolerance=1e-08,
    parallel_iterations=1,
    max_iterations=None,
    reflection=None,
    expansion=None,
    contraction=None,
    shrinkage=None,
    name=None
)
```

Minimum of the objective function using the Nelder Mead simplex algorithm.

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

#### Args:

* <b>`objective_function`</b>:  A Python callable that accepts a point as a
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
* <b>`initial_simplex`</b>: (Optional) `Tensor` of real dtype. The initial simplex to
    start the search. If supplied, should be a `Tensor` of shape `[n+1] + s`
    where `n` is the dimension of the problem and `s` is the shape of a
    single point in the domain. Each row (i.e. the `Tensor` with a given
    value of the first index) is interpreted as a vertex of a simplex and
    hence the rows must be affinely independent. If not supplied, an axes
    aligned simplex is constructed using the `initial_vertex` and
    `step_sizes`. Only one and at least one of `initial_simplex` and
    `initial_vertex` must be supplied.
* <b>`initial_vertex`</b>: (Optional) `Tensor` of real dtype and any shape that can
    be consumed by the `objective_function`. A single point in the domain that
    will be used to construct an axes aligned initial simplex.
* <b>`step_sizes`</b>: (Optional) `Tensor` of real dtype and shape broadcasting
    compatible with `initial_vertex`. Supplies the simplex scale along each
    axes. Only used if `initial_simplex` is not supplied. See description
    above for details on how step sizes and initial vertex are used to
    construct the initial simplex.
* <b>`objective_at_initial_simplex`</b>: (Optional) Rank `1` `Tensor` of real dtype
    of a rank `1` `Tensor`. The value of the objective function at the
    initial simplex. May be supplied only if `initial_simplex` is
    supplied. If not supplied, it will be computed.
* <b>`objective_at_initial_vertex`</b>: (Optional) Scalar `Tensor` of real dtype. The
    value of the objective function at the initial vertex. May be supplied
    only if the `initial_vertex` is also supplied.
* <b>`batch_evaluate_objective`</b>: (Optional) Python `bool`. If True, the objective
    function will be evaluated on all the vertices of the simplex packed
    into a single tensor. If False, the objective will be mapped across each
    vertex separately. Evaluating the objective function in a batch allows
    use of vectorization and should be preferred if the objective function
    allows it.
* <b>`func_tolerance`</b>: (Optional) Scalar `Tensor` of real dtype. The algorithm
    stops if the absolute difference between the largest and the smallest
    function value on the vertices of the simplex is below this number.
* <b>`position_tolerance`</b>: (Optional) Scalar `Tensor` of real dtype. The
    algorithm stops if the largest absolute difference between the
    coordinates of the vertices is below this threshold.
* <b>`parallel_iterations`</b>: (Optional) Positive integer. The number of iterations
    allowed to run in parallel.
* <b>`max_iterations`</b>: (Optional) Scalar positive `Tensor` of dtype `int32`.
    The maximum number of iterations allowed. If `None` then no limit is
    applied.
* <b>`reflection`</b>: (Optional) Positive Scalar `Tensor` of same dtype as
    `initial_vertex`. This parameter controls the scaling of the reflected
    vertex. See, [Press et al(2007)][1] for details. If not specified,
    uses the dimension dependent prescription of [Gao and Han(2012)][3].
* <b>`expansion`</b>: (Optional) Positive Scalar `Tensor` of same dtype as
    `initial_vertex`. Should be greater than `1` and `reflection`. This
    parameter controls the expanded scaling of a reflected vertex.
    See, [Press et al(2007)][1] for details. If not specified, uses the
    dimension dependent prescription of [Gao and Han(2012)][3].
* <b>`contraction`</b>: (Optional) Positive scalar `Tensor` of same dtype as
    `initial_vertex`. Must be between `0` and `1`. This parameter controls
    the contraction of the reflected vertex when the objective function at
    the reflected point fails to show sufficient decrease.
    See, [Press et al(2007)][1] for more details. If not specified, uses
    the dimension dependent prescription of [Gao and Han(2012][3].
* <b>`shrinkage`</b>: (Optional) Positive scalar `Tensor` of same dtype as
    `initial_vertex`. Must be between `0` and `1`. This parameter is the scale
    by which the simplex is shrunk around the best point when the other
    steps fail to produce improvements.
    See, [Press et al(2007)][1] for more details. If not specified, uses
    the dimension dependent prescription of [Gao and Han(2012][3].
* <b>`name`</b>: (Optional) Python str. The name prefixed to the ops created by this
    function. If not supplied, the default name 'minimize' is used.


#### Returns:

* <b>`optimizer_results`</b>: A namedtuple containing the following items:
* <b>`converged`</b>: Scalar boolean tensor indicating whether the minimum was
      found within tolerance.
* <b>`num_objective_evaluations`</b>: The total number of objective
      evaluations performed.
* <b>`position`</b>: A `Tensor` containing the last argument value found
      during the search. If the search converged, then
      this value is the argmin of the objective function.
* <b>`objective_value`</b>: A tensor containing the value of the objective
      function at the `position`. If the search
      converged, then this is the (local) minimum of
      the objective function.
* <b>`final_simplex`</b>: The last simplex constructed before stopping.
* <b>`final_objective_values`</b>: The objective function evaluated at the
      vertices of the final simplex.
* <b>`initial_simplex`</b>: The starting simplex.
* <b>`initial_objective_values`</b>: The objective function evaluated at the
      vertices of the initial simplex.
* <b>`num_iterations`</b>: The number of iterations of the main algorithm body.


#### Raises:

* <b>`ValueError`</b>: If any of the following conditions hold
    1. If none or more than one of `initial_simplex` and `initial_vertex` are
      supplied.
    2. If `initial_simplex` and `step_sizes` are both specified.