<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.optimizer.bfgs_minimize" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.optimizer.bfgs_minimize


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/optimizer/bfgs.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Applies the BFGS algorithm to minimize a differentiable function.

``` python
tfp.optimizer.bfgs_minimize(
    value_and_gradients_function,
    initial_position,
    tolerance=1e-08,
    x_tolerance=0,
    f_relative_tolerance=0,
    initial_inverse_hessian_estimate=None,
    max_iterations=50,
    parallel_iterations=1,
    stopping_condition=None,
    name=None
)
```



<!-- Placeholder for "Used in" -->

Performs unconstrained minimization of a differentiable function using the
BFGS scheme. For details of the algorithm, see [Nocedal and Wright(2006)][1].

### Usage:

The following example demonstrates the BFGS optimizer attempting to find the
minimum for a simple two dimensional quadratic objective function.

```python
  minimum = np.array([1.0, 1.0])  # The center of the quadratic bowl.
  scales = np.array([2.0, 3.0])  # The scales along the two axes.

  # The objective function and the gradient.
  def quadratic(x):
    value = tf.reduce_sum(scales * (x - minimum) ** 2)
    return value, tf.gradients(value, x)[0]

  start = tf.constant([0.6, 0.8])  # Starting point for the search.
  optim_results = tfp.optimizer.bfgs_minimize(
      quadratic, initial_position=start, tolerance=1e-8)

  with tf.Session() as session:
    results = session.run(optim_results)
    # Check that the search converged
    assert(results.converged)
    # Check that the argmin is close to the actual value.
    np.testing.assert_allclose(results.position, minimum)
    # Print out the total number of function evaluations it took. Should be 6.
    print ("Function evaluations: %d" % results.num_objective_evaluations)
```

### References:
[1]: Jorge Nocedal, Stephen Wright. Numerical Optimization. Springer Series in
  Operations Research. pp 136-140. 2006
  http://pages.mtu.edu/~struther/Courses/OLD/Sp2013/5630/Jorge_Nocedal_Numerical_optimization_267490.pdf

#### Args:


* <b>`value_and_gradients_function`</b>:  A Python callable that accepts a point as a
  real `Tensor` and returns a tuple of `Tensor`s of real dtype containing
  the value of the function and its gradient at that point. The function
  to be minimized. The input should be of shape `[..., n]`, where `n` is
  the size of the domain of input points, and all others are batching
  dimensions. The first component of the return value should be a real
  `Tensor` of matching shape `[...]`. The second component (the gradient)
  should also be of shape `[..., n]` like the input value to the function.
* <b>`initial_position`</b>: real `Tensor` of shape `[..., n]`. The starting point, or
  points when using batching dimensions, of the search procedure. At these
  points the function value and the gradient norm should be finite.
* <b>`tolerance`</b>: Scalar `Tensor` of real dtype. Specifies the gradient tolerance
  for the procedure. If the supremum norm of the gradient vector is below
  this number, the algorithm is stopped.
* <b>`x_tolerance`</b>: Scalar `Tensor` of real dtype. If the absolute change in the
  position between one iteration and the next is smaller than this number,
  the algorithm is stopped.
* <b>`f_relative_tolerance`</b>: Scalar `Tensor` of real dtype. If the relative change
  in the objective value between one iteration and the next is smaller
  than this value, the algorithm is stopped.
* <b>`initial_inverse_hessian_estimate`</b>: Optional `Tensor` of the same dtype
  as the components of the output of the `value_and_gradients_function`.
  If specified, the shape should broadcastable to shape `[..., n, n]`; e.g.
  if a single `[n, n]` matrix is provided, it will be automatically
  broadcasted to all batches. Alternatively, one can also specify a
  different hessian estimate for each batch member.
  For the correctness of the algorithm, it is required that this parameter
  be symmetric and positive definite. Specifies the starting estimate for
  the inverse of the Hessian at the initial point. If not specified,
  the identity matrix is used as the starting estimate for the
  inverse Hessian.
* <b>`max_iterations`</b>: Scalar positive int32 `Tensor`. The maximum number of
  iterations for BFGS updates.
* <b>`parallel_iterations`</b>: Positive integer. The number of iterations allowed to
  run in parallel.
* <b>`stopping_condition`</b>: (Optional) A Python function that takes as input two
  Boolean tensors of shape `[...]`, and returns a Boolean scalar tensor.
  The input tensors are `converged` and `failed`, indicating the current
  status of each respective batch member; the return value states whether
  the algorithm should stop. The default is tfp.optimizer.converged_all
  which only stops when all batch members have either converged or failed.
  An alternative is tfp.optimizer.converged_any which stops as soon as one
  batch member has converged, or when all have failed.
* <b>`name`</b>: (Optional) Python str. The name prefixed to the ops created by this
  function. If not supplied, the default name 'minimize' is used.


#### Returns:


* <b>`optimizer_results`</b>: A namedtuple containing the following items:
  converged: boolean tensor of shape `[...]` indicating for each batch
    member whether the minimum was found within tolerance.
  failed:  boolean tensor of shape `[...]` indicating for each batch
    member whether a line search step failed to find a suitable step size
    satisfying Wolfe conditions. In the absence of any constraints on the
    number of objective evaluations permitted, this value will
    be the complement of `converged`. However, if there is
    a constraint and the search stopped due to available
    evaluations being exhausted, both `failed` and `converged`
    will be simultaneously False.
  num_objective_evaluations: The total number of objective
    evaluations performed.
  position: A tensor of shape `[..., n]` containing the last argument value
    found during the search from each starting point. If the search
    converged, then this value is the argmin of the objective function.
  objective_value: A tensor of shape `[...]` with the value of the
    objective function at the `position`. If the search converged, then
    this is the (local) minimum of the objective function.
  objective_gradient: A tensor of shape `[..., n]` containing the gradient
    of the objective function at the `position`. If the search converged
    the max-norm of this tensor should be below the tolerance.
  inverse_hessian_estimate: A tensor of shape `[..., n, n]` containing the
    inverse of the estimated Hessian.