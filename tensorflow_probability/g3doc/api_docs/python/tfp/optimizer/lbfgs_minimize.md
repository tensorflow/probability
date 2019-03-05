<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.optimizer.lbfgs_minimize" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.optimizer.lbfgs_minimize

``` python
tfp.optimizer.lbfgs_minimize(
    value_and_gradients_function,
    initial_position,
    num_correction_pairs=10,
    tolerance=1e-08,
    x_tolerance=0,
    f_relative_tolerance=0,
    initial_inverse_hessian_estimate=None,
    max_iterations=50,
    parallel_iterations=1,
    name=None
)
```

Applies the L-BFGS algorithm to minimize a differentiable function.

Performs unconstrained minimization of a differentiable function using the
L-BFGS scheme. See [Nocedal and Wright(2006)][1] for details of the algorithm.

### Usage:

The following example demonstrates the L-BFGS optimizer attempting to find the
minimum for a simple high-dimensional quadratic objective function.

```python
  # A high-dimensional quadratic bowl.
  ndims = 60
  minimum = np.ones([ndims], dtype='float64')
  scales = np.arange(ndims, dtype='float64') + 1.0

  # The objective function and the gradient.
  def quadratic(x):
    value = tf.reduce_sum(scales * (x - minimum) ** 2)
    return value, tf.gradients(value, x)[0]

  start = np.arange(ndims, 0, -1, dtype='float64')
  optim_results = tfp.optimizer.lbfgs_minimize(
      quadratic, initial_position=start, num_correction_pairs=10,
      tolerance=1e-8)

  with tf.Session() as session:
    results = session.run(optim_results)
    # Check that the search converged
    assert(results.converged)
    # Check that the argmin is close to the actual value.
    np.testing.assert_allclose(results.position, minimum)
```

### References:

[1] Jorge Nocedal, Stephen Wright. Numerical Optimization. Springer Series
    in Operations Research. pp 176-180. 2006

http://pages.mtu.edu/~struther/Courses/OLD/Sp2013/5630/Jorge_Nocedal_Numerical_optimization_267490.pdf

#### Args:

* <b>`value_and_gradients_function`</b>:  A Python callable that accepts a point as a
    real `Tensor` and returns a tuple of `Tensor`s of real dtype containing
    the value of the function and its gradient at that point. The function
    to be minimized. The first component of the return value should be a
    real scalar `Tensor`. The second component (the gradient) should have the
    same shape as the input value to the function.
* <b>`initial_position`</b>: `Tensor` of real dtype. The starting point of the search
    procedure. Should be a point at which the function value and the gradient
    norm are finite.
* <b>`num_correction_pairs`</b>: Positive integer. Specifies the maximum number of
    (position_delta, gradient_delta) correction pairs to keep as implicit
    approximation of the Hessian matrix.
* <b>`tolerance`</b>: Scalar `Tensor` of real dtype. Specifies the gradient tolerance
    for the procedure. If the supremum norm of the gradient vector is below
    this number, the algorithm is stopped.
* <b>`x_tolerance`</b>: Scalar `Tensor` of real dtype. If the absolute change in the
    position between one iteration and the next is smaller than this number,
    the algorithm is stopped.
* <b>`f_relative_tolerance`</b>: Scalar `Tensor` of real dtype. If the relative change
    in the objective value between one iteration and the next is smaller
    than this value, the algorithm is stopped.
* <b>`initial_inverse_hessian_estimate`</b>: None. Option currently not supported.
* <b>`max_iterations`</b>: Scalar positive int32 `Tensor`. The maximum number of
    iterations for BFGS updates.
* <b>`parallel_iterations`</b>: Positive integer. The number of iterations allowed to
    run in parallel.
* <b>`name`</b>: (Optional) Python str. The name prefixed to the ops created by this
    function. If not supplied, the default name 'minimize' is used.


#### Returns:

* <b>`optimizer_results`</b>: A namedtuple containing the following items:
* <b>`converged`</b>: Scalar boolean tensor indicating whether the minimum was
      found within tolerance.
* <b>`failed`</b>:  Scalar boolean tensor indicating whether a line search
      step failed to find a suitable step size satisfying Wolfe
      conditions. In the absence of any constraints on the
      number of objective evaluations permitted, this value will
      be the complement of `converged`. However, if there is
      a constraint and the search stopped due to available
      evaluations being exhausted, both `failed` and `converged`
      will be simultaneously False.
* <b>`num_objective_evaluations`</b>: The total number of objective
      evaluations performed.
* <b>`position`</b>: A tensor containing the last argument value found
      during the search. If the search converged, then
      this value is the argmin of the objective function.
* <b>`objective_value`</b>: A tensor containing the value of the objective
      function at the `position`. If the search converged, then this is
      the (local) minimum of the objective function.
* <b>`objective_gradient`</b>: A tensor containing the gradient of the objective
      function at the `position`. If the search converged the
      max-norm of this tensor should be below the tolerance.
* <b>`position_deltas`</b>: A tensor encoding information about the latest
      changes in `position` during the algorithm execution.
* <b>`gradient_deltas`</b>: A tensor encoding information about the latest
      changes in `objective_gradient` during the algorithm execution.