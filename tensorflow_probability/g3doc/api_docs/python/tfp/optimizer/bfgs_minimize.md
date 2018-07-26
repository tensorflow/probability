Project: /probability/_project.yaml
Book: /probability/_book.yaml
page_type: reference
<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.optimizer.bfgs_minimize" />
</div>

# tfp.optimizer.bfgs_minimize

``` python
tfp.optimizer.bfgs_minimize(
    value_and_gradients_function,
    initial_position,
    tolerance=1e-08,
    initial_inverse_hessian_estimate=None,
    parallel_iterations=1,
    name=None
)
```

Applies the BFGS algorithm to minimize a differentiable function.

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
    to be minimized. The first component of the return value should be a
    real scalar `Tensor`. The second component (the gradient) should have the
    same shape as the input value to the function.
* <b>`initial_position`</b>: `Tensor` of real dtype. The starting point of the search
    procedure. Should be a point at which the function value and the gradient
    norm are finite.
* <b>`tolerance`</b>: Scalar `Tensor` of real dtype. Specifies the tolerance for the
    procedure. The algorithm is said to have converged when the Euclidean
    norm of the gradient is below this value.
* <b>`initial_inverse_hessian_estimate`</b>: Optional `Tensor` of the same dtype
    as the components of the output of the `value_and_gradients_function`.
    If specified, the shape should be `initial_position.shape` * 2.
    For example, if the shape of `initial_position` is `[n]`, then the
    acceptable shape of `initial_inverse_hessian_estimate` is as a square
    matrix of shape `[n, n]`.
    If the shape of `initial_position` is `[n, m]`, then the required shape
    is `[n, m, n, m]`.
    For the correctness of the algorithm, it is required that this parameter
    be symmetric and positive definite. Specifies the starting estimate for
    the inverse of the Hessian at the initial point. If not specified,
    the identity matrix is used as the starting estimate for the
    inverse Hessian.
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
      function at the `position`. If the search converged this
      L2-norm of this tensor should be below the tolerance.
* <b>`inverse_hessian_estimate`</b>: A tensor containing the inverse of the
      estimated Hessian.