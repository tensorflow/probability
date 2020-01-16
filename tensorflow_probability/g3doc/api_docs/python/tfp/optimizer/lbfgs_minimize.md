<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.optimizer.lbfgs_minimize" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.optimizer.lbfgs_minimize


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/optimizer/lbfgs.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Applies the L-BFGS algorithm to minimize a differentiable function.

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
    stopping_condition=None,
    name=None
)
```



<!-- Placeholder for "Used in" -->

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
  to be minimized. The input is of shape `[..., n]`, where `n` is the size
  of the domain of input points, and all others are batching dimensions.
  The first component of the return value is a real `Tensor` of matching
  shape `[...]`. The second component (the gradient) is also of shape
  `[..., n]` like the input value to the function.
* <b>`initial_position`</b>: Real `Tensor` of shape `[..., n]`. The starting point, or
  points when using batching dimensions, of the search procedure. At these
  points the function value and the gradient norm should be finite.
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
  iterations for L-BFGS updates.
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
  converged: Scalar boolean tensor indicating whether the minimum was
    found within tolerance.
  failed:  Scalar boolean tensor indicating whether a line search
    step failed to find a suitable step size satisfying Wolfe
    conditions. In the absence of any constraints on the
    number of objective evaluations permitted, this value will
    be the complement of `converged`. However, if there is
    a constraint and the search stopped due to available
    evaluations being exhausted, both `failed` and `converged`
    will be simultaneously False.
  num_objective_evaluations: The total number of objective
    evaluations performed.
  position: A tensor containing the last argument value found
    during the search. If the search converged, then
    this value is the argmin of the objective function.
  objective_value: A tensor containing the value of the objective
    function at the `position`. If the search converged, then this is
    the (local) minimum of the objective function.
  objective_gradient: A tensor containing the gradient of the objective
    function at the `position`. If the search converged the
    max-norm of this tensor should be below the tolerance.
  position_deltas: A tensor encoding information about the latest
    changes in `position` during the algorithm execution.
  gradient_deltas: A tensor encoding information about the latest
    changes in `objective_gradient` during the algorithm execution.