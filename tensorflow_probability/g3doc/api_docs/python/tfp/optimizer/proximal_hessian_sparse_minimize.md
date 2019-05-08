<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.optimizer.proximal_hessian_sparse_minimize" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.optimizer.proximal_hessian_sparse_minimize

Minimize using Hessian-informed proximal gradient descent.

``` python
tfp.optimizer.proximal_hessian_sparse_minimize(
    grad_and_hessian_loss_fn,
    x_start,
    tolerance,
    l1_regularizer,
    l2_regularizer=None,
    maximum_iterations=1,
    maximum_full_sweeps_per_iteration=1,
    learning_rate=None,
    name=None
)
```



Defined in [`python/optimizer/proximal_hessian_sparse.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/optimizer/proximal_hessian_sparse.py).

<!-- Placeholder for "Used in" -->

This function solves the regularized minimization problem

```none
argmin{ Loss(x)
          + l1_regularizer * ||x||_1
          + l2_regularizer * ||x||_2**2
        : x in R^n }
```

where `Loss` is a convex C^2 function (typically, `Loss` is the negative log
likelihood of a model and `x` is a vector of model coefficients).  The `Loss`
function does not need to be supplied directly, but this optimizer does need a
way to compute the gradient and Hessian of the Loss function at a given value
of `x`.  The gradient and Hessian are often computationally expensive, and
this optimizer calls them relatively few times compared with other algorithms.

#### Args:

* <b>`grad_and_hessian_loss_fn`</b>: callable that takes as input a (batch of) `Tensor`
  of the same shape and dtype as `x_start` and returns the triple
  `(gradient_unregularized_loss, hessian_unregularized_loss_outer,
  hessian_unregularized_loss_middle)` as defined in the argument spec of
  `minimize_one_step`.
* <b>`x_start`</b>: (Batch of) vector-shaped, `float` `Tensor` representing the initial
  value of the argument to the `Loss` function.
* <b>`tolerance`</b>: scalar, `float` `Tensor` representing the tolerance for each
  optimization step; see the `tolerance` argument of
  `minimize_one_step`.
* <b>`l1_regularizer`</b>: scalar, `float` `Tensor` representing the weight of the L1
  regularization term (see equation above).
* <b>`l2_regularizer`</b>: scalar, `float` `Tensor` representing the weight of the L2
  regularization term (see equation above).
  Default value: `None` (i.e., no L2 regularization).
* <b>`maximum_iterations`</b>: Python integer specifying the maximum number of
  iterations of the outer loop of the optimizer.  After this many iterations
  of the outer loop, the algorithm will terminate even if the return value
  `optimal_x` has not converged.
  Default value: `1`.
* <b>`maximum_full_sweeps_per_iteration`</b>: Python integer specifying the maximum
  number of sweeps allowed in each iteration of the outer loop of the
  optimizer.  Passed as the `maximum_full_sweeps` argument to
  `minimize_one_step`.
  Default value: `1`.
* <b>`learning_rate`</b>: scalar, `float` `Tensor` representing a multiplicative factor
  used to dampen the proximal gradient descent steps.
  Default value: `None` (i.e., factor is conceptually `1`).
* <b>`name`</b>: Python string representing the name of the TensorFlow operation.
  The default name is `"minimize"`.


#### Returns:

  x: `Tensor` of the same shape and dtype as `x_start`, representing the
    (batches of) computed values of `x` which minimizes `Loss(x)`.
  is_converged: scalar, `bool` `Tensor` indicating whether the minimization
    procedure converged within the specified number of iterations across all
    batches.  Here convergence means that an iteration of the inner loop
    (`minimize_one_step`) returns `True` for its `is_converged` output value.
  iter: scalar, `int` `Tensor` indicating the actual number of iterations of
    the outer loop of the optimizer completed (i.e., number of calls to
    `minimize_one_step` before achieving convergence).

#### References

[1]: Jerome Friedman, Trevor Hastie and Rob Tibshirani. Regularization Paths
     for Generalized Linear Models via Coordinate Descent. _Journal of
     Statistical Software_, 33(1), 2010.
     https://www.jstatsoft.org/article/view/v033i01/v33i01.pdf

[2]: Guo-Xun Yuan, Chia-Hua Ho and Chih-Jen Lin. An Improved GLMNET for
     L1-regularized Logistic Regression. _Journal of Machine Learning
     Research_, 13, 2012.
     http://www.jmlr.org/papers/volume13/yuan12a/yuan12a.pdf