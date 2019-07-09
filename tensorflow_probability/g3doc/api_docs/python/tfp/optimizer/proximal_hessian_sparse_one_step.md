<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.optimizer.proximal_hessian_sparse_one_step" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.optimizer.proximal_hessian_sparse_one_step

One step of (the outer loop of) the minimization algorithm.

``` python
tfp.optimizer.proximal_hessian_sparse_one_step(
    gradient_unregularized_loss,
    hessian_unregularized_loss_outer,
    hessian_unregularized_loss_middle,
    x_start,
    tolerance,
    l1_regularizer,
    l2_regularizer=None,
    maximum_full_sweeps=1,
    learning_rate=None,
    name=None
)
```



Defined in [`python/optimizer/proximal_hessian_sparse.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/optimizer/proximal_hessian_sparse.py).

<!-- Placeholder for "Used in" -->

This function returns a new value of `x`, equal to `x_start + x_update`.  The
increment `x_update in R^n` is computed by a coordinate descent method, that
is, by a loop in which each iteration updates exactly one coordinate of
`x_update`.  (Some updates may leave the value of the coordinate unchanged.)

The particular update method used is to apply an L1-based proximity operator,
"soft threshold", whose fixed point `x_update_fix` is the desired minimum

```none
x_update_fix = argmin{
    Loss(x_start + x_update')
      + l1_regularizer * ||x_start + x_update'||_1
      + l2_regularizer * ||x_start + x_update'||_2**2
    : x_update' }
```

where in each iteration `x_update'` is constrained to have at most one nonzero
coordinate.

This update method preserves sparsity, i.e., tends to find sparse solutions if
`x_start` is sparse.  Additionally, the choice of step size is based on
curvature (Hessian), which significantly speeds up convergence.

This algorithm assumes that `Loss` is convex, at least in a region surrounding
the optimum.  (If `l2_regularizer > 0`, then only weak convexity is needed.)

#### Args:


* <b>`gradient_unregularized_loss`</b>: (Batch of) `Tensor` with the same shape and
  dtype as `x_start` representing the gradient, evaluated at `x_start`, of
  the unregularized loss function (denoted `Loss` above).  (In all current
  use cases, `Loss` is the negative log likelihood.)
* <b>`hessian_unregularized_loss_outer`</b>: (Batch of) `Tensor` or `SparseTensor`
  having the same dtype as `x_start`, and shape `[N, n]` where `x_start` has
  shape `[n]`, satisfying the property
  `Transpose(hessian_unregularized_loss_outer)
  @ diag(hessian_unregularized_loss_middle)
  @ hessian_unregularized_loss_inner
  = (approximation of) Hessian matrix of Loss, evaluated at x_start`.
* <b>`hessian_unregularized_loss_middle`</b>: (Batch of) vector-shaped `Tensor` having
  the same dtype as `x_start`, and shape `[N]` where
  `hessian_unregularized_loss_outer` has shape `[N, n]`, satisfying the
  property
  `Transpose(hessian_unregularized_loss_outer)
  @ diag(hessian_unregularized_loss_middle)
  @ hessian_unregularized_loss_inner
  = (approximation of) Hessian matrix of Loss, evaluated at x_start`.
* <b>`x_start`</b>: (Batch of) vector-shaped, `float` `Tensor` representing the current
  value of the argument to the Loss function.
* <b>`tolerance`</b>: scalar, `float` `Tensor` representing the convergence threshold.
  The optimization step will terminate early, returning its current value of
  `x_start + x_update`, once the following condition is met:
  `||x_update_end - x_update_start||_2 / (1 + ||x_start||_2)
  < sqrt(tolerance)`,
  where `x_update_end` is the value of `x_update` at the end of a sweep and
  `x_update_start` is the value of `x_update` at the beginning of that
  sweep.
* <b>`l1_regularizer`</b>: scalar, `float` `Tensor` representing the weight of the L1
  regularization term (see equation above).  If L1 regularization is not
  required, then <a href="../../tfp/glm/fit_one_step.md"><code>tfp.glm.fit_one_step</code></a> is preferable.
* <b>`l2_regularizer`</b>: scalar, `float` `Tensor` representing the weight of the L2
  regularization term (see equation above).
  Default value: `None` (i.e., no L2 regularization).
* <b>`maximum_full_sweeps`</b>: Python integer specifying maximum number of sweeps to
  run.  A "sweep" consists of an iteration of coordinate descent on each
  coordinate. After this many sweeps, the algorithm will terminate even if
  convergence has not been reached.
  Default value: `1`.
* <b>`learning_rate`</b>: scalar, `float` `Tensor` representing a multiplicative factor
  used to dampen the proximal gradient descent steps.
  Default value: `None` (i.e., factor is conceptually `1`).
* <b>`name`</b>: Python string representing the name of the TensorFlow operation.
  The default name is `"minimize_one_step"`.


#### Returns:


* <b>`x`</b>: (Batch of) `Tensor` having the same shape and dtype as `x_start`,
  representing the updated value of `x`, that is, `x_start + x_update`.
* <b>`is_converged`</b>: scalar, `bool` `Tensor` indicating whether convergence
  occurred across all batches within the specified number of sweeps.
* <b>`iter`</b>: scalar, `int` `Tensor` representing the actual number of coordinate
  updates made (before achieving convergence).  Since each sweep consists of
  `tf.size(x_start)` iterations, the maximum number of updates is
  `maximum_full_sweeps * tf.size(x_start)`.

#### References

[1]: Jerome Friedman, Trevor Hastie and Rob Tibshirani. Regularization Paths
     for Generalized Linear Models via Coordinate Descent. _Journal of
     Statistical Software_, 33(1), 2010.
     https://www.jstatsoft.org/article/view/v033i01/v33i01.pdf

[2]: Guo-Xun Yuan, Chia-Hua Ho and Chih-Jen Lin. An Improved GLMNET for
     L1-regularized Logistic Regression. _Journal of Machine Learning
     Research_, 13, 2012.
     http://www.jmlr.org/papers/volume13/yuan12a/yuan12a.pdf