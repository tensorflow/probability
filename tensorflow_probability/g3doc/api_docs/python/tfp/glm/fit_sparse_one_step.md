<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.glm.fit_sparse_one_step" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.glm.fit_sparse_one_step

``` python
tfp.glm.fit_sparse_one_step(
    model_matrix,
    response,
    model,
    model_coefficients_start,
    tolerance,
    l1_regularizer,
    l2_regularizer=None,
    maximum_full_sweeps=None,
    learning_rate=None,
    name=None
)
```

One step of (the outer loop of) the GLM fitting algorithm.

This function returns a new value of `model_coefficients`, equal to
`model_coefficients_start + model_coefficients_update`.  The increment
`model_coefficients_update in R^n` is computed by a coordinate descent method,
that is, by a loop in which each iteration updates exactly one coordinate of
`model_coefficients_update`.  (Some updates may leave the value of the
coordinate unchanged.)

The particular update method used is to apply an L1-based proximity operator,
"soft threshold", whose fixed point `model_coefficients_update^*` is the
desired minimum

```none
model_coefficients_update^* = argmin{
    -LogLikelihood(model_coefficients_start + model_coefficients_update')
      + l1_regularizer *
          ||model_coefficients_start + model_coefficients_update'||_1
      + l2_regularizer *
          ||model_coefficients_start + model_coefficients_update'||_2**2
    : model_coefficients_update' }
```

where in each iteration `model_coefficients_update'` has at most one nonzero
coordinate.

This update method preserves sparsity, i.e., tends to find sparse solutions if
`model_coefficients_start` is sparse.  Additionally, the choice of step size
is based on curvature (Fisher information matrix), which significantly speeds
up convergence.

#### Args:

* <b>`model_matrix`</b>: (Batch of) matrix-shaped, `float` `Tensor` or `SparseTensor`
    where each row represents a sample's features.  Has shape `[N, n]` where
    `N` is the number of data samples and `n` is the number of features per
    sample.
* <b>`response`</b>: (Batch of) vector-shaped `Tensor` with the same dtype as
    `model_matrix` where each element represents a sample's observed response
    (to the corresponding row of features).
* <b>`model`</b>: <a href="../../tfp/glm/ExponentialFamily.md"><code>tfp.glm.ExponentialFamily</code></a>-like instance, which specifies the link
    function and distribution of the GLM, and thus characterizes the negative
    log-likelihood which will be minimized. Must have sufficient statistic
    equal to the response, that is, `T(y) = y`.
* <b>`model_coefficients_start`</b>: (Batch of) vector-shaped, `float` `Tensor` with
    the same dtype as `model_matrix`, representing the initial values of the
    coefficients for the GLM regression.  Has shape `[n]` where `model_matrix`
    has shape `[N, n]`.
* <b>`tolerance`</b>: scalar, `float` `Tensor` representing the convergence threshold.
    The optimization step will terminate early, returning its current value of
    `model_coefficients_start + model_coefficients_update`, once the following
    condition is met:
    `||model_coefficients_update_end - model_coefficients_update_start||_2
       / (1 + ||model_coefficients_start||_2)
     < sqrt(tolerance)`,
    where `model_coefficients_update_end` is the value of
    `model_coefficients_update` at the end of a sweep and
    `model_coefficients_update_start` is the value of
    `model_coefficients_update` at the beginning of that sweep.
* <b>`l1_regularizer`</b>: scalar, `float` `Tensor` representing the weight of the L1
    regularization term (see equation above).
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
* <b>`name`</b>: Python string representing the name of the TensorFlow operation. The
    default name is `"fit_sparse_one_step"`.


#### Returns:

* <b>`model_coefficients`</b>: (Batch of) `Tensor` having the same shape and dtype as
    `model_coefficients_start`, representing the updated value of
    `model_coefficients`, that is, `model_coefficients_start +
    model_coefficients_update`.
* <b>`is_converged`</b>: scalar, `bool` `Tensor` indicating whether convergence
    occurred across all batches within the specified number of sweeps.
* <b>`iter`</b>: scalar, `int` `Tensor` representing the actual number of coordinate
    updates made (before achieving convergence).  Since each sweep consists of
    `tf.size(model_coefficients_start)` iterations, the maximum number of
    updates is `maximum_full_sweeps * tf.size(model_coefficients_start)`.