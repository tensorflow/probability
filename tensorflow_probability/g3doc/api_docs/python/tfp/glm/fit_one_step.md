<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.glm.fit_one_step" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.glm.fit_one_step

``` python
tfp.glm.fit_one_step(
    model_matrix,
    response,
    model,
    model_coefficients_start=None,
    predicted_linear_response_start=None,
    l2_regularizer=None,
    dispersion=None,
    offset=None,
    learning_rate=None,
    fast_unsafe_numerics=True,
    name=None
)
```

Runs one step of Fisher scoring.

#### Args:

* <b>`model_matrix`</b>: (Batch of) `float`-like, matrix-shaped `Tensor` where each row
    represents a sample's features.
* <b>`response`</b>: (Batch of) vector-shaped `Tensor` where each element represents a
    sample's observed response (to the corresponding row of features). Must
    have same `dtype` as `model_matrix`.
* <b>`model`</b>: <a href="../../tfp/glm/ExponentialFamily.md"><code>tfp.glm.ExponentialFamily</code></a>-like instance used to construct the
    negative log-likelihood loss, gradient, and expected Hessian (i.e., the
    Fisher information matrix).
* <b>`model_coefficients_start`</b>: Optional (batch of) vector-shaped `Tensor`
    representing the initial model coefficients, one for each column in
    `model_matrix`. Must have same `dtype` as `model_matrix`.
    Default value: Zeros.
* <b>`predicted_linear_response_start`</b>: Optional `Tensor` with `shape`, `dtype`
    matching `response`; represents `offset` shifted initial linear
    predictions based on `model_coefficients_start`.
    Default value: `offset` if `model_coefficients is None`, and
    `tf.linalg.matvec(model_matrix, model_coefficients_start) + offset`
    otherwise.
* <b>`l2_regularizer`</b>: Optional scalar `Tensor` representing L2 regularization
    penalty, i.e.,
    `loss(w) = sum{-log p(y[i]|x[i],w) : i=1..n} + l2_regularizer ||w||_2^2`.
    Default value: `None` (i.e., no L2 regularization).
* <b>`dispersion`</b>: Optional (batch of) `Tensor` representing `response` dispersion,
    i.e., as in, `p(y|theta) := exp((y theta - A(theta)) / dispersion)`.
    Must broadcast with rows of `model_matrix`.
    Default value: `None` (i.e., "no dispersion").
* <b>`offset`</b>: Optional `Tensor` representing constant shift applied to
    `predicted_linear_response`.  Must broadcast to `response`.
    Default value: `None` (i.e., `tf.zeros_like(response)`).
* <b>`learning_rate`</b>: Optional (batch of) scalar `Tensor` used to dampen iterative
    progress. Typically only needed if optimization diverges, should be no
    larger than `1` and typically very close to `1`.
    Default value: `None` (i.e., `1`).
* <b>`fast_unsafe_numerics`</b>: Optional Python `bool` indicating if solve should be
    based on Cholesky or QR decomposition.
    Default value: `True` (i.e., "prefer speed via Cholesky decomposition").
* <b>`name`</b>: Python `str` used as name prefix to ops created by this function.
    Default value: `"fit_one_step"`.


#### Returns:

* <b>`model_coefficients`</b>: (Batch of) vector-shaped `Tensor`; represents the
    next estimate of the model coefficients, one for each column in
    `model_matrix`.
* <b>`predicted_linear_response`</b>: `response`-shaped `Tensor` representing linear
    predictions based on new `model_coefficients`, i.e.,
    `tf.linalg.matvec(model_matrix, model_coefficients_next) + offset`.