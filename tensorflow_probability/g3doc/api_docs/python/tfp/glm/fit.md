<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.glm.fit" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.glm.fit

Runs multiple Fisher scoring steps.

``` python
tfp.glm.fit(
    model_matrix,
    response,
    model,
    model_coefficients_start=None,
    predicted_linear_response_start=None,
    l2_regularizer=None,
    dispersion=None,
    offset=None,
    convergence_criteria_fn=None,
    learning_rate=None,
    fast_unsafe_numerics=True,
    maximum_iterations=None,
    name=None
)
```



Defined in [`python/glm/fisher_scoring.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/glm/fisher_scoring.py).

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`model_matrix`</b>: (Batch of) `float`-like, matrix-shaped `Tensor` where each row
  represents a sample's features.
* <b>`response`</b>: (Batch of) vector-shaped `Tensor` where each element represents a
  sample's observed response (to the corresponding row of features). Must
  have same `dtype` as `model_matrix`.
* <b>`model`</b>: <a href="../../tfp/glm/ExponentialFamily.md"><code>tfp.glm.ExponentialFamily</code></a>-like instance which implicitly
  characterizes a negative log-likelihood loss by specifying the
  distribuion's `mean`, `gradient_mean`, and `variance`.
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
* <b>`convergence_criteria_fn`</b>: Python `callable` taking:
  `is_converged_previous`, `iter_`, `model_coefficients_previous`,
  `predicted_linear_response_previous`, `model_coefficients_next`,
  `predicted_linear_response_next`, `response`, `model`, `dispersion` and
  returning a `bool` `Tensor` indicating that Fisher scoring has converged.
  See `convergence_criteria_small_relative_norm_weights_change` as an
  example function.
  Default value: `None` (i.e.,
  `convergence_criteria_small_relative_norm_weights_change`).
* <b>`learning_rate`</b>: Optional (batch of) scalar `Tensor` used to dampen iterative
  progress. Typically only needed if optimization diverges, should be no
  larger than `1` and typically very close to `1`.
  Default value: `None` (i.e., `1`).
* <b>`fast_unsafe_numerics`</b>: Optional Python `bool` indicating if faster, less
  numerically accurate methods can be employed for computing the weighted
  least-squares solution.
  Default value: `True` (i.e., "fast but possibly diminished accuracy").
* <b>`maximum_iterations`</b>: Optional maximum number of iterations of Fisher scoring
  to run; "and-ed" with result of `convergence_criteria_fn`.
  Default value: `None` (i.e., `infinity`).
* <b>`name`</b>: Python `str` used as name prefix to ops created by this function.
  Default value: `"fit"`.


#### Returns:


* <b>`model_coefficients`</b>: (Batch of) vector-shaped `Tensor`; represents the
  fitted model coefficients, one for each column in `model_matrix`.
* <b>`predicted_linear_response`</b>: `response`-shaped `Tensor` representing linear
  predictions based on new `model_coefficients`, i.e.,
  `tf.linalg.matvec(model_matrix, model_coefficients) + offset`.
* <b>`is_converged`</b>: `bool` `Tensor` indicating that the returned
  `model_coefficients` met the `convergence_criteria_fn` criteria within the
  `maximum_iterations` limit.
* <b>`iter_`</b>: `int32` `Tensor` indicating the number of iterations taken.

#### Example

```python
from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def make_dataset(n, d, link, scale=1., dtype=np.float32):
  model_coefficients = tfd.Uniform(
      low=np.array(-1, dtype),
      high=np.array(1, dtype)).sample(d, seed=42)
  radius = np.sqrt(2.)
  model_coefficients *= radius / tf.linalg.norm(model_coefficients)
  model_matrix = tfd.Normal(
      loc=np.array(0, dtype),
      scale=np.array(1, dtype)).sample([n, d], seed=43)
  scale = tf.convert_to_tensor(scale, dtype)
  linear_response = tf.tensordot(
      model_matrix, model_coefficients, axes=[[1], [0]])
  if link == 'linear':
    response = tfd.Normal(loc=linear_response, scale=scale).sample(seed=44)
  elif link == 'probit':
    response = tf.cast(
        tfd.Normal(loc=linear_response, scale=scale).sample(seed=44) > 0,
        dtype)
  elif link == 'logit':
    response = tfd.Bernoulli(logits=linear_response).sample(seed=44)
  else:
    raise ValueError('unrecognized true link: {}'.format(link))
  return model_matrix, response, model_coefficients

X, Y, w_true = make_dataset(n=int(1e6), d=100, link='probit')

w, linear_response, is_converged, num_iter = tfp.glm.fit(
    model_matrix=X,
    response=Y,
    model=tfp.glm.BernoulliNormalCDF())
log_likelihood = tfp.glm.BernoulliNormalCDF().log_prob(Y, linear_response)

with tf.Session() as sess:
  [w_, linear_response_, is_converged_, num_iter_, Y_, w_true_,
   log_likelihood_] = sess.run([
      w, linear_response, is_converged, num_iter, Y, w_true,
      log_likelihood])

print('is_converged: ', is_converged_)
print('    num_iter: ', num_iter_)
print('    accuracy: ', np.mean((linear_response_ > 0.) == Y_))
print('    deviance: ', 2. * np.mean(log_likelihood_))
print('||w0-w1||_2 / (1+||w0||_2): ', (np.linalg.norm(w_true_ - w_, ord=2) /
                                       (1. + np.linalg.norm(w_true_, ord=2))))

# ==>
# is_converged:  True
#     num_iter:  6
#     accuracy:  0.804382
#     deviance:  -0.820746600628
# ||w0-w1||_2 / (1+||w0||_2):  0.00619245105309
```