<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.trainable_distributions.multivariate_normal_tril" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.trainable_distributions.multivariate_normal_tril

Constructs a trainable `tfd.MultivariateNormalTriL` distribution.

``` python
tfp.trainable_distributions.multivariate_normal_tril(
    x,
    dims,
    layer_fn=tf.compat.v1.layers.dense,
    loc_fn=(lambda x: x),
    scale_fn=tfp.trainable_distributions.tril_with_diag_softplus_and_shift,
    name=None
)
```



Defined in [`python/trainable_distributions/trainable_distributions_lib.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/trainable_distributions/trainable_distributions_lib.py).

<!-- Placeholder for "Used in" -->

This function creates a MultivariateNormal (MVN) with lower-triangular scale
matrix. By default the MVN is parameterized via affine transformation of input
tensor `x`. Using default args, this function is mathematically equivalent to:

```none
Y = MVN(loc=matmul(W, x) + b,
        scale_tril=f(reshape_tril(matmul(M, x) + c)))

where,
  W in R^[d, n]
  M in R^[d*(d+1)/2, n]
  b in R^d
  c in R^d
  f(S) = set_diag(S, softplus(matrix_diag_part(S)) + 1e-5)
```

Observe that `f` makes the diagonal of the triangular-lower scale matrix be
positive and no smaller than `1e-5`.

#### Examples

```python
# This example fits a multilinear regression loss.
import tensorflow as tf
import tensorflow_probability as tfp

# Create fictitious training data.
dtype = np.float32
n = 3000    # number of samples
x_size = 4  # size of single x
y_size = 2  # size of single y
def make_training_data():
  np.random.seed(142)
  x = np.random.randn(n, x_size).astype(dtype)
  w = np.random.randn(x_size, y_size).astype(dtype)
  b = np.random.randn(1, y_size).astype(dtype)
  true_mean = np.tensordot(x, w, axes=[[-1], [0]]) + b
  noise = np.random.randn(n, y_size).astype(dtype)
  y = true_mean + noise
  return y, x
y, x = make_training_data()

# Build TF graph for fitting MVNTriL maximum likelihood estimator.
mvn = tfp.trainable_distributions.multivariate_normal_tril(x, dims=y_size)
loss = -tf.reduce_mean(mvn.log_prob(y))
train_op = tf.train.AdamOptimizer(learning_rate=2.**-3).minimize(loss)
mse = tf.reduce_mean(tf.squared_difference(y, mvn.mean()))
init_op = tf.global_variables_initializer()

# Run graph 1000 times.
num_steps = 1000
loss_ = np.zeros(num_steps)   # Style: `_` to indicate sess.run result.
mse_ = np.zeros(num_steps)
with tf.Session() as sess:
  sess.run(init_op)
  for it in xrange(loss_.size):
    _, loss_[it], mse_[it] = sess.run([train_op, loss, mse])
    if it % 200 == 0 or it == loss_.size - 1:
      print("iteration:{}  loss:{}  mse:{}".format(it, loss_[it], mse_[it]))

# ==> iteration:0    loss:38.2020797729  mse:4.17175960541
#     iteration:200  loss:2.90179634094  mse:0.990987896919
#     iteration:400  loss:2.82727336884  mse:0.990926623344
#     iteration:600  loss:2.82726788521  mse:0.990926682949
#     iteration:800  loss:2.82726788521  mse:0.990926682949
#     iteration:999  loss:2.82726788521  mse:0.990926682949
```

#### Args:


* <b>`x`</b>: `Tensor` with floating type. Must have statically defined rank and
  statically known right-most dimension.
* <b>`dims`</b>: Scalar, `int`, `Tensor` indicated the MVN event size, i.e., the
  created MVN will be distribution over length-`dims` vectors.
* <b>`layer_fn`</b>: Python `callable` which takes input `x` and `int` scalar `d` and
  returns a transformation of `x` with shape
  `tf.concat([tf.shape(x)[:-1], [d]], axis=0)`.
  Default value: `tf.layers.dense`.
* <b>`loc_fn`</b>: Python `callable` which transforms the `loc` parameter. Takes a
  (batch of) length-`dims` vectors and returns a `Tensor` of same shape and
  `dtype`.
  Default value: `lambda x: x`.
* <b>`scale_fn`</b>: Python `callable` which transforms the `scale` parameters. Takes a
  (batch of) length-`dims * (dims + 1) / 2` vectors and returns a
  lower-triangular `Tensor` of same batch shape with rightmost dimensions
  having shape `[dims, dims]`.
  Default value: `tril_with_diag_softplus_and_shift`.
* <b>`name`</b>: A `name_scope` name for operations created by this function.
  Default value: `None` (i.e., "multivariate_normal_tril").


#### Returns:


* <b>`mvntril`</b>: An instance of `tfd.MultivariateNormalTriL`.