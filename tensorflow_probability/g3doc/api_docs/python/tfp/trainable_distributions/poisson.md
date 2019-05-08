<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.trainable_distributions.poisson" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.trainable_distributions.poisson

Constructs a trainable `tfd.Poisson` distribution.

``` python
tfp.trainable_distributions.poisson(
    x,
    layer_fn=tf.compat.v1.layers.dense,
    log_rate_fn=(lambda x: x),
    name=None
)
```



Defined in [`python/trainable_distributions/trainable_distributions_lib.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/trainable_distributions/trainable_distributions_lib.py).

<!-- Placeholder for "Used in" -->

This function creates a Poisson distribution parameterized by log rate.
Using default args, this function is mathematically equivalent to:

```none
Y = Poisson(log_rate=matmul(W, x) + b)

where,
  W in R^[d, n]
  b in R^d
```

#### Examples

This can be used as a [Poisson regression](
https://en.wikipedia.org/wiki/Poisson_regression) loss.

```python
# This example fits a poisson regression loss.
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# Create fictitious training data.
dtype = np.float32
n = 3000    # number of samples
x_size = 4  # size of single x
def make_training_data():
  np.random.seed(142)
  x = np.random.randn(n, x_size).astype(dtype)
  w = np.random.randn(x_size).astype(dtype)
  b = np.random.randn(1).astype(dtype)
  true_log_rate = np.tensordot(x, w, axes=[[-1], [-1]]) + b
  y = np.random.poisson(lam=np.exp(true_log_rate)).astype(dtype)
  return y, x
y, x = make_training_data()

# Build TF graph for fitting Poisson maximum likelihood estimator.
poisson = tfp.trainable_distributions.poisson(x)
loss = -tf.reduce_mean(poisson.log_prob(y))
train_op = tf.train.AdamOptimizer(learning_rate=2.**-5).minimize(loss)
mse = tf.reduce_mean(tf.squared_difference(y, poisson.mean()))
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

# ==> iteration:0    loss:37.0814208984  mse:6359.41259766
#     iteration:200  loss:1.42010736465  mse:40.7654914856
#     iteration:400  loss:1.39027583599  mse:8.77660560608
#     iteration:600  loss:1.3902695179   mse:8.78443241119
#     iteration:800  loss:1.39026939869  mse:8.78443622589
#     iteration:999  loss:1.39026939869  mse:8.78444766998
```

#### Args:

* <b>`x`</b>: `Tensor` with floating type. Must have statically defined rank and
  statically known right-most dimension.
* <b>`layer_fn`</b>: Python `callable` which takes input `x` and `int` scalar `d` and
  returns a transformation of `x` with shape
  `tf.concat([tf.shape(x)[:-1], [1]], axis=0)`.
  Default value: `tf.layers.dense`.
* <b>`log_rate_fn`</b>: Python `callable` which transforms the `log_rate` parameter.
  Takes a (batch of) length-`dims` vectors and returns a `Tensor` of same
  shape and `dtype`.
  Default value: `lambda x: x`.
* <b>`name`</b>: A `name_scope` name for operations created by this function.
  Default value: `None` (i.e., "poisson").


#### Returns:

* <b>`poisson`</b>: An instance of `tfd.Poisson`.