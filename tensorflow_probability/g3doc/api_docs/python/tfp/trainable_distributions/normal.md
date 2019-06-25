<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.trainable_distributions.normal" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.trainable_distributions.normal

Constructs a trainable `tfd.Normal` distribution.

``` python
tfp.trainable_distributions.normal(
    x,
    layer_fn=tf.compat.v1.layers.dense,
    loc_fn=(lambda x: x),
    scale_fn=1.0,
    name=None
)
```



Defined in [`python/trainable_distributions/trainable_distributions_lib.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/trainable_distributions/trainable_distributions_lib.py).

<!-- Placeholder for "Used in" -->


This function creates a Normal distribution parameterized by loc and scale.
Using default args, this function is mathematically equivalent to:

```none
Y = Normal(loc=matmul(W, x) + b, scale=1)

where,
  W in R^[d, n]
  b in R^d
```

#### Examples

This function can be used as a [linear regression](
https://en.wikipedia.org/wiki/Linear_regression) loss.

```python
# This example fits a linear regression loss.
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
  true_mean = np.tensordot(x, w, axes=[[-1], [-1]]) + b
  noise = np.random.randn(n).astype(dtype)
  y = true_mean + noise
  return y, x
y, x = make_training_data()

# Build TF graph for fitting Normal maximum likelihood estimator.
normal = tfp.trainable_distributions.normal(x)
loss = -tf.reduce_mean(normal.log_prob(y))
train_op = tf.train.AdamOptimizer(learning_rate=2.**-5).minimize(loss)
mse = tf.reduce_mean(tf.squared_difference(y, normal.mean()))
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

# ==> iteration:0    loss:6.34114170074  mse:10.8444051743
#     iteration:200  loss:1.40146839619  mse:0.965059816837
#     iteration:400  loss:1.40052902699  mse:0.963181257248
#     iteration:600  loss:1.40052902699  mse:0.963181257248
#     iteration:800  loss:1.40052902699  mse:0.963181257248
#     iteration:999  loss:1.40052902699  mse:0.963181257248
```

#### Args:


* <b>`x`</b>: `Tensor` with floating type. Must have statically defined rank and
  statically known right-most dimension.
* <b>`layer_fn`</b>: Python `callable` which takes input `x` and `int` scalar `d` and
  returns a transformation of `x` with shape
  `tf.concat([tf.shape(x)[:-1], [1]], axis=0)`.
  Default value: `tf.layers.dense`.
* <b>`loc_fn`</b>: Python `callable` which transforms the `loc` parameter. Takes a
  (batch of) length-`dims` vectors and returns a `Tensor` of same shape and
  `dtype`.
  Default value: `lambda x: x`.
* <b>`scale_fn`</b>: Python `callable` or `Tensor`. If a `callable` transforms the
  `scale` parameters; if `Tensor` is the `tfd.Normal` `scale` argument.
  Takes a (batch of) length-`dims` vectors and returns a `Tensor` of same
  size. (Taking a `callable` or `Tensor` is how `tf.Variable` intializers
  behave.)
  Default value: `1`.
* <b>`name`</b>: A `name_scope` name for operations created by this function.
  Default value: `None` (i.e., "normal").


#### Returns:


* <b>`normal`</b>: An instance of `tfd.Normal`.