<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.trainable_distributions.bernoulli" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.trainable_distributions.bernoulli

Constructs a trainable `tfd.Bernoulli` distribution.

``` python
tfp.trainable_distributions.bernoulli(
    x,
    layer_fn=tf.compat.v1.layers.dense,
    name=None
)
```



Defined in [`python/trainable_distributions/trainable_distributions_lib.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/trainable_distributions/trainable_distributions_lib.py).

<!-- Placeholder for "Used in" -->

This function creates a Bernoulli distribution parameterized by logits.
Using default args, this function is mathematically equivalent to:

```none
Y = Bernoulli(logits=matmul(W, x) + b)

where,
  W in R^[d, n]
  b in R^d
```

#### Examples

This function can be used as a [logistic regression](
https://en.wikipedia.org/wiki/Logistic_regression) loss.

```python
# This example fits a logistic regression loss.
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
  true_logits = np.tensordot(x, w, axes=[[-1], [-1]]) + b
  noise = np.random.logistic(size=n).astype(dtype)
  y = dtype(true_logits + noise > 0.)
  return y, x
y, x = make_training_data()

# Build TF graph for fitting Bernoulli maximum likelihood estimator.
bernoulli = tfp.trainable_distributions.bernoulli(x)
loss = -tf.reduce_mean(bernoulli.log_prob(y))
train_op = tf.train.AdamOptimizer(learning_rate=2.**-5).minimize(loss)
mse = tf.reduce_mean(tf.squared_difference(y, bernoulli.mean()))
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

# ==> iteration:0    loss:0.635675370693  mse:0.222526371479
#     iteration:200  loss:0.440077394247  mse:0.143687799573
#     iteration:400  loss:0.440077394247  mse:0.143687844276
#     iteration:600  loss:0.440077394247  mse:0.143687844276
#     iteration:800  loss:0.440077424049  mse:0.143687844276
#     iteration:999  loss:0.440077424049  mse:0.143687844276
```

#### Args:


* <b>`x`</b>: `Tensor` with floating type. Must have statically defined rank and
  statically known right-most dimension.
* <b>`layer_fn`</b>: Python `callable` which takes input `x` and `int` scalar `d` and
  returns a transformation of `x` with shape
  `tf.concat([tf.shape(x)[:-1], [1]], axis=0)`.
  Default value: `tf.layers.dense`.
* <b>`name`</b>: A `name_scope` name for operations created by this function.
  Default value: `None` (i.e., "bernoulli").


#### Returns:


* <b>`bernoulli`</b>: An instance of `tfd.Bernoulli`.