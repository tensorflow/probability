# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Trainable distributions.

"Trainable distributions" are instances of `tfp.distributions` which are
parameterized by a transformation of a single input `Tensor`. The
transformations are presumed to use TensorFlow variables and typically need to
be fit, e.g., using `tf.train` optimizers or `tfp.optimizers`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_probability.python import distributions as tfd


__all__ = [
    'bernoulli',
    'multivariate_normal_tril',
    'normal',
    'poisson',
    'softplus_and_shift',
    'tril_with_diag_softplus_and_shift',
]


def softplus_and_shift(x, shift=1e-5, name=None):
  """Converts (batch of) scalars to (batch of) positive valued scalars.

  Args:
    x: (Batch of) `float`-like `Tensor` representing scalars which will be
      transformed into positive elements.
    shift: `Tensor` added to `softplus` transformation of elements.
      Default value: `1e-5`.
    name: A `name_scope` name for operations created by this function.
      Default value: `None` (i.e., "positive_tril_with_shift").

  Returns:
    scale: (Batch of) scalars`with `x.dtype` and `x.shape`.
  """
  with tf.compat.v1.name_scope(name, 'softplus_and_shift', [x, shift]):
    x = tf.convert_to_tensor(value=x, name='x')
    y = tf.nn.softplus(x)
    if shift is not None:
      y += shift
    return y


def tril_with_diag_softplus_and_shift(x, diag_shift=1e-5, name=None):
  """Converts (batch of) vectors to (batch of) lower-triangular scale matrices.

  Args:
    x: (Batch of) `float`-like `Tensor` representing vectors which will be
      transformed into lower-triangular scale matrices with positive diagonal
      elements. Rightmost shape `n` must be such that
      `n = dims * (dims + 1) / 2` for some positive, integer `dims`.
    diag_shift: `Tensor` added to `softplus` transformation of diagonal
      elements.
      Default value: `1e-5`.
    name: A `name_scope` name for operations created by this function.
      Default value: `None` (i.e., "tril_with_diag_softplus_and_shift").

  Returns:
    scale_tril: (Batch of) lower-triangular `Tensor` with `x.dtype` and
      rightmost shape `[dims, dims]` where `n = dims * (dims + 1) / 2` where
      `n = x.shape[-1]`.
  """
  with tf.compat.v1.name_scope(name, 'tril_with_diag_softplus_and_shift',
                               [x, diag_shift]):
    x = tf.convert_to_tensor(value=x, name='x')
    x = tfd.fill_triangular(x)
    diag = softplus_and_shift(tf.linalg.diag_part(x), diag_shift)
    x = tf.linalg.set_diag(x, diag)
    return x


def multivariate_normal_tril(x,
                             dims,
                             layer_fn=tf.compat.v1.layers.dense,
                             loc_fn=lambda x: x,
                             scale_fn=tril_with_diag_softplus_and_shift,
                             name=None):
  """Constructs a trainable `tfd.MultivariateNormalTriL` distribution.

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

  Args:
    x: `Tensor` with floating type. Must have statically defined rank and
      statically known right-most dimension.
    dims: Scalar, `int`, `Tensor` indicated the MVN event size, i.e., the
      created MVN will be distribution over length-`dims` vectors.
    layer_fn: Python `callable` which takes input `x` and `int` scalar `d` and
      returns a transformation of `x` with shape
      `tf.concat([tf.shape(x)[:-1], [d]], axis=0)`.
      Default value: `tf.layers.dense`.
    loc_fn: Python `callable` which transforms the `loc` parameter. Takes a
      (batch of) length-`dims` vectors and returns a `Tensor` of same shape and
      `dtype`.
      Default value: `lambda x: x`.
    scale_fn: Python `callable` which transforms the `scale` parameters. Takes a
      (batch of) length-`dims * (dims + 1) / 2` vectors and returns a
      lower-triangular `Tensor` of same batch shape with rightmost dimensions
      having shape `[dims, dims]`.
      Default value: `tril_with_diag_softplus_and_shift`.
    name: A `name_scope` name for operations created by this function.
      Default value: `None` (i.e., "multivariate_normal_tril").

  Returns:
    mvntril: An instance of `tfd.MultivariateNormalTriL`.
  """
  with tf.compat.v1.name_scope(name, 'multivariate_normal_tril', [x, dims]):
    x = tf.convert_to_tensor(value=x, name='x')
    x = layer_fn(x, dims + dims * (dims + 1) // 2)
    return tfd.MultivariateNormalTriL(
        loc=loc_fn(x[..., :dims]),
        scale_tril=scale_fn(x[..., dims:]))


def bernoulli(x, layer_fn=tf.compat.v1.layers.dense, name=None):
  """Constructs a trainable `tfd.Bernoulli` distribution.

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

  Args:
    x: `Tensor` with floating type. Must have statically defined rank and
      statically known right-most dimension.
    layer_fn: Python `callable` which takes input `x` and `int` scalar `d` and
      returns a transformation of `x` with shape
      `tf.concat([tf.shape(x)[:-1], [1]], axis=0)`.
      Default value: `tf.layers.dense`.
    name: A `name_scope` name for operations created by this function.
      Default value: `None` (i.e., "bernoulli").

  Returns:
    bernoulli: An instance of `tfd.Bernoulli`.
  """
  with tf.compat.v1.name_scope(name, 'bernoulli', [x]):
    x = tf.convert_to_tensor(value=x, name='x')
    logits = tf.squeeze(layer_fn(x, 1), axis=-1)
    return tfd.Bernoulli(logits=logits)


def normal(x,
           layer_fn=tf.compat.v1.layers.dense,
           loc_fn=lambda x: x,
           scale_fn=1.,
           name=None):
  """Constructs a trainable `tfd.Normal` distribution.


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

  Args:
    x: `Tensor` with floating type. Must have statically defined rank and
      statically known right-most dimension.
    layer_fn: Python `callable` which takes input `x` and `int` scalar `d` and
      returns a transformation of `x` with shape
      `tf.concat([tf.shape(x)[:-1], [1]], axis=0)`.
      Default value: `tf.layers.dense`.
    loc_fn: Python `callable` which transforms the `loc` parameter. Takes a
      (batch of) length-`dims` vectors and returns a `Tensor` of same shape and
      `dtype`.
      Default value: `lambda x: x`.
    scale_fn: Python `callable` or `Tensor`. If a `callable` transforms the
      `scale` parameters; if `Tensor` is the `tfd.Normal` `scale` argument.
      Takes a (batch of) length-`dims` vectors and returns a `Tensor` of same
      size. (Taking a `callable` or `Tensor` is how `tf.Variable` intializers
      behave.)
      Default value: `1`.
    name: A `name_scope` name for operations created by this function.
      Default value: `None` (i.e., "normal").

  Returns:
    normal: An instance of `tfd.Normal`.
  """
  with tf.compat.v1.name_scope(name, 'normal', [x]):
    x = tf.convert_to_tensor(value=x, name='x')
    if callable(scale_fn):
      y = layer_fn(x, 2)
      loc = loc_fn(y[..., 0])
      scale = scale_fn(y[..., 1])
    else:
      y = tf.squeeze(layer_fn(x, 1), axis=-1)
      loc = loc_fn(y)
      scale = tf.cast(scale_fn, loc.dtype.base_dtype)
    return tfd.Normal(loc=loc, scale=scale)


def poisson(x,
            layer_fn=tf.compat.v1.layers.dense,
            log_rate_fn=lambda x: x,
            name=None):
  """Constructs a trainable `tfd.Poisson` distribution.

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

  Args:
    x: `Tensor` with floating type. Must have statically defined rank and
      statically known right-most dimension.
    layer_fn: Python `callable` which takes input `x` and `int` scalar `d` and
      returns a transformation of `x` with shape
      `tf.concat([tf.shape(x)[:-1], [1]], axis=0)`.
      Default value: `tf.layers.dense`.
    log_rate_fn: Python `callable` which transforms the `log_rate` parameter.
      Takes a (batch of) length-`dims` vectors and returns a `Tensor` of same
      shape and `dtype`.
      Default value: `lambda x: x`.
    name: A `name_scope` name for operations created by this function.
      Default value: `None` (i.e., "poisson").

  Returns:
    poisson: An instance of `tfd.Poisson`.
  """
  with tf.compat.v1.name_scope(name, 'poisson', [x]):
    x = tf.convert_to_tensor(value=x, name='x')
    log_rate = log_rate_fn(tf.squeeze(layer_fn(x, 1), axis=-1))
    return tfd.Poisson(log_rate=log_rate)
