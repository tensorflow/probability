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

"Trainable distributions" are instances of `tf.contrib.distributions` which are
parameterized by a transformation of a single input `Tensor`. The
transformations are presumed to use TensorFlow variables and typically need to
be fit, e.g., using `tf.train` optimizers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

tfd = tf.contrib.distributions


__all__ = [
    'bernoulli',
    'multivariate_normal_tril',
    'positive_tril_with_diag_shift',
]


def positive_tril_with_diag_shift(x, diag_shift=1e-5, name=None):
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
      Default value: `None` (i.e., "positive_tril_with_diag_shift").

  Returns:
    scale_tril: (Batch of) lower-triangular `Tensor` with `x.dtype` and
      rightmost shape `[dims, dims]` where `n = dims * (dims + 1) / 2` where
      `n = x.shape[-1]`.
  """
  with tf.name_scope(name, 'positive_tril_with_diag_shift', [x, diag_shift]):
    x = tf.convert_to_tensor(x, name='x')
    x = tfd.fill_triangular(x)
    diag = tf.nn.softplus(tf.matrix_diag_part(x))
    if diag_shift is not None:
      diag += diag_shift
    x = tf.matrix_set_diag(x, diag)
    return x


def multivariate_normal_tril(
    x,
    dims,
    layer_fn=tf.layers.dense,
    loc_fn=lambda x: x,
    scale_fn=positive_tril_with_diag_shift,
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
  np.random.seed(142)
  x = np.random.randn(n, x_size).astype(dtype)
  w = np.random.randn(x_size, y_size).astype(dtype)
  b = np.random.randn(1, y_size).astype(dtype)
  y = np.tensordot(x, w, axes=[[-1], [0]]) + b

  # Build TF graph for fitting MVN.
  mvn = tfp.trainable_distribution.multivariate_normal_tril(x, dims=y_size)
  loss = -tf.reduce_mean(mvn.log_prob(y))
  train_op = tf.train.AdamOptimizer(learning_rate=2.**-5).minimize(loss)
  mse = tf.reduce_mean(tf.squared_difference(y, mvn.mean()))
  init_op = tf.global_variables_initializer()

  # Run graph 1000 times.
  loss_ = np.zeros(1000)
  mse_ = np.zeros(1000)
  init_op.run()
  for it in xrange(loss_.size):
    _, loss_[it], mse_[it] = sess.run([train_op, loss, mse])
    if it % 200 == 0 or it == loss_.size - 1:
      print("iteration:{}  loss:{}  mse:{}".format(it, loss_[it], mse_[it]))

  # ==> iteration:0    loss:670.471557617  mse:3.88558006287
  #     iteration:200  loss:3.4884390831   mse:1.78182530403
  #     iteration:400  loss:2.4829223156   mse:1.04152262211
  #     iteration:600  loss:1.9646422863   mse:0.571185767651
  #     iteration:800  loss:1.59305691719  mse:0.276959866285
  #     iteration:999  loss:1.26750314236  mse:0.112104855478
  ```

  Args:
    x: `Tensor` with floating type. Must have statically defined rank and
      statically known right-most dimension.
    dims: Scalar, `int`, `Tensor` indicated the MVN event size, i.e., the
      created MVN will be distribution over length-`dims` vectors.
    layer_fn: Python `callable` which takes input `x` and `int` scalar `d` and
      returns a transformation of `x` with size
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
      Default value: `positive_tril_with_diag_shift`.
    name: A `name_scope` name for operations created by this function.
      Default value: `None` (i.e., "multivariate_normal_tril").

  Returns:
    mvntril: An instance of `tfd.MultivariateNormalTriL`.
  """
  with tf.name_scope(name, 'multivariate_normal_tril', [x, dims]):
    x = tf.convert_to_tensor(x, name='x')
    x = layer_fn(x, dims + dims * (dims + 1) // 2)
    return tfd.MultivariateNormalTriL(
        loc=loc_fn(x[..., :dims]),
        scale_tril=scale_fn(x[..., dims:]))


def bernoulli(x, layer_fn=tf.layers.dense, name=None):
  """Constructs a trainable `tfd.Bernoulli` distribution.

  This function creates a distribution suitable for [logistic regression](
  https://en.wikipedia.org/wiki/Logistic_regression).

  This function creates a Bernoulli distribution parameterized by logits.
  Using default args, this function is mathematically equivalent to:

  ```none
  Y = Bernoulli(logits=matmul(W, x) + b)

  where,
    W in R^[d, n]
    b in R^d
  ```

  #### Examples

  ```python
  # This example fits a logistic regression loss.
  import tensorflow as tf
  import tensorflow_probability as tfp

  # Create fictitious training data.
  dtype = np.float32
  n = 3000    # number of samples
  x_size = 4  # size of single x
  np.random.seed(142)
  x = np.random.randn(n, x_size).astype(dtype)
  w = np.random.randn(x_size).astype(dtype)
  b = np.random.randn(1).astype(dtype)
  y = dtype(np.tensordot(x, w, axes=[[-1], [-1]]) + b > 0.)

  # Build TF graph for fitting Bernoulli.
  bernoulli = tfp.trainable_distribution.bernoulli(x)
  loss = -tf.reduce_mean(bernoulli.log_prob(y))
  train_op = tf.train.AdamOptimizer(learning_rate=2.**-5).minimize(loss)
  mse = tf.reduce_mean(tf.squared_difference(y, bernoulli.mean()))
  init_op = tf.global_variables_initializer()

  # Run graph 1000 times.
  loss_ = np.zeros(1000)
  mse_ = np.zeros(1000)
  init_op.run()
  for it in xrange(loss_.size):
    _, loss_[it], mse_[it] = sess.run([train_op, loss, mse])
    if it % 200 == 0 or it == loss_.size - 1:
      print("iteration:{}  loss:{}  mse:{}".format(it, loss_[it], mse_[it]))

  # ==> iteration:0    loss:1.17989099026    mse:0.4212923944
  #     iteration:200  loss:0.213382124901   mse:0.0547544509172
  #     iteration:400  loss:0.14739997685    mse:0.0365632660687
  #     iteration:600  loss:0.118733644485   mse:0.0290872063488
  #     iteration:800  loss:0.101618662477   mse:0.0247089788318
  #     iteration:999  loss:0.0898892953992  mse:0.0217369645834
  ```

  Args:
    x: `Tensor` with floating type. Must have statically defined rank and
      statically known right-most dimension.
    layer_fn: Python `callable` which takes input `x` and `int` scalar `d` and
      returns a transformation of `x` with size
      `tf.concat([tf.shape(x)[:-1], [1]], axis=0)`.
      Default value: `tf.layers.dense`.
    name: A `name_scope` name for operations created by this function.
      Default value: `None` (i.e., "bernoulli").

  Returns:
    bernoulli: An instance of `tfd.Bernoulli`.
  """
  with tf.name_scope(name, 'bernoulli', [x]):
    x = tf.convert_to_tensor(x, name='x')
    logits = tf.squeeze(layer_fn(x, 1), axis=-1)
    return tfd.Bernoulli(logits=logits)
