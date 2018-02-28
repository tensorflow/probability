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
  import tensorflow as tf
  import tensorflow_probability as tfp

  # Create fictitious training data.
  n = 3000   # number of samples
  x_size = 4   # size of single x
  y_size = 2    # size of single y
  x = np.random.randn(n, x_size)
  w = np.random.randn(y_size, x_size)
  b = np.random.randn(y_size, 1)
  y = np.tensordot(x, w, axes=[[-1], [-1]]) + b

  # Build TF graph for fitting MVN.
  mvn = tfp.trainable_distribution.multivariate_normal_tril(x, dims=y_size)
  loss = -tf.reduce_mean(mvn.log_prob(y))
  train_op = tf.train.AdamOptimizer(learning_rate=2.**-5).minimize(loss)
  mse = tf.reduce_mean(tf.squared_difference(y, mvn.loc))  # Monitor convergence
  tf.global_variables_initializer().run()

  # Run graph 1000 times.
  loss_ = np.zeros(1000)
  mse_ = np.zeros(1000)
  for it in xrange(loss_.size):
    [_, loss_[it], mse_[it]] = sess.run([train_op, loss, mse])
    if it % 200 == 0 or it == loss_.size - 1:
      print("iteration:{}  loss:{}  mse:{}".format(it, loss_[it], mse_[it]))

  # ==> iteration:0    loss:956.214439873  mse:2.36651925571
  #     iteration:200  loss:2.79611814404  mse:0.784222219306
  #     iteration:400  loss:2.14970138114  mse:0.504379955415
  #     iteration:600  loss:1.7965903912   mse:0.311770522879
  #     iteration:800  loss:1.52653481953  mse:0.178522642874
  #     iteration:999  loss:1.27916316673  mse:0.0912386968493
  ```

  Args:
    x: `Tensor` with floating type. Must have statically defined rank and
      statically known right-most dimension.
    dims: Scalar, `int`, `Tensor` indicated the MVN event size, i.e., the
      created MVN will be distribution over length-`dims` vectors.
    layer_fn: Python `callable` which takes input `x` and `int` scalar `d` and
      returns a transformation of `x` with size `tf.concat([tf.shape(x)[:-1],
      [d]], axis=0)`.
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
