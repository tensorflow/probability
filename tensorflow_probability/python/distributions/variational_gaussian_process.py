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
"""The VariationalGaussianProcess distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_probability.python.distributions import gaussian_process
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import mvn_linear_operator
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.internal import dtype_util

__all__ = [
    'VariationalGaussianProcess',
]


def _add_diagonal_shift(matrix, shift):
  return tf.linalg.set_diag(
      matrix, tf.linalg.diag_part(matrix) + shift, name='add_diagonal_shift')


def _solve_cholesky_factored_system(
    cholesky_factor, rhs, name=None):
  with tf.compat.v1.name_scope(name, '_solve_cholesky_factored_system',
                               values=[cholesky_factor, rhs]) as scope:
    cholesky_factor = tf.convert_to_tensor(value=cholesky_factor,
                                           name='cholesky_factor')
    rhs = tf.convert_to_tensor(value=rhs, name='rhs')
    lin_op = tf.linalg.LinearOperatorLowerTriangular(
        cholesky_factor, name=scope)
    return lin_op.solve(lin_op.solve(rhs), adjoint=True)


def _solve_cholesky_factored_system_vec(cholesky_factor, rhs, name=None):
  with tf.compat.v1.name_scope(name, '_solve_cholesky_factored_system',
                               values=[cholesky_factor, rhs]) as scope:
    cholesky_factor = tf.convert_to_tensor(
        value=cholesky_factor, name='cholesky_factor')
    rhs = tf.convert_to_tensor(value=rhs, name='rhs')
    lin_op = tf.linalg.LinearOperatorLowerTriangular(
        cholesky_factor, name=scope)
    return lin_op.solvevec(lin_op.solvevec(rhs), adjoint=True)


class VariationalGaussianProcess(
    mvn_linear_operator.MultivariateNormalLinearOperator):
  """Posterior predictive of a variational Gaussian process.

  This distribution implements the variational Gaussian process (VGP), as
  described in [Titsias, 2009][1] and [Hensman, 2013][2]. The VGP is an
  inducing point-based approximation of an exact GP posterior
  (see Mathematical Details, below). Ultimately, this Distribution class
  represents a marginal distrbution over function values at a
  collection of `index_points`. It is parameterized by

    - a kernel function,
    - a mean function,
    - the (scalar) observation noise variance of the normal likelihood,
    - a set of index points,
    - a set of inducing index points, and
    - the parameters of the (full-rank, Gaussian) variational posterior
      distribution over function values at the inducing points, conditional on
      some observations.

  A VGP is "trained" by selecting any kernel parameters, the locations of the
  inducing index points, and the variational parameters. [Titsias, 2009][1] and
  [Hensman, 2013][2] describe a variational lower bound on the marginal log
  likelihood of observed data, which this class offers through the
  `variational_loss` method (this is the negative lower bound, for convenience
  when plugging into a TF Optimizer's `minimize` function).
  Training may be done in minibatches.

  [Titsias, 2009][1] describes a closed form for the optimal variational
  parameters, in the case of sufficiently small observational data (ie,
  small enough to fit in memory but big enough to warrant approximating the GP
  posterior). A method to compute these optimal parameters in terms of the full
  observational data set is provided as a staticmethod,
  `optimal_variational_posterior`. It returns a
  `MultivariateNormalLinearOperator` instance with optimal location and
  scale parameters.

  #### Mathematical Details

  ##### Notation

  We will in general be concerned about three collections of index points, and
  it'll be good to give them names:

    * `x[1], ..., x[N]`: observation index points -- locations of our observed
      data.
    * `z[1], ..., z[M]`: inducing index points  -- locations of the
      "summarizing" inducing points
    * `t[1], ..., t[P]`: predictive index points -- locations where we are
      making posterior predictions based on observations and the variational
      parameters.

  To lighten notation, we'll use `X, Z, T` to denote the above collections.
  Similarly, we'll denote by `f(X)` the collection of function values at each of
  the `x[i]`, and by `Y`, the collection of (noisy) observed data at each `x[i].
  We'll denote kernel matrices generated from pairs of index points as `K_tt`,
  `K_xt`, `K_tz`, etc, e.g.,

  ```none
           | k(t[1], z[1])    k(t[1], z[2])  ...  k(t[1], z[M]) |
    K_tz = | k(t[2], z[1])    k(t[2], z[2])  ...  k(t[2], z[M]) |
           |      ...              ...                 ...      |
           | k(t[P], z[1])    k(t[P], z[2])  ...  k(t[P], z[M]) |
  ```

  ##### Preliminaries

  A Gaussian process is an indexed collection of random variables, any finite
  collection of which are jointly Gaussian. Typically, the index set is some
  finite-dimensional, real vector space, and indeed we make this assumption in
  what follows. The GP may then be thought of as a distribution over functions
  on the index set. Samples from the GP are functions *on the whole index set*;
  these can't be represented in finite compute memory, so one typically works
  with the marginals at a finite collection of index points. The properties of
  the GP are entirely determined by its mean function `m` and covariance
  function `k`. The generative process, assuming a mean-zero normal likelihood
  with stddev `sigma`, is

  ```none
    f ~ GP(m, k)

    Y | f(X) ~ Normal(f(X), sigma),   i = 1, ... , N
  ```

  In finite terms (ie, marginalizing out all but a finite number of f(X)'sigma),
  we can write

  ```none
    f(X) ~ MVN(loc=m(X), cov=K_xx)

    Y | f(X) ~ Normal(f(X), sigma),   i = 1, ... , N
  ```

  Posterior inference is possible in analytical closed form but becomes
  intractible as data sizes get large. See [Rasmussen, 2006][3] for details.

  ##### The VGP

  The VGP is an inducing point-based approximation of an exact GP posterior,
  where two approximating assumptions have been made:

    1. function values at non-inducing points are mutually independent
       conditioned on function values at the inducing points,
    2. the (expensive) posterior over function values at inducing points
       conditional on obseravtions is replaced with an arbitrary (learnable)
       full-rank Gaussian distribution,

       ```none
         q(f(Z)) = MVN(loc=m, scale=S),
       ```

       where `m` and `S` are parameters to be chosen by optimizing an evidence
       lower bound (ELBO).

  The posterior predictive distribution becomes

  ```none
    q(f(T)) = integral df(Z) p(f(T) | f(Z)) q(f(Z))
            = MVN(loc = A @ m, scale = B^(1/2))
  ```

  where

  ```none
    A = K_tz @ K_zz^-1
    B = K_tt - A @ (K_zz - S S^T) A^T
  ```

  ***The approximate posterior predictive distribution `q(f(T))` is what the
  `VariationalGaussianProcess` class represents.***

  Model selection in this framework entails choosing the kernel parameters,
  inducing point locations, and variational parameters. We do this by optimizing
  a variational lower bound on the marginal log likelihood of observed data. The
  lower bound takes the following form (see [Titsias, 2009][1] and
  [Hensman, 2013][2] for details on the derivation):

  ```none
    L(Z, m, S, Y) = (
        MVN(loc=(K_zx @ K_zz^-1) @ m, scale_diag=sigma).log_prob(Y) -
        (Tr(K_xx - K_zx @ K_zz^-1 @ K_xz) +
         Tr(S @ S^T @ K_zz^1 @ K_zx @ K_xz @ K_zz^-1)) / (2 * sigma^2) -
        KL(q(f(Z)) || p(f(Z))))
  ```

  where in the final KL term, `p(f(Z))` is the GP prior on inducing point
  function values. This variational lower bound can be computed on minibatches
  of the full data set `(X, Y)`. A method to compute the *negative* variational
  lower bound is implemented as `VariationalGaussianProcess.variational_loss`.

  ##### Optimal variational parameters

  As described in [Titsias, 2009][1], a closed form optimum for the variational
  location and scale parameters, `m` and `S`, can be computed when the
  observational data are not prohibitively voluminous. The
  `optimal_variational_posterior` function to computes the optimal variational
  posterior distribution over inducing point function values in terms of the GP
  parameters (mean and kernel functions), inducing point locations, observation
  index points, and observations. Note that the inducing index point locations
  must still be optimized even when these parameters are known functions of the
  inducing index points. The optimal parameters are computed as follows:

  ```none
    C = sigma^-2 (K_zz + K_zx @ K_xz)^-1

    optimal Gaussian covariance: K_zz @ C @ K_zz
    optimal Gaussian location: sigma^-2 K_zz @ C @ K_zx @ Y
  ```

  #### Usage Examples

  Here's an example of defining and training a VariationalGaussianProcess on
  some toy generated data.

  ```python
  # We'll use double precision throughout for better numerics.
  dtype = np.float64

  # Generate noisy data from a known function.
  f = lambda x: np.exp(-x[..., 0]**2 / 20.) * np.sin(1. * x[..., 0])
  true_observation_noise_variance_ = dtype(1e-1) ** 2

  num_training_points_ = 100
  x_train_ = np.stack(
      [np.random.uniform(-6., 0., [num_training_points_/ 2 , 1]),
       np.random.uniform(1., 10., [num_training_points_/ 2 , 1])],
      axis=0).astype(dtype)
  y_train_ = (f(x_train_) +
              np.random.normal(
                  0., np.sqrt(true_observation_noise_variance_),
                  [num_training_points_]).astype(dtype))

  # Create kernel with trainable parameters, and trainable observation noise
  # variance variable. Each of these is constrained to be positive.
  amplitude = (tf.nn.softplus(tf.Variable(-1., dtype=dtype, name='amplitude')))
  length_scale = (1e-5 +
                  tf.nn.softplus(
                      tf.Variable(-3., dtype=dtype, name='length_scale')))
  kernel = tfk.ExponentiatedQuadratic(
      amplitude=amplitude,
      length_scale=length_scale)

  observation_noise_variance = tf.nn.softplus(
      tf.Variable(0, dtype=dtype, name='observation_noise_variance'))

  # Create trainable inducing point locations and variational parameters.
  num_inducing_points_ = 20

  inducing_index_points = tf.Variable(
      initial_inducing_points_, dtype=dtype,
      name='inducing_index_points')
  variational_inducing_observations_loc = tf.Variable(
      np.zeros([num_inducing_points_], dtype=dtype),
      name='variational_inducing_observations_loc')
  variational_inducing_observations_scale = tf.Variable(
      np.eye(num_inducing_points_, dtype=dtype),
      name='variational_inducing_observations_scale')

  # These are the index point locations over which we'll construct the
  # (approximate) posterior predictive distribution.
  num_predictive_index_points_ = 500
  index_points_ = np.linspace(-13, 13,
                              num_predictive_index_points_,
                              dtype=dtype)[..., np.newaxis]


  # Construct our variational GP Distribution instance.
  vgp = tfd.VariationalGaussianProcess(
      kernel,
      index_points=index_points_,
      inducing_index_points=inducing_index_points,
      variational_inducing_observations_loc=variational_inducing_observations_loc,
      variational_inducing_observations_scale=variational_inducing_observations_scale,
      observation_noise_variance=observation_noise_variance)

  # For training, we use some simplistic numpy-based minibatching.
  batch_size = 64
  x_train_batch = tf.placeholder(dtype, [batch_size, 1], name='x_train_batch')
  y_train_batch = tf.placeholder(dtype, [batch_size], name='y_train_batch')

  # Create the loss function we want to optimize.
  loss = vgp.variational_loss(
      observations=y_train_batch,
      observation_index_points=x_train_batch,
      kl_weight=float(batch_size) / float(num_training_points_))

  optimizer = tf.train.AdamOptimizer()
  train_op = optimizer.minimize(loss)

  num_iters = 10000
  num_logs = 10
  with tf.Session() as sess:
    for i in range(num_iters):
      batch_idxs = np.random.randint(num_training_points_, size=[batch_size])
      x_train_batch_ = x_train_[batch_idxs, ...]
      y_train_batch_ = y_train_[batch_idxs]

      [_, loss_] = sess.run([train_op, loss],
                            feed_dict={x_train_batch: x_train_batch_,
                                       y_train_batch: y_train_batch_})
      if i % (num_iters / num_logs) == 0 or i + 1 == num_iters:
        print(i, loss_)

  # Generate a plot with
  #   - the posterior predictive mean
  #   - training data
  #   - inducing index points (plotted vertically at the mean of the variational
  #     posterior over inducing point function values)
  #   - 50 posterior predictive samples

  num_samples = 50
  [
      samples_,
      mean_,
      inducing_index_points_,
      variational_loc_,
  ] = sess.run([
      vgp.sample(num_samples),
      vgp.mean(),
      inducing_index_points,
      variational_inducing_observations_loc
  ])
  plt.figure(figsize=(15, 5))
  plt.scatter(inducing_index_points_[..., 0], variational_loc_
              marker='x', s=50, color='k', zorder=10)
  plt.scatter(x_train_[..., 0], y_train_, color='#00ff00', zorder=9)
  plt.plot(np.tile(index_points_[..., 0], num_samples),
           samples_.T, color='r', alpha=.1)
  plt.plot(index_points_, mean_, color='k')
  plt.plot(index_points_, f(index_points_), color='b')
  ```

  # Here we use the same data setup, but compute the optimal variational
  # parameters instead of training them.
  ```python
  # We'll use double precision throughout for better numerics.
  dtype = np.float64

  # Generate noisy data from a known function.
  f = lambda x: np.exp(-x[..., 0]**2 / 20.) * np.sin(1. * x[..., 0])
  true_observation_noise_variance_ = dtype(1e-1) ** 2

  num_training_points_ = 1000
  x_train_ = np.random.uniform(-10., 10., [num_training_points_, 1])
  y_train_ = (f(x_train_) +
              np.random.normal(
                  0., np.sqrt(true_observation_noise_variance_),
                  [num_training_points_]))

  # Create kernel with trainable parameters, and trainable observation noise
  # variance variable. Each of these is constrained to be positive.
  amplitude = (tf.nn.softplus(
    tf.Variable(.54, dtype=dtype, name='amplitude', use_resource=True)))
  length_scale = (
    1e-5 +
    tf.nn.softplus(
      tf.Variable(.54, dtype=dtype, name='length_scale', use_resource=True)))
  kernel = tfk.ExponentiatedQuadratic(
      amplitude=amplitude,
      length_scale=length_scale)

  observation_noise_variance = tf.nn.softplus(
      tf.Variable(
        .54, dtype=dtype, name='observation_noise_variance', use_resource=True))

  # Create trainable inducing point locations and variational parameters.
  num_inducing_points_ = 10

  inducing_index_points = tf.Variable(
      np.linspace(-10., 10., num_inducing_points_)[..., np.newaxis],
      dtype=dtype, name='inducing_index_points', use_resource=True)

  variational_loc, variational_scale = (
      tfd.VariationalGaussianProcess.optimal_variational_posterior(
          kernel=kernel,
          inducing_index_points=inducing_index_points,
          observation_index_points=x_train_,
          observations=y_train_,
          observation_noise_variance=observation_noise_variance))

  # These are the index point locations over which we'll construct the
  # (approximate) posterior predictive distribution.
  num_predictive_index_points_ = 500
  index_points_ = np.linspace(-13, 13,
                              num_predictive_index_points_,
                              dtype=dtype)[..., np.newaxis]

  # Construct our variational GP Distribution instance.
  vgp = tfd.VariationalGaussianProcess(
      kernel,
      index_points=index_points_,
      inducing_index_points=inducing_index_points,
      variational_inducing_observations_loc=variational_loc,
      variational_inducing_observations_scale=variational_scale,
      observation_noise_variance=observation_noise_variance)

  # For training, we use some simplistic numpy-based minibatching.
  batch_size = 64
  x_train_batch = tf.placeholder(dtype, [batch_size, 1], name='x_train_batch')
  y_train_batch = tf.placeholder(dtype, [batch_size], name='y_train_batch')

  # Create the loss function we want to optimize.
  loss = vgp.variational_loss(
      observations=y_train_batch,
      observation_index_points=x_train_batch,
      kl_weight=float(batch_size) / float(num_training_points_))

  optimizer = tf.train.AdamOptimizer(learning_rate=.01)
  train_op = optimizer.minimize(loss)

  num_iters = 300
  num_logs = 10
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_iters):
      batch_idxs = np.random.randint(num_training_points_, size=[batch_size])
      x_train_batch_ = x_train_[batch_idxs, ...]
      y_train_batch_ = y_train_[batch_idxs]

      [_, loss_] = sess.run([train_op, loss],
                            feed_dict={x_train_batch: x_train_batch_,
                                       y_train_batch: y_train_batch_})
      if i % (num_iters / num_logs) == 0 or i + 1 == num_iters:
        print(i, loss_)

    # Generate a plot with
    #   - the posterior predictive mean
    #   - training data
    #   - inducing index points (plotted vertically at the mean of the
    #     variational posterior over inducing point function values)
    #   - 50 posterior predictive samples

    num_samples = 50
    [
        samples_,
        mean_,
        inducing_index_points_,
        variational_loc_,
    ] = sess.run([
        vgp.sample(num_samples),
        vgp.mean(),
        inducing_index_points,
        variational_loc
    ])
    plt.figure(figsize=(15, 5))
    plt.scatter(inducing_index_points_[..., 0], variational_loc_,
                marker='x', s=50, color='k', zorder=10)
    plt.scatter(x_train_[..., 0], y_train_, color='#00ff00', alpha=.1, zorder=9)
    plt.plot(np.tile(index_points_, num_samples),
             samples_.T, color='r', alpha=.1)
    plt.plot(index_points_, mean_, color='k')
    plt.plot(index_points_, f(index_points_), color='b')

  ```

  #### References

  [1]: Titsias, M. "Variational Model Selection for Sparse Gaussian Process
       Regression", 2009.
       http://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf
  [2]: Hensman, J., Lawrence, N. "Gaussian Processes for Big Data", 2013
       https://arxiv.org/abs/1309.6835
  [3]: Carl Rasmussen, Chris Williams. Gaussian Processes For Machine Learning,
       2006. http://www.gaussianprocess.org/gpml/
  """

  def __init__(self,
               kernel,
               index_points,
               inducing_index_points,
               variational_inducing_observations_loc,
               variational_inducing_observations_scale,
               mean_fn=None,
               observation_noise_variance=0.,
               predictive_noise_variance=0.,
               jitter=1e-6,
               validate_args=False,
               allow_nan_stats=False,
               name='VariataionalGaussianProcess'):
    """Instantiate a VariationalGaussianProcess Distribution.

    Args:
      kernel: `PositiveSemidefiniteKernel`-like instance representing the
        GP's covariance function.
      index_points: `float` `Tensor` representing finite (batch of) vector(s) of
        points in the index set over which the VGP is defined. Shape has the
        form `[b1, ..., bB, e1, f1, ..., fF]` where `F` is the number of feature
        dimensions and must equal `kernel.feature_ndims` and `e1` is the number
        (size) of index points in each batch (we denote it `e1` to distinguish
        it from the numer of inducing index points, denoted `e2` below).
        Ultimately the VariationalGaussianProcess distribution corresponds to an
        `e1`-dimensional multivariate normal. The batch shape must be
        broadcastable with `kernel.batch_shape`, the batch shape of
        `inducing_index_points`, and any batch dims yielded by `mean_fn`.
      inducing_index_points: `float` `Tensor` of locations of inducing points in
        the index set. Shape has the form `[b1, ..., bB, e2, f1, ..., fF]`, just
        like `index_points`. The batch shape components needn't be identical to
        those of `index_points`, but must be broadcast compatible with them.
      variational_inducing_observations_loc: `float` `Tensor`; the mean of the
        (full-rank Gaussian) variational posterior over function values at the
        inducing points, conditional on observed data. Shape has the form `[b1,
        ..., bB, e2]`, where `b1, ..., bB` is broadcast compatible with other
        parameters' batch shapes, and `e2` is the number of inducing points.
      variational_inducing_observations_scale: `float` `Tensor`; the scale
        matrix of the (full-rank Gaussian) variational posterior over function
        values at the inducing points, conditional on observed data. Shape has
        the form `[b1, ..., bB, e2, e2]`, where `b1, ..., bB` is broadcast
        compatible with other parameters and `e2` is the number of inducing
        points.
      mean_fn: Python `callable` that acts on index points to produce a (batch
        of) vector(s) of mean values at those index points. Takes a `Tensor` of
        shape `[b1, ..., bB, f1, ..., fF]` and returns a `Tensor` whose shape is
        (broadcastable with) `[b1, ..., bB]`. Default value: `None` implies
        constant zero function.
      observation_noise_variance: `float` `Tensor` representing the variance
        of the noise in the Normal likelihood distribution of the model. May be
        batched, in which case the batch shape must be broadcastable with the
        shapes of all other batched parameters (`kernel.batch_shape`,
        `index_points`, etc.).
        Default value: `0.`
      predictive_noise_variance: `float` `Tensor` representing additional
        variance in the posterior predictive model. If `None`, we simply re-use
        `observation_noise_variance` for the posterior predictive noise. If set
        explicitly, however, we use the given value. This allows us, for
        example, to omit predictive noise variance (by setting this to zero) to
        obtain noiseless posterior predictions of function values, conditioned
        on noisy observations.
      jitter: `float` scalar `Tensor` added to the diagonal of the covariance
        matrix to ensure positive definiteness of the covariance matrix.
        Default value: `1e-6`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        Default value: `False`.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
        Default value: `False`.
      name: Python `str` name prefixed to Ops created by this class.
        Default value: "VariationalGaussianProcess".

    Raises:
      ValueError: if `mean_fn` is not `None` and is not callable.
    """
    parameters = dict(locals())
    with tf.compat.v1.name_scope(
        name, 'VariationalGaussianProcess', values=[
            index_points,
            inducing_index_points,
            variational_inducing_observations_loc,
            variational_inducing_observations_scale,
            observation_noise_variance,
            jitter]) as name:
      dtype = dtype_util.common_dtype(
          [kernel,
           index_points,
           inducing_index_points,
           variational_inducing_observations_loc,
           variational_inducing_observations_scale,
           observation_noise_variance,
           predictive_noise_variance,
           jitter], tf.float32)

      index_points = tf.convert_to_tensor(
          value=index_points, dtype=dtype, name='index_points')
      inducing_index_points = tf.convert_to_tensor(
          value=inducing_index_points, dtype=dtype,
          name='inducing_index_points')
      variational_inducing_observations_loc = tf.convert_to_tensor(
          value=variational_inducing_observations_loc, dtype=dtype,
          name='variational_inducing_observations_loc')
      variational_inducing_observations_scale = tf.convert_to_tensor(
          value=variational_inducing_observations_scale, dtype=dtype,
          name='variational_inducing_observations_scale')
      observation_noise_variance = tf.convert_to_tensor(
          value=observation_noise_variance,
          dtype=dtype,
          name='observation_noise_variance')
      if predictive_noise_variance is None:
        predictive_noise_variance = observation_noise_variance
      else:
        predictive_noise_variance = tf.convert_to_tensor(
            value=predictive_noise_variance, dtype=dtype,
            name='predictive_noise_variance')
      jitter = tf.convert_to_tensor(
          value=jitter, dtype=dtype, name='jitter')

      self._kernel = kernel
      self._index_points = index_points
      self._inducing_index_points = inducing_index_points
      self._variational_inducing_observations_posterior = (
          mvn_linear_operator.MultivariateNormalLinearOperator(
              loc=variational_inducing_observations_loc,
              scale=tf.linalg.LinearOperatorFullMatrix(
                  variational_inducing_observations_scale),
              name='variational_inducing_observations_posterior'))

      # Default to a constant zero function, borrowing the dtype from
      # index_points to ensure consistency.
      if mean_fn is None:
        mean_fn = lambda x: tf.zeros([1], dtype=dtype)
      else:
        if not callable(mean_fn):
          raise ValueError('`mean_fn` must be a Python callable')
      self._mean_fn = mean_fn
      self._observation_noise_variance = observation_noise_variance
      self._predictive_noise_variance = predictive_noise_variance
      self._jitter = jitter

      with tf.compat.v1.name_scope(
          'init', values=[index_points,
                          inducing_index_points,
                          variational_inducing_observations_loc,
                          variational_inducing_observations_scale,
                          observation_noise_variance,
                          jitter]):
        # We let t and z denote predictive and inducing index points, resp.
        kzz = _add_diagonal_shift(
            kernel.matrix(inducing_index_points, inducing_index_points),
            jitter)

        self._chol_kzz = tf.linalg.cholesky(kzz)
        self._kzz_inv_varloc = _solve_cholesky_factored_system_vec(
            self._chol_kzz,
            (variational_inducing_observations_loc -
             mean_fn(inducing_index_points)),
            name='kzz_inv_varloc')

        loc, scale = self._compute_posterior_predictive_params()

        super(VariationalGaussianProcess, self).__init__(
            loc=loc,
            scale=scale,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name)
        self._parameters = parameters
        self._graph_parents = [
            index_points,
            inducing_index_points,
            variational_inducing_observations_loc,
            variational_inducing_observations_scale,
            observation_noise_variance,
            jitter]

  def _compute_posterior_predictive_params(self):
    ktt = _add_diagonal_shift(
        self._kernel.matrix(self._index_points, self._index_points),
        self._jitter)
    kzt = tf.linalg.LinearOperatorFullMatrix(
        self._kernel.matrix(self._inducing_index_points, self._index_points))

    kzz_inv_kzt = tf.linalg.LinearOperatorFullMatrix(
        _solve_cholesky_factored_system(
            self._chol_kzz, kzt.to_dense(), name='kzz_inv_kzt'))

    var_cov = tf.linalg.LinearOperatorFullMatrix(
        self._variational_inducing_observations_posterior.covariance())
    posterior_predictive_cov = (
        ktt -
        kzt.matmul(kzz_inv_kzt.to_dense(), adjoint=True) +
        kzz_inv_kzt.matmul(var_cov.matmul(kzz_inv_kzt.to_dense()),
                           adjoint=True))

    # Add predictive_noise_variance
    posterior_predictive_cov = _add_diagonal_shift(
        posterior_predictive_cov, self._predictive_noise_variance)

    scale = tf.linalg.LinearOperatorLowerTriangular(
        tf.linalg.cholesky(posterior_predictive_cov))

    loc = (self._mean_fn(self._index_points) +
           kzt.matvec(self._kzz_inv_varloc, adjoint=True))

    return loc, scale

  @property
  def mean_fn(self):
    return self._mean_fn

  @property
  def kernel(self):
    return self._kernel

  @property
  def index_points(self):
    return self._index_points

  @property
  def inducing_index_points(self):
    return self._inducing_index_points

  @property
  def variational_inducing_observations_loc(self):
    return self._variational_inducing_observations_posterior.loc

  @property
  def variational_inducing_observations_scale(self):
    return self._variational_inducing_observations_posterior.scale

  @property
  def observation_noise_variance(self):
    return self._observation_noise_variance

  @property
  def predictive_noise_variance(self):
    return self._predictive_noise_variance

  @property
  def jitter(self):
    return self._jitter

  def _covariance(self):
    return self._covariance_matrix

  def variational_loss(self,
                       observations,
                       observation_index_points=None,
                       kl_weight=1.,
                       name='variational_loss'):
    """Variational loss for the VGP.

    Given `observations` and `observation_index_points`, compute the
    negative variational lower bound as specified in [Hensman, 2013][1].

    Args:
      observations: `float` `Tensor` representing collection, or batch of
        collections, of observations corresponding to
        `observation_index_points`. Shape has the form `[b1, ..., bB, e]`, which
        must be brodcastable with the batch and example shapes of
        `observation_index_points`. The batch shape `[b1, ..., bB]` must be
        broadcastable with the shapes of all other batched parameters
        (`kernel.batch_shape`, `observation_index_points`, etc.).
      observation_index_points: `float` `Tensor` representing finite (batch of)
        vector(s) of points where observations are defined. Shape has the
        form `[b1, ..., bB, e1, f1, ..., fF]` where `F` is the number of feature
        dimensions and must equal `kernel.feature_ndims` and `e1` is the number
        (size) of index points in each batch (we denote it `e1` to distinguish
        it from the numer of inducing index points, denoted `e2` below). If
        set to `None` uses `index_points` as the origin for observations.
        Default value: None.
      kl_weight: Amount by which to scale the KL divergence loss between prior
        and posterior.
        Default value: 1.
      name: Python `str` name prefixed to Ops created by this class.
        Default value: "GaussianProcess".
    Returns:
      loss: Scalar tensor representing the negative variational lower bound.
        Can be directly used in a `tf.Optimizer`.
    Raises:
      ValueError: if `mean_fn` is not `None` and is not callable.

    #### References

    [1]: Hensman, J., Lawrence, N. "Gaussian Processes for Big Data", 2013
         https://arxiv.org/abs/1309.6835
    """

    with tf.compat.v1.name_scope(
        name, 'variational_gp_loss', values=[
            observations,
            observation_index_points,
            kl_weight]):
      if observation_index_points is None:
        observation_index_points = self._index_points
      observation_index_points = tf.convert_to_tensor(
          value=observation_index_points, dtype=self._dtype,
          name='observation_index_points')
      observations = tf.convert_to_tensor(
          value=observations, dtype=self._dtype, name='observations')
      kl_weight = tf.convert_to_tensor(
          value=kl_weight, dtype=self._dtype,
          name='kl_weight')

      # The variational loss is a negative ELBO. The ELBO can be broken down
      # into three terms:
      #  1. a likelihood term
      #  2. a trace term arising from the covariance of the posterior predictive

      kzx = self.kernel.matrix(self._inducing_index_points,
                               observation_index_points)

      kzx_linop = tf.linalg.LinearOperatorFullMatrix(kzx)
      loc = (self._mean_fn(observation_index_points) +
             kzx_linop.matvec(self._kzz_inv_varloc, adjoint=True))

      likelihood = independent.Independent(
          normal.Normal(
              loc=loc,
              scale=tf.sqrt(self._observation_noise_variance + self._jitter),
              name='NormalLikelihood'),
          reinterpreted_batch_ndims=1)
      obs_ll = likelihood.log_prob(observations)

      chol_kzz_linop = tf.linalg.LinearOperatorLowerTriangular(self._chol_kzz)
      chol_kzz_inv_kzx = chol_kzz_linop.solve(kzx)
      kzz_inv_kzx = chol_kzz_linop.solve(chol_kzz_inv_kzx, adjoint=True)

      kxx_diag = tf.linalg.diag_part(
          self.kernel.matrix(
              observation_index_points, observation_index_points))
      ktilde_trace_term = (
          tf.reduce_sum(input_tensor=kxx_diag, axis=-1) -
          tf.reduce_sum(input_tensor=chol_kzz_inv_kzx ** 2, axis=[-2, -1]))

      # Tr(SB)
      # where S = A A.T, A = variational_inducing_observations_scale
      # and B = Kzz^-1 Kzx Kzx.T Kzz^-1
      #
      # Now Tr(SB) = Tr(A A.T Kzz^-1 Kzx Kzx.T Kzz^-1)
      #            = Tr(A.T Kzz^-1 Kzx Kzx.T Kzz^-1 A)
      #            = sum_ij (A.T Kzz^-1 Kzx)_{ij}^2
      other_trace_term = tf.reduce_sum(
          input_tensor=(
              self._variational_inducing_observations_posterior.scale.matmul(
                  kzz_inv_kzx) ** 2),
          axis=[-2, -1])

      trace_term = (.5 * (ktilde_trace_term + other_trace_term) /
                    self._observation_noise_variance)

      inducing_prior = gaussian_process.GaussianProcess(
          kernel=self._kernel,
          mean_fn=self._mean_fn,
          index_points=self._inducing_index_points,
          observation_noise_variance=self._observation_noise_variance)

      kl_term = kl_weight * kullback_leibler.kl_divergence(
          self._variational_inducing_observations_posterior,
          inducing_prior)

      lower_bound = (obs_ll - trace_term - kl_term)

      return -tf.reduce_mean(input_tensor=lower_bound)

  @staticmethod
  def optimal_variational_posterior(
      kernel,
      inducing_index_points,
      observation_index_points,
      observations,
      observation_noise_variance,
      mean_fn=None,
      jitter=1e-6,
      name=None):
    """Model selection for optimal variational hyperparameters.

    Given the full training set (parameterized by `observations` and
    `observation_index_points`), compute the optimal variational
    location and scale for the VGP. This is based of the method suggested
    in [Titsias, 2009][1].

    Args:
      kernel: `PositiveSemidefiniteKernel`-like instance representing the
        GP's covariance function.
      inducing_index_points: `float` `Tensor` of locations of inducing points in
        the index set. Shape has the form `[b1, ..., bB, e2, f1, ..., fF]`, just
        like `observation_index_points`. The batch shape components needn't be
        identical to those of `observation_index_points`, but must be broadcast
        compatible with them.
      observation_index_points: `float` `Tensor` representing finite (batch of)
        vector(s) of points where observations are defined. Shape has the
        form `[b1, ..., bB, e1, f1, ..., fF]` where `F` is the number of feature
        dimensions and must equal `kernel.feature_ndims` and `e1` is the number
        (size) of index points in each batch (we denote it `e1` to distinguish
        it from the numer of inducing index points, denoted `e2` below).
      observations: `float` `Tensor` representing collection, or batch of
        collections, of observations corresponding to
        `observation_index_points`. Shape has the form `[b1, ..., bB, e]`, which
        must be brodcastable with the batch and example shapes of
        `observation_index_points`. The batch shape `[b1, ..., bB]` must be
        broadcastable with the shapes of all other batched parameters
        (`kernel.batch_shape`, `observation_index_points`, etc.).
      observation_noise_variance: `float` `Tensor` representing the variance
        of the noise in the Normal likelihood distribution of the model. May be
        batched, in which case the batch shape must be broadcastable with the
        shapes of all other batched parameters (`kernel.batch_shape`,
        `index_points`, etc.).
        Default value: `0.`
      mean_fn: Python `callable` that acts on index points to produce a (batch
        of) vector(s) of mean values at those index points. Takes a `Tensor` of
        shape `[b1, ..., bB, f1, ..., fF]` and returns a `Tensor` whose shape is
        (broadcastable with) `[b1, ..., bB]`. Default value: `None` implies
        constant zero function.
      jitter: `float` scalar `Tensor` added to the diagonal of the covariance
        matrix to ensure positive definiteness of the covariance matrix.
        Default value: `1e-6`.
      name: Python `str` name prefixed to Ops created by this class.
        Default value: "optimal_variational_posterior".
    Returns:
      loc, scale: Tuple representing the variational location and scale.
    Raises:
      ValueError: if `mean_fn` is not `None` and is not callable.

    #### References

    [1]: Titsias, M. "Variational Model Selection for Sparse Gaussian Process
         Regression", 2009.
         http://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf
    """

    with tf.compat.v1.name_scope(
        name, 'optimal_variational_posterior',
        values=[inducing_index_points,
                observation_index_points,
                observations,
                observation_noise_variance]):
      dtype = dtype_util.common_dtype(
          [inducing_index_points,
           observation_index_points,
           observations,
           observation_noise_variance,
           jitter], tf.float32)

      inducing_index_points = tf.convert_to_tensor(
          value=inducing_index_points,
          dtype=dtype, name='inducing_index_points')
      observation_index_points = tf.convert_to_tensor(
          value=observation_index_points, dtype=dtype,
          name='observation_index_points')
      observations = tf.convert_to_tensor(
          value=observations, dtype=dtype, name='observations')
      observation_noise_variance = tf.convert_to_tensor(
          value=observation_noise_variance,
          dtype=dtype,
          name='observation_noise_variance')
      jitter = tf.convert_to_tensor(
          value=jitter, dtype=dtype, name='jitter')

      # Default to a constant zero function.
      if mean_fn is None:
        mean_fn = lambda x: tf.zeros([1], dtype=dtype)
      else:
        if not callable(mean_fn):
          raise ValueError('`mean_fn` must be a Python callable')

      # z are the inducing points and x are the observation index points.
      kzz = kernel.matrix(inducing_index_points, inducing_index_points)
      kzx = kernel.matrix(inducing_index_points, observation_index_points)

      noise_var_inv = tf.math.reciprocal(observation_noise_variance)

      sigma_inv = _add_diagonal_shift(
          kzz + noise_var_inv * tf.matmul(kzx, kzx, adjoint_b=True),
          jitter)

      chol_sigma_inv = tf.linalg.cholesky(sigma_inv)

      kzx_lin_op = tf.linalg.LinearOperatorFullMatrix(kzx)
      kzx_obs = kzx_lin_op.matvec(
          observations - mean_fn(observation_index_points))
      kzz_lin_op = tf.linalg.LinearOperatorFullMatrix(kzz)
      loc = (mean_fn(inducing_index_points) +
             noise_var_inv * kzz_lin_op.matvec(
                 _solve_cholesky_factored_system_vec(chol_sigma_inv, kzx_obs)))

      chol_sigma_inv_lin_op = tf.linalg.LinearOperatorLowerTriangular(
          chol_sigma_inv)
      scale = chol_sigma_inv_lin_op.solve(kzz)

      return loc, scale
