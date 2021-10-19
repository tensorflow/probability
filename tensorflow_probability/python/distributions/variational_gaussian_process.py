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

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import util as tfp_util
from tensorflow_probability.python.bijectors import fill_scale_tril as fill_scale_tril_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import gaussian_process
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import mvn_linear_operator
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math.psd_kernels import positive_semidefinite_kernel as psd_kernel
from tensorflow_probability.python.math.psd_kernels.internal import util as kernel_util

__all__ = [
    'VariationalGaussianProcess',
]


class _VariationalKernel(psd_kernel.AutoCompositeTensorPsdKernel):
  """A PSDKernel which computes the variational kernel from [Titsias, 2009].

  The VariationalGaussianProcess can be cast as a special case of
  GaussianProcess with a particular class of kernel function. The idea is that
  instead of conditioning exactly on data, which would cost us time cubic in the
  number of data points, we learn a variational posterior over function values
  at a set of so-called "inducing points" and then make predictions by
  marginalizing over the variational posterior over the function values at the
  inducing point locations. This yields a new model, which is again just a GP
  but with a more complicated kernel function than we started with (similar to
  the way in which an exact GP posterior is just another GP with a more
  complicated kernel function).

  Having established that context, we can describe the `_VariationalKernel`. It
  is parameterized by
    - a `base_kernel` (another PSDKernel instance), the kernel we use to model
      the covariance structure of our true GP model,
    - `inducing_index_points`, the locations of the inducing points in the index
      set,
    - `variational_scale`, the scale matrix of the (multivariate Normal)
      variational posterior in the approximate model, that is, the scale matrix
      of the variational distribution `q(f(z)) ~= p(f(z) | f(z))`, where `z` are
      the inducing point locations, `x` the observation locations, and `f(.)`
      the function values at those locations.

  More precisely, it computes its value on inputs `x`, `y` as follows where, for
  brevity,

    - `z` are the aforementioned `inducing_index_points`
    - `var_scale` is the variational scale matrix
    - `k0` is the function represented by `base_kernel`
    - `k` is an the function represented by this class

  ```none
    kxy = k0(x, y)
    kxz = k0(x, z)
    kzy = k0(z, y)
    kzz = k0(z, z)
    k(x, y) = (kxy -
               kxz @ inv(kzz) @ kzy +
               kxz @ inv(kzz) @ var_scale @ var_scale^T @ inv(kzz) @ kzy)
  ```

  """

  def __init__(self,
               base_kernel,
               inducing_index_points,
               variational_scale,
               jitter=1e-6,
               name='VariationalKernel'):
    """Construct a _VariationalKernel instance.

    Args:
      base_kernel: a `PositiveSemidefiniteKernel` instance, the kernel used to
        build the constituent kernel matrices this kernel's computation is based
        around.
      inducing_index_points: a `Tensor` of shape `[..., N, F]`, where `N` is the
        number of such inputs ("examples" in the PSDKernels parlance) and `F`
        represents the feature shape of `base_kernel`. This set of index points,
        similar to the `inducing_index_points` argument to SchurComplement, will
        typically be the inducing points of the variational GP. Batch dimensions
        must be broadcast-compatible with the batch shape of `base_kernel` and
        `variational_scale`.
      variational_scale: a `Tensor` of shape `[..., N, N]` where `N` is the
        number of examples in `inducing_index_points`. Batch dimensions must be
        broadcast-compatible with the batch shape of `base_kernel` and
        `inducing_index_points`.
      jitter: `float` scalar `Tensor` added to the diagonal of the covariance
        matrix to ensure positive definiteness of the covariance matrix.
        Default value: `1e-6`.
      name: Python `str` name prefixed to `Op`A created by this class.
        Default value: `"VariationalKernel"`
    """
    parameters = dict(locals())
    with tf.name_scope(name):
      dtype = dtype_util.common_dtype([
          inducing_index_points, variational_scale, jitter
      ], dtype_hint=tf.float32)

      self._base_kernel = base_kernel
      self._inducing_index_points = tensor_util.convert_nonref_to_tensor(
          inducing_index_points, dtype=dtype, name='inducing_index_points')
      self._variational_scale = tensor_util.convert_nonref_to_tensor(
          variational_scale, dtype=dtype, name='variational_scale')
      self._jitter = tensor_util.convert_nonref_to_tensor(
          jitter, dtype=dtype, name='jitter')

      def _compute_chol_kzz(z):
        kzz = base_kernel.matrix(z, z)
        result = tf.linalg.cholesky(_add_diagonal_shift(kzz, self._jitter))
        return result

      # Somewhat confusingly, but for the sake of brevity, we use `var` to refer
      # to the *variance* of the variational distribution, which is
      # `variational_scale @ variational_scale^T`.
      def _compute_kzzinv_var_kzzinv(variational_scale):
        mat_sq_fn = lambda x: tf.linalg.matmul(x, x, adjoint_b=True)
        return mat_sq_fn(
            _solve_cholesky_factored_system(self._chol_kzz, variational_scale))

      self._chol_kzz = tfp_util.DeferredTensor(
          inducing_index_points,
          transform_fn=_compute_chol_kzz,
          shape=None,
          name='chol_kzz')

      self._kzzinv_var_kzzinv = tfp_util.DeferredTensor(
          variational_scale,
          transform_fn=_compute_kzzinv_var_kzzinv,
          shape=None,
          name='kzzinv_var_kzzinv')

      super(_VariationalKernel, self).__init__(
          feature_ndims=base_kernel.feature_ndims,
          dtype=dtype,
          name=name,
          parameters=parameters)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        base_kernel=parameter_properties.BatchedComponentProperties(),
        inducing_index_points=parameter_properties.ParameterProperties(
            event_ndims=lambda self: self.base_kernel.feature_ndims + 1,
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED),
        variational_scale=(
            parameter_properties.ParameterProperties(
                event_ndims=2,
                shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED,
                default_constraining_bijector_fn=(
                    fill_scale_tril_bijector.FillScaleTriL))))

  @property
  def base_kernel(self):
    return self._base_kernel

  @property
  def inducing_index_points(self):
    return self._inducing_index_points

  @property
  def variational_scale(self):
    return self._variational_scale

  def chol_kzz(self):
    return tf.convert_to_tensor(self._chol_kzz)

  def _apply(self, x1, x2, example_ndims=1):
    # We follow nearly-identical patterns here to the SchurComplement kernel.

    # Shape: bc(Bk, B1, B2) + bc(E1, E2)
    k12 = self._base_kernel.apply(x1, x2, example_ndims)

    # Shape: bc(Bk, B1, Bz) + E1 + [ez]
    k1z = self._base_kernel.tensor(x1, self._inducing_index_points,
                                   x1_example_ndims=example_ndims,
                                   x2_example_ndims=1)

    # Shape: bc(Bk, B2, Bz) + E2 + [ez]
    k2z = self._base_kernel.tensor(x2, self._inducing_index_points,
                                   x1_example_ndims=example_ndims,
                                   x2_example_ndims=1)

    chol_kzz = kernel_util.pad_shape_with_ones(
        self._chol_kzz, example_ndims - 1, -3)
    kzzchol_linop = tf.linalg.LinearOperatorLowerTriangular(chol_kzz)

    # Shape: bc(Bz, Bk, B2) + E2 + [ez]
    kzzinv_kz2 = tf.linalg.matrix_transpose(
        # Shape: bc(Bz, Bk, B2) + E2[:-1] + [ez] + E2[-1]
        kzzchol_linop.solve(
            # Shape: bc(Bz, Bk, B2) + E2[:-1] + [ez] + E2[-1]
            kzzchol_linop.solve(k2z, adjoint_arg=True),
            adjoint=True))

    # Shape: bc(Bz, Bk, B1, B2) + bc(E1, E2)
    k1z_kzzinv_kz2 = tf.reduce_sum(
        # Shape: bc(Bz, Bk, B1, B2) + bc(E1, E2) + [ez]
        input_tensor=k1z * kzzinv_kz2,
        axis=-1)

    # Do this c2t only once
    kzzinv_var_kzzinv = tf.convert_to_tensor(self._kzzinv_var_kzzinv)

    # Shape: bc(Bz, Bk, Bv) + [1, ..., 1] + [ez, ez]
    kzzinv_var_kzzinv = kernel_util.pad_shape_with_ones(
        kzzinv_var_kzzinv, example_ndims - 1, -3)

    # Shape: bc(Bz, Bk, Bv) + E2 + [ez]
    kzzinv_var_kzzinv_kz2 = tf.linalg.matrix_transpose(
        # Shape: bc(Bz, Bk, B2) + E2[:-1] + [ez] + E2[-1]
        tf.linalg.matmul(kzzinv_var_kzzinv, k2z, adjoint_b=True))

    # Shape: bc(Bz, Bk, Bv) + bc(E1, E2)
    k1z_kzzinv_var_kzzinv_kz2 = tf.reduce_sum(
        # Shape: bc(Bz, Bk, Bv) + bc(E1, E2) + [ez]
        input_tensor=k1z * kzzinv_var_kzzinv_kz2,
        axis=-1)

    # K12 - K1z @ inv(Kzz) @ Kz2 + K1z @ inv(Kzz) @ S @ inv(Kzz) @ Kz2
    result = (
        # Shape: bc(Bk, B1, B2) + bc(E1, E2)
        k12 -
        # Shape: bc(Bk, B1, B2) + bc(E1, E2)
        k1z_kzzinv_kz2 +
        # Shape: bc(Bk, B1, B2, Bv) + bc(E1, E2)
        k1z_kzzinv_var_kzzinv_kz2)

    return result


def _make_posterior_predictive_mean_fn(
    kernel,
    mean_fn,
    inducing_index_points,
    variational_inducing_observations_loc,
    chol_kzz_fn):
  """Define the VGP's variational posterior predictive mean function."""

  def _post_pred_mean_fn(index_points):
    """The variatioanl posterior predictive mean function."""
    kzt = tf.linalg.LinearOperatorFullMatrix(
        kernel.matrix(inducing_index_points, index_points))

    kzzinv_varloc = _solve_cholesky_factored_system_vec(
        chol_kzz_fn(),
        (variational_inducing_observations_loc -
         mean_fn(inducing_index_points)),
        name='kzzinv_varloc')

    return (mean_fn(index_points) +
            kzt.matvec(kzzinv_varloc, adjoint=True))

  return _post_pred_mean_fn


def _add_diagonal_shift(matrix, shift):
  return tf.linalg.set_diag(
      matrix, tf.linalg.diag_part(matrix) + shift, name='add_diagonal_shift')


def _solve_cholesky_factored_system(
    cholesky_factor, rhs, name=None):
  with tf.name_scope(
      name or '_solve_cholesky_factored_system') as scope:
    cholesky_factor = tf.convert_to_tensor(
        cholesky_factor, name='cholesky_factor')
    rhs = tf.convert_to_tensor(rhs, name='rhs')
    lin_op = tf.linalg.LinearOperatorLowerTriangular(
        cholesky_factor, name=scope)
    return lin_op.solve(lin_op.solve(rhs), adjoint=True)


def _solve_cholesky_factored_system_vec(cholesky_factor, rhs, name=None):
  with tf.name_scope(
      name or '_solve_cholesky_factored_system') as scope:
    cholesky_factor = tf.convert_to_tensor(
        cholesky_factor, name='cholesky_factor')
    rhs = tf.convert_to_tensor(rhs, name='rhs')
    lin_op = tf.linalg.LinearOperatorLowerTriangular(
        cholesky_factor, name=scope)
    return lin_op.solvevec(lin_op.solvevec(rhs), adjoint=True)


class VariationalGaussianProcess(gaussian_process.GaussianProcess):
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
       conditional on observations is replaced with an arbitrary (learnable)
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
  import matplotlib.pyplot as plt
  import numpy as np
  import tensorflow.compat.v2 as tf
  import tensorflow_probability as tfp

  tfb = tfp.bijectors
  tfd = tfp.distributions
  tfk = tfp.math.psd_kernels

  # We'll use double precision throughout for better numerics.
  dtype = np.float64

  # Generate noisy data from a known function.
  f = lambda x: np.exp(-x[..., 0]**2 / 20.) * np.sin(1. * x[..., 0])
  true_observation_noise_variance_ = dtype(1e-1) ** 2

  num_training_points_ = 100
  x_train_ = np.concatenate(
      [np.random.uniform(-6., 0., [num_training_points_ // 2 , 1]),
      np.random.uniform(1., 10., [num_training_points_ // 2 , 1])],
      axis=0).astype(dtype)
  y_train_ = (f(x_train_) +
              np.random.normal(
                  0., np.sqrt(true_observation_noise_variance_),
                  [num_training_points_]).astype(dtype))

  # Create kernel with trainable parameters, and trainable observation noise
  # variance variable. Each of these is constrained to be positive.
  amplitude = tfp.util.TransformedVariable(
      1., tfb.Softplus(), dtype=dtype, name='amplitude')
  length_scale = tfp.util.TransformedVariable(
      1., tfb.Softplus(), dtype=dtype, name='length_scale')
  kernel = tfk.ExponentiatedQuadratic(
      amplitude=amplitude,
      length_scale=length_scale)

  observation_noise_variance = tfp.util.TransformedVariable(
      1., tfb.Softplus(), dtype=dtype, name='observation_noise_variance')

  # Create trainable inducing point locations and variational parameters.
  num_inducing_points_ = 20
  inducing_index_points = tf.Variable(
      np.linspace(-5., 5., num_inducing_points_)[..., np.newaxis],
      dtype=dtype, name='inducing_index_points')
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
      variational_inducing_observations_loc=
          variational_inducing_observations_loc,
      variational_inducing_observations_scale=
          variational_inducing_observations_scale,
      observation_noise_variance=observation_noise_variance)

  # For training, we use some simplistic numpy-based minibatching.
  batch_size = 64

  optimizer = tf.optimizers.Adam(learning_rate=.1)

  @tf.function
  def optimize(x_train_batch, y_train_batch):
    with tf.GradientTape() as tape:
      # Create the loss function we want to optimize.
      loss = vgp.variational_loss(
          observations=y_train_batch,
          observation_index_points=x_train_batch,
          kl_weight=float(batch_size) / float(num_training_points_))
    grads = tape.gradient(loss, vgp.trainable_variables)
    optimizer.apply_gradients(zip(grads, vgp.trainable_variables))
    return loss

  num_iters = 10000
  num_logs = 10
  for i in range(num_iters):
    batch_idxs = np.random.randint(num_training_points_, size=[batch_size])
    x_train_batch = x_train_[batch_idxs, ...]
    y_train_batch = y_train_[batch_idxs]
    loss = optimize(x_train_batch, y_train_batch)

    if i % (num_iters / num_logs) == 0 or i + 1 == num_iters:
      print(i, loss.numpy())

  # Generate a plot with
  #   - the posterior predictive mean
  #   - training data
  #   - inducing index points (plotted vertically at the mean of the variational
  #     posterior over inducing point function values)
  #   - 50 posterior predictive samples

  num_samples = 50
  samples = vgp.sample(num_samples).numpy()
  mean = vgp.mean().numpy()
  inducing_index_points_ = inducing_index_points.numpy()
  variational_loc = variational_inducing_observations_loc.numpy()

  plt.figure(figsize=(15, 5))
  plt.scatter(inducing_index_points_[..., 0], variational_loc,
              marker='x', s=50, color='k', zorder=10)
  plt.scatter(x_train_[..., 0], y_train_, color='#00ff00', zorder=9)
  plt.plot(np.tile(index_points_, (num_samples)),
            samples.T, color='r', alpha=.1)
  plt.plot(index_points_, mean, color='k')
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
  amplitude = tfp.util.TransformedVariable(
      1., tfb.Softplus(), dtype=dtype, name='amplitude')
  length_scale = tfp.util.TransformedVariable(
      1., tfb.Softplus(), dtype=dtype, name='length_scale')
  kernel = tfk.ExponentiatedQuadratic(
      amplitude=amplitude,
      length_scale=length_scale)

  observation_noise_variance = tfp.util.TransformedVariable(
      1., tfb.Softplus(), dtype=dtype, name='observation_noise_variance')

  # Create trainable inducing point locations and variational parameters.
  num_inducing_points_ = 10

  inducing_index_points = tf.Variable(
      np.linspace(-10., 10., num_inducing_points_)[..., np.newaxis],
      dtype=dtype, name='inducing_index_points')

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

  optimizer = tf.optimizers.Adam(learning_rate=.05, beta_1=.5, beta_2=.99)

  @tf.function
  def optimize(x_train_batch, y_train_batch):
    with tf.GradientTape() as tape:
      # Create the loss function we want to optimize.
      loss = vgp.variational_loss(
          observations=y_train_batch,
          observation_index_points=x_train_batch,
          kl_weight=float(batch_size) / float(num_training_points_))
    grads = tape.gradient(loss, vgp.trainable_variables)
    optimizer.apply_gradients(zip(grads, vgp.trainable_variables))
    return loss

  num_iters = 300
  num_logs = 10
  for i in range(num_iters):
    batch_idxs = np.random.randint(num_training_points_, size=[batch_size])
    x_train_batch_ = x_train_[batch_idxs, ...]
    y_train_batch_ = y_train_[batch_idxs]

    loss = optimize(x_train_batch, y_train_batch)
    if i % (num_iters / num_logs) == 0 or i + 1 == num_iters:
      print(i, loss.numpy())

  # Generate a plot with
  #   - the posterior predictive mean
  #   - training data
  #   - inducing index points (plotted vertically at the mean of the
  #     variational posterior over inducing point function values)
  #   - 50 posterior predictive samples

  num_samples = 50

  samples_ = vgp.sample(num_samples).numpy()
  mean_ = vgp.mean().numpy()
  inducing_index_points_ = inducing_index_points.numpy()
  variational_loc_ = variational_loc.numpy()

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
               observation_noise_variance=None,
               predictive_noise_variance=None,
               jitter=1e-6,
               validate_args=False,
               allow_nan_stats=False,
               name='VariationalGaussianProcess'):
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
    with tf.name_scope(
        name or 'VariationalGaussianProcess') as name:
      dtype = dtype_util.common_dtype([
          kernel, index_points, inducing_index_points,
          variational_inducing_observations_loc,
          variational_inducing_observations_scale, observation_noise_variance,
          predictive_noise_variance, jitter
      ], tf.float32)

      index_points = tensor_util.convert_nonref_to_tensor(
          index_points, dtype=dtype, name='index_points')
      inducing_index_points = tensor_util.convert_nonref_to_tensor(
          inducing_index_points, dtype=dtype, name='inducing_index_points')
      variational_inducing_observations_loc = (
          tensor_util.convert_nonref_to_tensor(
              variational_inducing_observations_loc,
              dtype=dtype,
              name='variational_inducing_observations_loc'))
      variational_inducing_observations_scale = (
          tensor_util.convert_nonref_to_tensor(
              variational_inducing_observations_scale,
              dtype=dtype,
              name='variational_inducing_observations_scale'))
      observation_noise_variance = tensor_util.convert_nonref_to_tensor(
          observation_noise_variance,
          dtype=dtype,
          name='observation_noise_variance')

      predictive_noise_variance = tensor_util.convert_nonref_to_tensor(
          predictive_noise_variance,
          dtype=dtype,
          name='predictive_noise_variance')
      if predictive_noise_variance is None:
        predictive_noise_variance = observation_noise_variance
      jitter = tensor_util.convert_nonref_to_tensor(
          jitter, dtype=dtype, name='jitter')

      self._kernel = kernel
      self._index_points = index_points
      self._inducing_index_points = inducing_index_points
      self._variational_inducing_observations_loc = (
          variational_inducing_observations_loc)
      self._variational_inducing_observations_scale = (
          variational_inducing_observations_scale)
      self._variational_inducing_observations_posterior = (
          mvn_linear_operator.MultivariateNormalLinearOperator(
              loc=variational_inducing_observations_loc,
              scale=tf.linalg.LinearOperatorLowerTriangular(
                  variational_inducing_observations_scale),
              name='variational_inducing_observations_posterior'))

      # Default to a constant zero function, borrowing the dtype from
      # index_points to ensure consistency.
      if mean_fn is None:
        mean_fn = lambda x: tf.zeros([1], dtype=dtype)
      else:
        if not callable(mean_fn):
          raise ValueError('`mean_fn` must be a Python callable')
      self._prior_mean_fn = mean_fn

      # Set up the prior over function values at inducing points using the
      # original kernel and mean_fn (as opposed to the variational posterior
      # predictive kernel and mean fn we define below). This is a distribution
      # over latent function values so there is no observation noise term.
      self._inducing_prior = gaussian_process.GaussianProcess(
          kernel=kernel,
          mean_fn=mean_fn,
          index_points=inducing_index_points)

      self._predictive_noise_variance = predictive_noise_variance
      self._vgp_observation_noise_variance = observation_noise_variance

      variational_kernel = _VariationalKernel(
          kernel,
          inducing_index_points,
          variational_inducing_observations_scale,
          jitter=jitter)

      posterior_predictive_mean_fn = _make_posterior_predictive_mean_fn(
          kernel=kernel,
          mean_fn=mean_fn,
          inducing_index_points=inducing_index_points,
          variational_inducing_observations_loc=(
              variational_inducing_observations_loc),
          chol_kzz_fn=variational_kernel.chol_kzz)

      super(VariationalGaussianProcess, self).__init__(
          kernel=variational_kernel,
          mean_fn=posterior_predictive_mean_fn,
          index_points=index_points,
          jitter=jitter,
          # What the GP super class calls "observation noise variance" we call
          # here the "predictive noise variance". We use the observation noise
          # variance for the fit/solve process, and predictive for downstream
          # computations like sampling.
          observation_noise_variance=predictive_noise_variance,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=name)
      self._parameters = parameters

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        kernel=parameter_properties.BatchedComponentProperties(),
        index_points=parameter_properties.ParameterProperties(
            event_ndims=lambda self: self.kernel.feature_ndims + 1,
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED),
        inducing_index_points=parameter_properties.ParameterProperties(
            event_ndims=lambda self: self.kernel.feature_ndims + 1,
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED),
        variational_inducing_observations_loc=(
            parameter_properties.ParameterProperties(
                event_ndims=1,
                shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED)),
        variational_inducing_observations_scale=(
            parameter_properties.ParameterProperties(
                event_ndims=2,
                shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED,
                default_constraining_bijector_fn=(
                    fill_scale_tril_bijector.FillScaleTriL))),
        observation_noise_variance=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype))),
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED),
        predictive_noise_variance=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype))),
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED))

  @property
  def inducing_index_points(self):
    return self._inducing_index_points

  @property
  def variational_inducing_observations_loc(self):
    return self._variational_inducing_observations_loc

  @property
  def variational_inducing_observations_scale(self):
    return self._variational_inducing_observations_scale

  @property
  def predictive_noise_variance(self):
    return self._predictive_noise_variance

  def surrogate_posterior_expected_log_likelihood(
      self,
      observations,
      observation_index_points=None,
      log_likelihood_fn=None,
      quadrature_size=10,
      name=None):
    """Compute the expected log likelihood term in the ELBO, using quadrature.

    In variational inference, we're interested in optimizing the ELBO, which
    looks like

    ```none
      ELBO = -E_{q(z)} log p(x | z) + KL(q(z) || p(z))
    ```

    where `q(z)` is the variational, or "surrogate", posterior over latents `z`,
    `p(x | z)` is the likelihood of some data `x` conditional on latents `z`,
    and `p(z)` is the prior over `z`.

    In the specific case of the VariationalGaussianProcess model, the
    surrograte posterior `q(z)` is such that the above expectation factorizes
    into a sum over *1-dimensional* integrals of the log likelihood times a
    certain Gaussian distribution (a 1-dimensional marginal of the full
    variational GP). This means we can get a really good estimate of the
    likelihood term using Gauss-Hermite quadrature, which is what this method
    does. In the particular case of a Gaussian likelihood, we can actually get
    an exact answer with 3 quadrature points (we could also work it out
    analytically, but it's still exact and a bit simpler to just have one
    implementation for all likelihoods).

    The `observation_index_points` arguments are optional and if omitted default
    to the `index_points` of this class (ie, the predictive locations).

    ## Example: binary classification

    ```None
      def log_prob(observations, f):
        # Parameterize a collection of independent Bernoulli random variables
        # with logits given by the passed-in function values `f`. Return the
        # joint log probability of the (binary) `observations` under that
        # model.
        berns = tfd.Independent(tfd.Bernoulli(logits=f),
                                reinterpreted_batch_ndims=1)
        return berns.log_prob(observations)

      # Compute the expected log likelihood using Gauss-Hermite quadrature.
      recon = vgp.surrogate_posterior_expected_log_likelihood(
          observations,
          observation_index_points,
          log_likelihood_fn=log_prob,
          quadrature_size=20)

      elbo = -recon + vgp.surrogate_posterior_kl_divergence_prior()
    ```

    Args:
      observations: observed data at the given `observation_index_points`; must
        be acceptable inputs to the given `log_likelihood_fn` callable.
      observation_index_points: `float` `Tensor` representing finite collection,
        or batch of collections, of points in the index set for which some data
        has been observed. Shape has the form `[b1, .., bB, e, f1, ..., fF]'
        where `F` is the number of feature dimensions and must equal
        `self.kernel.feature_ndims`, and `e` is the number (size) of index
        points in each batch. `[b1, ..., bB, e]` must be broadcastable with the
        shape of `observations`, and `[b1, ..., bB]` must be broadcastable with
        the shapes of all other batched parameters of this
        `VariationalGaussianProcess` instance (`kernel.batch_shape`,
        `index_points`, etc).
      log_likelihood_fn: A `callable`, which takes a set of observed data and
        function values (ie, events under this GP model at the
        observation_index_points) and returns the log likelihood of those data
        conditioned on those function values. Default value is `None`, which
        implies a `Normal` likelihood and 3 qudrature points.
      quadrature_size: number of grid points to use in Gauss-Hermite quadrature
        scheme. Default of `10` (arbitrarily), or if `3` if `log_likelihood_fn`
        is `None` (implying a Gaussian likelihood for which `3` points will give
        an exact answer.)
      name: Python `str` name prefixed to Ops created by this class.
        Default value: "surrogate_posterior_expected_log_likelihood".

    Returns:
      surrogate_posterior_expected_log_likelihood: the value of the expected log
        likelihood of the given observed data under the surrogate posterior
        model of latent function values and given likelihood model.
    """
    def _maybe_expand_dims_neg2(a):
      """Inject a `1` into the shape of `a` only if `a` is non-scalar.

      Also handles the dynamic shape case.

      Args:
        a: a `Tensor`.

      Returns:
        maybe_expanded: a `Tensor` whose shape is unchanged if `a` was scalar,
        or, if `a` had shape `[..., A, B]`, a new `Tensor` which is the same as
        `a` but with shape `[..., A, 1, B]`.
      """
      if tensorshape_util.rank(a.shape) == 0:
        return a
      if tensorshape_util.rank(a.shape) is not None:
        return a[..., tf.newaxis, :]
      return tf.cond(tf.equal(tf.rank(a), 0),
                     lambda: a,
                     lambda: a[..., tf.newaxis, :])

    with tf.name_scope(name or 'surrogate_posterior_expected_log_likelihood'):
      if observation_index_points is None:
        observation_index_points = self._index_points
      observation_index_points = tf.convert_to_tensor(
          observation_index_points, dtype=self._dtype,
          name='observation_index_points')
      observations = tf.convert_to_tensor(
          observations, dtype=self._dtype, name='observations')

      if log_likelihood_fn is None:
        scale = _maybe_expand_dims_neg2(
            tf.math.sqrt(self._vgp_observation_noise_variance))
        def _normal_ll(obs, fn_vals):
          return independent.Independent(
              normal.Normal(loc=fn_vals, scale=scale),
              reinterpreted_batch_ndims=1).log_prob(obs)
        log_likelihood_fn = _normal_ll
        # Gauss-Hermite quadrature with 2n - 1 quadrature points is exact for
        # polynomials of degree n. Since the normal log-likelihood is quadratic
        # (n = 2), a sufficient quadrature_size is 3.
        quadrature_size = 3

      qf_loc = self.mean(index_points=observation_index_points)
      qf_scale = self.stddev(index_points=observation_index_points)

      grid, weights = np.polynomial.hermite.hermgauss(quadrature_size)
      grid = grid.astype(dtype_util.as_numpy_dtype(self._dtype))
      weights = weights.astype(dtype_util.as_numpy_dtype(self._dtype))

      # Use this weird _maybe_expand_dims_neg2 function, to only expand dims if
      # the inputs are non-scalar. Also handles the fully dynamic shape case.
      # The `1` we're injecting into the shape will broadcast with the
      # `quadrature_size` Gauss-Hermite grid points below. We need the broadcast
      # to happen at the 2nd to right-most shape position, since the
      # log_likelihood_fn expects to reduce over the right-most dim. Of course,
      # this is trivial if the shape was scalar (hence "maybe") but calls for a
      # reshape if the shape was non-scalar.
      qf_loc = _maybe_expand_dims_neg2(qf_loc)
      qf_scale = _maybe_expand_dims_neg2(qf_scale)
      observations = _maybe_expand_dims_neg2(observations)

      grid = grid[..., tf.newaxis]

      # Change of variables
      fn_values = np.sqrt(2.) * qf_scale * grid + qf_loc
      log_probs = log_likelihood_fn(observations, fn_values)
      result = tf.reduce_sum(weights * log_probs, axis=-1) / np.sqrt(np.pi)
      return result

  def surrogate_posterior_kl_divergence_prior(self, name=None):
    """The KL divergence between the surrograte posterior and GP prior.

    Args:
      name: Python `str` name prefixed to Ops created by this class.
        Default value: "surrogate_posterior_kl_divergence_prior".

    Returns:
      kl_divergence: the value of the KL divergence between the surrograte
        posterior implied by this `VariationalGaussianProcess` instance and the
        prior, which is an unconditional GP with the same kernel and prior
        `mean_fn`
    """
    with tf.name_scope(name or 'surrogate_posterior_kl_divergence_prior'):
      return kullback_leibler.kl_divergence(
          self._variational_inducing_observations_posterior,
          self._inducing_prior)

  def variational_loss(self,
                       observations,
                       observation_index_points=None,
                       log_likelihood_fn=None,
                       quadrature_size=3,
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
      log_likelihood_fn: log likelihood function.
      quadrature_size: num quadrature grid points.
      kl_weight: Amount by which to scale the KL divergence loss between prior
        and posterior.
        Default value: 1.
      name: Python `str` name prefixed to Ops created by this class.
        Default value: 'variational_loss'.
    Returns:
      loss: Scalar tensor representing the negative variational lower bound.
        Can be directly used in a `tf.Optimizer`.

    #### References

    [1]: Hensman, J., Lawrence, N. "Gaussian Processes for Big Data", 2013
         https://arxiv.org/abs/1309.6835
    """

    with tf.name_scope(name or 'variational_loss'):
      if observation_index_points is None:
        observation_index_points = self._index_points
      observation_index_points = tf.convert_to_tensor(
          observation_index_points,
          dtype=self._dtype,
          name='observation_index_points')

      observations = tf.convert_to_tensor(
          observations, dtype=self._dtype, name='observations')
      kl_weight = tf.convert_to_tensor(
          kl_weight, dtype=self._dtype, name='kl_weight')

      recon = self.surrogate_posterior_expected_log_likelihood(
          observations=observations,
          observation_index_points=observation_index_points,
          log_likelihood_fn=log_likelihood_fn,
          quadrature_size=quadrature_size)
      kl_penalty = self.surrogate_posterior_kl_divergence_prior()
      return -recon + kl_weight * kl_penalty

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

    with tf.name_scope(name or 'optimal_variational_posterior'):
      dtype = dtype_util.common_dtype(
          [inducing_index_points,
           observation_index_points,
           observations,
           observation_noise_variance,
           jitter], tf.float32)

      inducing_index_points = tf.convert_to_tensor(
          inducing_index_points, dtype=dtype, name='inducing_index_points')
      observation_index_points = tf.convert_to_tensor(
          observation_index_points,
          dtype=dtype,
          name='observation_index_points')
      observations = tf.convert_to_tensor(
          observations, dtype=dtype, name='observations')
      observation_noise_variance = tf.convert_to_tensor(
          observation_noise_variance,
          dtype=dtype,
          name='observation_noise_variance')
      jitter = tf.convert_to_tensor(jitter, dtype=dtype, name='jitter')

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
