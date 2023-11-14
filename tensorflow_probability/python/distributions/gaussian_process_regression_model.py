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
"""The GaussianProcessRegressionModel distribution class."""

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import cholesky_util
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import gaussian_process
from tensorflow_probability.python.distributions.internal import stochastic_process_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import nest_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import slicing
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.psd_kernels import schur_complement


__all__ = [
    'GaussianProcessRegressionModel',
]


class GaussianProcessRegressionModel(
    gaussian_process.GaussianProcess,
    distribution.AutoCompositeTensorDistribution):
  """Posterior predictive distribution in a conjugate GP regression model.

  This class represents the distribution over function values at a set of points
  in some index set, conditioned on noisy observations at some other set of
  points. More specifically, we assume a Gaussian process prior, `f ~ GP(m, k)`
  with IID normal noise on observations of function values. In this model
  posterior inference can be done analytically. This `Distribution` is
  parameterized by

    * the mean and covariance functions of the GP prior,
    * the set of (noisy) observations and index points to which they correspond,
    * the set of index points at which the resulting posterior predictive
      distribution over function values is defined,
    * the observation noise variance,
    * jitter, to compensate for numerical instability of Cholesky decomposition,

  in addition to the usual params like `validate_args` and `allow_nan_stats`.

  #### Mathematical Details

  Gaussian process regression (GPR) assumes a Gaussian process (GP) prior and a
  normal likelihood as a generative model for data. Given GP mean function `m`,
  covariance kernel `k`, and observation noise variance `v`, we have

  ```none
    f ~ GP(m, k)

                     iid
    (y[i] | f, x[i])  ~  Normal(f(x[i]), v),   i = 1, ... , N
  ```

  where `y[i]` are the noisy observations of function values at points `x[i]`.

  In practice, `f` is an infinite object (eg, a function over `R^n`) which can't
  be realized on a finite machine, but fortunately the marginal distribution
  over function values at a finite set of points is just a multivariate normal
  with mean and covariance given by the mean and covariance functions applied at
  our finite set of points (see [Rasmussen and Williams, 2006][1] for a more
  extensive discussion of these facts).

  We spell out the generative model in detail below, but first, a digression on
  notation. In what follows we drop the indices on vectorial objects such as
  `x[i]`, it being implied that we are generally considering finite collections
  of index points and corresponding function values and noisy observations
  thereof. Thus `x` should be considered to stand for a collection of index
  points (indeed, themselves often vectorial). Furthermore:

    * `f(x)` refers to the collection of function values at the index points in
      the collection `x`",
    * `m(t)` refers to the collection of values of the mean function at the
      index points in the collection `t`, and
    * `k(x, t)` refers to the *matrix* whose entries are values of the kernel
      function `k` at all pairs of index points from `x` and `t`.

  With these conventions in place, we may write

  ```none
    (f(x) | x) ~ MVN(m(x), k(x, x))

    (y | f(x), x) ~ Normal(f(x), v)
  ```

  When we condition on observed data `y` at the points `x`, we can derive the
  posterior distribution over function values `f(x)` at those points. We can
  then compute the posterior predictive distribution over function values `f(t)`
  at a new set of points `t`, conditional on those observed data.

  ```none
    (f(t) | t, x, y) ~ MVN(loc, cov)

    where

    loc = m(t) + k(t, x) @ inv(k(x, x) + v * I) @ (y - m(x))
    cov = k(t, t) - k(t, x) @ inv(k(x, x) + v * I) @ k(x, t)
  ```

  where `I` is the identity matrix of appropriate dimension. Finally, the
  distribution over noisy observations at the new set of points `t` is obtained
  by adding IID noise from `Normal(0., observation_noise_variance)`.

  #### Examples

  ##### Draw joint samples from the posterior predictive distribution in a GP
  regression model

  ```python
  import numpy as np
  import tensorflow.compat.v2 as tf
  import tensorflow_probability as tfp

  tfb = tfp.bijectors
  tfd = tfp.distributions
  psd_kernels = tfp.math.psd_kernels

  # Generate noisy observations from a known function at some random points.
  observation_noise_variance = .5
  f = lambda x: np.sin(10*x[..., 0]) * np.exp(-x[..., 0]**2)
  observation_index_points = np.random.uniform(-1., 1., 50)[..., np.newaxis]
  observations = (f(observation_index_points) +
                  np.random.normal(0., np.sqrt(observation_noise_variance)))

  index_points = np.linspace(-1., 1., 100)[..., np.newaxis]

  kernel = psd_kernels.MaternFiveHalves()

  gprm = tfd.GaussianProcessRegressionModel(
      kernel=kernel,
      index_points=index_points,
      observation_index_points=observation_index_points,
      observations=observations,
      observation_noise_variance=observation_noise_variance)

  samples = gprm.sample(10)
  # ==> 10 independently drawn, joint samples at `index_points`.
  ```

  Above, we have used the kernel with default parameters, which are unlikely to
  be good. Instead, we can train the kernel hyperparameters on the data, as in
  the next example.

  ##### Optimize model parameters via maximum marginal likelihood

  Here we learn the kernel parameters as well as the observation noise variance
  using gradient descent on the maximum marginal likelihood.

  ```python
  # Suppose we have some data from a known function. Note the index points in
  # general have shape `[b1, ..., bB, f1, ..., fF]` (here we assume `F == 1`),
  # so we need to explicitly consume the feature dimensions (just the last one
  # here).
  f = lambda x: np.sin(10*x[..., 0]) * np.exp(-x[..., 0]**2)

  observation_index_points = np.random.uniform(-1., 1., 50)[..., np.newaxis]
  observations = f(observation_index_points) + np.random.normal(0., .05, 50)

  # Define a kernel with trainable parameters. Note we use TransformedVariable
  # to apply a positivity constraint.
  amplitude = tfp.util.TransformedVariable(
    1., tfb.Exp(), dtype=tf.float64, name='amplitude')
  length_scale = tfp.util.TransformedVariable(
    1., tfb.Exp(), dtype=tf.float64, name='length_scale')
  kernel = psd_kernels.ExponentiatedQuadratic(amplitude, length_scale)

  observation_noise_variance = tfp.util.TransformedVariable(
      np.exp(-5), tfb.Exp(), name='observation_noise_variance')

  # We'll use an unconditioned GP to train the kernel parameters.
  gp = tfd.GaussianProcess(
      kernel=kernel,
      index_points=observation_index_points,
      observation_noise_variance=observation_noise_variance)

  optimizer = tf_keras.optimizers.Adam(learning_rate=.05, beta_1=.5, beta_2=.99)

  @tf.function
  def optimize():
    with tf.GradientTape() as tape:
      loss = -gp.log_prob(observations)
    grads = tape.gradient(loss, gp.trainable_variables)
    optimizer.apply_gradients(zip(grads, gp.trainable_variables))
    return loss

  # We can construct the posterior at a new set of `index_points` using the same
  # kernel (with the same parameters, which we'll optimize below).
  index_points = np.linspace(-1., 1., 100)[..., np.newaxis]
  gprm = tfd.GaussianProcessRegressionModel(
      kernel=kernel,
      index_points=index_points,
      observation_index_points=observation_index_points,
      observations=observations,
      observation_noise_variance=observation_noise_variance)

  # First train the model, then draw and plot posterior samples.
  for i in range(1000):
    neg_log_likelihood_ = optimize()
    if i % 100 == 0:
      print("Step {}: NLL = {}".format(i, neg_log_likelihood_))

  print("Final NLL = {}".format(neg_log_likelihood_))

  samples = gprm.sample(10).numpy()
  # ==> 10 independently drawn, joint samples at `index_points`.

  import matplotlib.pyplot as plt
  plt.scatter(np.squeeze(observation_index_points), observations)
  plt.plot(np.stack([index_points[:, 0]]*10).T, samples.T, c='r', alpha=.2)
  ```

  ##### Marginalization of model hyperparameters

  Here we use TensorFlow Probability's MCMC functionality to perform
  marginalization of the model hyperparameters: kernel params as well as
  observation noise variance.

  ```python
  f = lambda x: np.sin(10*x[..., 0]) * np.exp(-x[..., 0]**2)
  observation_index_points = np.random.uniform(-1., 1., 25)[..., np.newaxis]
  observations = np.random.normal(f(observation_index_points), .05)

  gaussian_process_model = tfd.JointDistributionSequential([
    tfd.LogNormal(np.float64(0.), np.float64(1.)),
    tfd.LogNormal(np.float64(0.), np.float64(1.)),
    tfd.LogNormal(np.float64(0.), np.float64(1.)),
    lambda noise_variance, length_scale, amplitude: tfd.GaussianProcess(
        kernel=psd_kernels.ExponentiatedQuadratic(amplitude, length_scale),
        index_points=observation_index_points,
        observation_noise_variance=noise_variance)
  ])

  initial_chain_states = [
      1e-1 * tf.ones([], dtype=np.float64, name='init_amplitude'),
      1e-1 * tf.ones([], dtype=np.float64, name='init_length_scale'),
      1e-1 * tf.ones([], dtype=np.float64, name='init_obs_noise_variance')
  ]

  # Since HMC operates over unconstrained space, we need to transform the
  # samples so they live in real-space.
  unconstraining_bijectors = [
      tfp.bijectors.Softplus(),
      tfp.bijectors.Softplus(),
      tfp.bijectors.Softplus(),
  ]

  def unnormalized_log_posterior(*args):
    return gaussian_process_model.log_prob(*args, x=observations)

  num_results = 200
  @tf.function
  def run_mcmc():
    return tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=500,
        num_steps_between_results=3,
        current_state=initial_chain_states,
        kernel=tfp.mcmc.TransformedTransitionKernel(
            inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=unnormalized_log_posterior,
                step_size=[np.float64(.15)],
                num_leapfrog_steps=3),
            bijector=unconstraining_bijectors),
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)
  [
        amplitudes,
        length_scales,
        observation_noise_variances
  ], is_accepted = run_mcmc()

  print("Acceptance rate: {}".format(np.mean(is_accepted)))

  # Now we can sample from the posterior predictive distribution at a new set
  # of index points.
  index_points = np.linspace(-1., 1., 200)[..., np.newaxis]
  gprm = tfd.GaussianProcessRegressionModel(
      # Batch of `num_results` kernels parameterized by the MCMC samples.
      kernel=psd_kernels.ExponentiatedQuadratic(amplitudes, length_scales),
      index_points=index_points,
      observation_index_points=observation_index_points,
      observations=observations,
      observation_noise_variance=observation_noise_variances)
  samples = gprm.sample()

  # Plot posterior samples and their mean, target function, and observations.
  plt.plot(np.stack([index_points[:, 0]]*num_results).T,
          samples.numpy().T,
          c='r',
          alpha=.01)
  plt.plot(index_points[:, 0], np.mean(samples, axis=0), c='k')
  plt.plot(index_points[:, 0], f(index_points))
  plt.scatter(observation_index_points[:, 0], observations)
  ```

  #### References
  [1]: Carl Rasmussen, Chris Williams. Gaussian Processes For Machine Learning,
       2006.
  """
  # pylint:disable=invalid-name

  def __init__(self,
               kernel,
               index_points=None,
               observation_index_points=None,
               observations=None,
               observation_noise_variance=0.,
               predictive_noise_variance=None,
               mean_fn=None,
               cholesky_fn=None,
               jitter=1e-6,
               validate_args=False,
               allow_nan_stats=False,
               name='GaussianProcessRegressionModel',
               _conditional_kernel=None,
               _conditional_mean_fn=None):
    """Construct a GaussianProcessRegressionModel instance.

    Args:
      kernel: `PositiveSemidefiniteKernel`-like instance representing the
        GP's covariance function.
      index_points: (nested) `Tensor` representing finite collection, or batch
        of collections, of points in the index set over which the GP is
        defined. Shape (of each nested component) has the form `[b1, ..., bB,
        e, f1, ..., fF]` where `F` is the number of feature dimensions and
        must equal `kernel.feature_ndims` (or its corresponding nested
        component) and `e` is the number (size) of index points in each
        batch. Ultimately this distribution corresponds to an `e`-dimensional
        multivariate normal. The batch shape must be broadcastable with
        `kernel.batch_shape` and any batch dims yielded by `mean_fn`.
      observation_index_points: (nested) `Tensor` representing finite
        collection, or batch of collections, of points in the index set for
        which some data has been observed. Shape (of each nested component)
        has the form `[b1, ..., bB, e, f1, ..., fF]` where `F` is the number
        of feature dimensions and must equal `kernel.feature_ndims` (or its
        corresponding nested component), and `e` is the number (size) of
        index points in each batch. `[b1, ..., bB, e]` must be broadcastable
        with the shape of `observations`, and `[b1, ..., bB]` must be
        broadcastable with the shapes of all other batched parameters
        (`kernel.batch_shape`, `index_points`, etc). The default value is
        `None`, which corresponds to the empty set of observations, and
        simply results in the prior predictive model (a GP with noise of
        variance `predictive_noise_variance`).
      observations: `float` `Tensor` representing collection, or batch of
        collections, of observations corresponding to
        `observation_index_points`. Shape has the form `[b1, ..., bB, e]`, which
        must be brodcastable with the batch and example shapes of
        `observation_index_points`. The batch shape `[b1, ..., bB]` must be
        broadcastable with the shapes of all other batched parameters
        (`kernel.batch_shape`, `index_points`, etc.). The default value is
        `None`, which corresponds to the empty set of observations, and simply
        results in the prior predictive model (a GP with noise of variance
        `predictive_noise_variance`).
      observation_noise_variance: `float` `Tensor` representing the variance
        of the noise in the Normal likelihood distribution of the model. May be
        batched, in which case the batch shape must be broadcastable with the
        shapes of all other batched parameters (`kernel.batch_shape`,
        `index_points`, etc.).
        Default value: `0.`
      predictive_noise_variance: `float` `Tensor` representing the variance in
        the posterior predictive model. If `None`, we simply re-use
        `observation_noise_variance` for the posterior predictive noise. If set
        explicitly, however, we use this value. This allows us, for example, to
        omit predictive noise variance (by setting this to zero) to obtain
        noiseless posterior predictions of function values, conditioned on noisy
        observations.
      mean_fn: Python `callable` that acts on `index_points` to produce a
        collection, or batch of collections, of mean values at `index_points`.
        Takes a (nested) `Tensor` of shape `[b1, ..., bB, e, f1, ..., fF]` and
        returns a `Tensor` whose shape is broadcastable with `[b1, ..., bB, e]`.
        Default value: `None` implies the constant zero function.
      cholesky_fn: Callable which takes a single (batch) matrix argument and
        returns a Cholesky-like lower triangular factor.  Default value: `None`,
        in which case `make_cholesky_with_jitter_fn` is used with the `jitter`
        parameter.
      jitter: `float` scalar `Tensor` added to the diagonal of the covariance
        matrix to ensure positive definiteness of the covariance matrix.
        This argument is ignored if `cholesky_fn` is set.
        Default value: `1e-6`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        Default value: `False`.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value `NaN` to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
        Default value: `False`.
      name: Python `str` name prefixed to Ops created by this class.
        Default value: 'GaussianProcessRegressionModel'.
      _conditional_kernel: Internal parameter -- do not use.
      _conditional_mean_fn: Internal parameter -- do not use.

    Raises:
      ValueError: if either
        - only one of `observations` and `observation_index_points` is given, or
        - `mean_fn` is not `None` and not callable.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      input_dtype = dtype_util.common_dtype(
          dict(
              kernel=kernel,
              index_points=index_points,
              observation_index_points=observation_index_points,
          ),
          dtype_hint=nest_util.broadcast_structure(
              kernel.feature_ndims, tf.float32))

      # If the input dtype is non-nested float, we infer a single dtype for the
      # input and the float parameters, which is also the dtype of the GP's
      # samples, log_prob, etc. If the input dtype is nested (or not float), we
      # do not use it to infer the GP's float dtype.
      if (not tf.nest.is_nested(input_dtype) and
          dtype_util.is_floating(input_dtype)):
        dtype = dtype_util.common_dtype(
            dict(
                kernel=kernel,
                index_points=index_points,
                observations=observations,
                observation_index_points=observation_index_points,
                observation_noise_variance=observation_noise_variance,
                predictive_noise_variance=predictive_noise_variance,
                jitter=jitter,
            ),
            dtype_hint=tf.float32,
        )
        input_dtype = dtype
      else:
        dtype = dtype_util.common_dtype(
            dict(
                observations=observations,
                observation_noise_variance=observation_noise_variance,
                predictive_noise_variance=predictive_noise_variance,
                jitter=jitter,
            ), dtype_hint=tf.float32)

      if index_points is not None:
        index_points = nest_util.convert_to_nested_tensor(
            index_points, dtype=input_dtype, convert_ref=False,
            name='index_points', allow_packing=True)
      if observation_index_points is not None:
        observation_index_points = nest_util.convert_to_nested_tensor(
            observation_index_points, dtype=input_dtype, convert_ref=False,
            name='observation_index_points', allow_packing=True)
      observations = tensor_util.convert_nonref_to_tensor(
          observations, dtype=dtype,
          name='observations')
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
      if (observation_index_points is None) != (observations is None):
        raise ValueError(
            '`observations` and `observation_index_points` must both be given '
            'or None. Got {} and {}, respectively.'.format(
                observations, observation_index_points))
      # Default to a constant zero function, borrowing the dtype from
      # index_points to ensure consistency.
      mean_fn = stochastic_process_util.maybe_create_mean_fn(mean_fn, dtype)
      if cholesky_fn is None:
        cholesky_fn = cholesky_util.make_cholesky_with_jitter_fn(jitter)

      self._name = name
      self._observation_index_points = observation_index_points
      self._observations = observations
      self._observation_noise_variance = observation_noise_variance
      self._predictive_noise_variance = predictive_noise_variance
      self._jitter = jitter
      self._validate_args = validate_args

      with tf.name_scope('init'):
        if _conditional_kernel is None:
          _conditional_kernel = schur_complement.SchurComplement(
              base_kernel=kernel,
              fixed_inputs=observation_index_points,
              cholesky_fn=cholesky_fn,
              diag_shift=observation_noise_variance)
        # Special logic for mean_fn only; SchurComplement already handles the
        # case of empty observations (ie, falls back to base_kernel).
        if stochastic_process_util.is_empty_observation_data(
            feature_ndims=kernel.feature_ndims,
            observation_index_points=observation_index_points,
            observations=observations):
          if _conditional_mean_fn is None:
            _conditional_mean_fn = mean_fn
        else:
          stochastic_process_util.validate_observation_data(
              kernel=kernel,
              observation_index_points=observation_index_points,
              observations=observations)

          if _conditional_mean_fn is None:

            def conditional_mean_fn(x):
              """Conditional mean."""
              observations = tf.convert_to_tensor(self._observations)
              observation_index_points = nest_util.convert_to_nested_tensor(
                  self._observation_index_points, dtype_hint=input_dtype,
                  allow_packing=True)
              k_x_obs_linop = tf.linalg.LinearOperatorFullMatrix(
                  kernel.matrix(x, observation_index_points))
              chol_linop = tf.linalg.LinearOperatorLowerTriangular(
                  _conditional_kernel.divisor_matrix_cholesky(
                      fixed_inputs=observation_index_points))

              diff = observations - mean_fn(observation_index_points)
              return mean_fn(x) + k_x_obs_linop.matvec(
                  chol_linop.solvevec(chol_linop.solvevec(diff), adjoint=True))
            _conditional_mean_fn = conditional_mean_fn

        # Store `_conditional_kernel` and `_conditional_mean_fn` as attributes
        # for `AutoCompositeTensor`.
        self._conditional_kernel = _conditional_kernel
        self._conditional_mean_fn = _conditional_mean_fn
        super(GaussianProcessRegressionModel, self).__init__(
            kernel=_conditional_kernel,
            mean_fn=_conditional_mean_fn,
            index_points=index_points,
            cholesky_fn=cholesky_fn,
            jitter=jitter,
            # What the GP super class calls "observation noise variance" we call
            # here the "predictive noise variance". We use the observation noise
            # variance for the fit/solve process above, and predictive for
            # downstream computations like sampling.
            observation_noise_variance=predictive_noise_variance,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats, name=name)
        self._parameters = parameters

  @staticmethod
  def precompute_regression_model(
      kernel,
      observation_index_points,
      observations,
      observations_is_missing=None,
      index_points=None,
      observation_noise_variance=0.,
      predictive_noise_variance=None,
      mean_fn=None,
      cholesky_fn=None,
      jitter=1e-6,
      validate_args=False,
      allow_nan_stats=False,
      name='PrecomputedGaussianProcessRegressionModel',
      _precomputed_divisor_matrix_cholesky=None,
      _precomputed_solve_on_observation=None):
    """Returns a GaussianProcessRegressionModel with precomputed quantities.

    This differs from the constructor by precomputing quantities associated with
    observations in a non-tape safe way. `index_points` is the only parameter
    that is allowed to vary (i.e. is a `Variable` / changes after
    initialization).

    Specifically:

    * We make `observation_index_points` and `observations` mandatory
      parameters.
    * We precompute `kernel(observation_index_points, observation_index_points)`
      along with any other associated quantities relating to the `kernel`,
      `observations` and `observation_index_points`.

    A typical usecase would be optimizing kernel hyperparameters for a
    `GaussianProcess`, and computing the posterior predictive with respect to
    those optimized hyperparameters and observation / index-points pairs.

    WARNING: This method assumes `index_points` is the only varying parameter
    (i.e. is a `Variable` / changes after initialization) and hence is not
    tape-safe.

    Args:
      kernel: `PositiveSemidefiniteKernel`-like instance representing the
        GP's covariance function.
      observation_index_points: (nested) `Tensor` representing finite
        collection, or batch of collections, of points in the index set for
        which some data has been observed. Shape (or shape of each nested
        component) has the form `[b1, ..., bB, e, f1, ..., fF]` where `F` is
        the number of feature dimensions and must equal
        `kernel.feature_ndims` (or its corresponding nested component), and
        `e` is the number (size) of index points in each batch. `[b1, ...,
        bB, e]` must be broadcastable with the shape of `observations`, and
        `[b1, ..., bB]` must be broadcastable with the shapes of all other
        batched parameters (`kernel.batch_shape`, `index_points`, etc). The
        default value is `None`, which corresponds to the empty set of
        observations, and simply results in the prior predictive model (a GP
        with noise of variance `predictive_noise_variance`).
      observations: `float` `Tensor` representing collection, or batch of
        collections, of observations corresponding to
        `observation_index_points`. Shape has the form `[b1, ..., bB, e]`, which
        must be brodcastable with the batch and example shapes of
        `observation_index_points`. The batch shape `[b1, ..., bB]` must be
        broadcastable with the shapes of all other batched parameters
        (`kernel.batch_shape`, `index_points`, etc.). The default value is
        `None`, which corresponds to the empty set of observations, and simply
        results in the prior predictive model (a GP with noise of variance
        `predictive_noise_variance`).
      observations_is_missing:  `bool` `Tensor` of shape `[..., e]`,
        representing a batch of boolean masks.  When `observations_is_missing`
        is not `None`, the returned distribution is conditioned only on the
        observations for which the corresponding elements of
        `observations_is_missing` are `True`.
      index_points: (nested) `Tensor` representing finite collection, or batch
        of collections, of points in the index set over which the GP is defined.
        Shape (or shape of each nested component) has the form `[b1, ..., bB,
        e, f1, ..., fF]` where `F` is the number of feature dimensions and
        must equal `kernel.feature_ndims` (or its corresponding nested
        component) and `e` is the number (size) of index points in each
        batch. Ultimately this distribution corresponds to an `e`-dimensional
        multivariate normal. The batch shape must be broadcastable with
        `kernel.batch_shape` and any batch dims yielded by `mean_fn`.
      observation_noise_variance: `float` `Tensor` representing the variance
        of the noise in the Normal likelihood distribution of the model. May be
        batched, in which case the batch shape must be broadcastable with the
        shapes of all other batched parameters (`kernel.batch_shape`,
        `index_points`, etc.).
        Default value: `0.`
      predictive_noise_variance: `float` `Tensor` representing the variance in
        the posterior predictive model. If `None`, we simply re-use
        `observation_noise_variance` for the posterior predictive noise. If set
        explicitly, however, we use this value. This allows us, for example, to
        omit predictive noise variance (by setting this to zero) to obtain
        noiseless posterior predictions of function values, conditioned on noisy
        observations.
      mean_fn: Python `callable` that acts on `index_points` to produce a
        collection, or batch of collections, of mean values at `index_points`.
        Takes a (nested) `Tensor` of shape `[b1, ..., bB, e, f1, ..., fF]` and
        returns a `Tensor` whose shape is broadcastable with `[b1, ..., bB, e]`.
        Default value: `None` implies the constant zero function.
      cholesky_fn: Callable which takes a single (batch) matrix argument and
        returns a Cholesky-like lower triangular factor.  Default value: `None`,
        in which case `make_cholesky_with_jitter_fn` is used with the `jitter`
        parameter.
      jitter: `float` scalar `Tensor` added to the diagonal of the covariance
        matrix to ensure positive definiteness of the covariance matrix.
        Default value: `1e-6`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        Default value: `False`.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value `NaN` to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
        Default value: `False`.
      name: Python `str` name prefixed to Ops created by this class.
        Default value: 'PrecomputedGaussianProcessRegressionModel'.
      _precomputed_divisor_matrix_cholesky: Internal parameter -- do not use.
      _precomputed_solve_on_observation: Internal parameter -- do not use.
    Returns
      An instance of `GaussianProcessRegressionModel` with precomputed
      quantities associated with observations.
    """

    with tf.name_scope(name) as name:
      if tf.nest.is_nested(kernel.feature_ndims):
        input_dtype = dtype_util.common_dtype(
            dict(
                kernel=kernel,
                index_points=index_points,
                observation_index_points=observation_index_points,
            ),
            dtype_hint=nest_util.broadcast_structure(
                kernel.feature_ndims, tf.float32
            ),
        )
        dtype = dtype_util.common_dtype(
            dict(
                observations=observations,
                observation_noise_variance=observation_noise_variance,
                predictive_noise_variance=predictive_noise_variance,
                jitter=jitter,
            ),
            tf.float32,
        )
      else:
        # If the index points are not nested, we assume they are of the same
        # dtype as the GPRM.
        dtype = dtype_util.common_dtype(
            dict(
                index_points=index_points,
                observation_index_points=observation_index_points,
                observations=observations,
                observation_noise_variance=observation_noise_variance,
                predictive_noise_variance=predictive_noise_variance,
                jitter=jitter,
            ),
            tf.float32,
        )
        input_dtype = dtype

      # Convert-to-tensor arguments that are expected to not be Variables / not
      # going to change.
      jitter = tf.convert_to_tensor(jitter, dtype=dtype)

      observation_index_points = nest_util.convert_to_nested_tensor(
          observation_index_points, dtype=input_dtype, allow_packing=True)
      observation_noise_variance = tf.convert_to_tensor(
          observation_noise_variance, dtype=dtype)
      observations = tf.convert_to_tensor(observations, dtype=dtype)

      if observations_is_missing is not None:
        observations_is_missing = tf.convert_to_tensor(observations_is_missing)

      if cholesky_fn is None:
        cholesky_fn = cholesky_util.make_cholesky_with_jitter_fn(jitter)

      conditional_kernel = schur_complement.SchurComplement.with_precomputed_divisor(
          base_kernel=kernel,
          fixed_inputs=observation_index_points,
          fixed_inputs_is_missing=observations_is_missing,
          cholesky_fn=cholesky_fn,
          diag_shift=observation_noise_variance,
          _precomputed_divisor_matrix_cholesky=(
              _precomputed_divisor_matrix_cholesky))

      mean_fn = stochastic_process_util.maybe_create_mean_fn(mean_fn, dtype)

      solve_on_observation = _precomputed_solve_on_observation
      if solve_on_observation is None:
        observation_cholesky_operator = tf.linalg.LinearOperatorLowerTriangular(
            conditional_kernel.divisor_matrix_cholesky())
        diff = observations - mean_fn(observation_index_points)
        if observations_is_missing is not None:
          diff = tf.where(
              observations_is_missing, tf.zeros([], dtype=diff.dtype), diff)
        solve_on_observation = observation_cholesky_operator.solvevec(
            observation_cholesky_operator.solvevec(diff), adjoint=True)

      def conditional_mean_fn(x):
        k_x_obs = kernel.matrix(x, observation_index_points)
        if observations_is_missing is not None:
          k_x_obs = tf.where(observations_is_missing[..., tf.newaxis, :],
                             tf.zeros([], dtype=k_x_obs.dtype),
                             k_x_obs)
        return mean_fn(x) + tf.linalg.matvec(k_x_obs, solve_on_observation)

      gprm = GaussianProcessRegressionModel(
          kernel=kernel,
          observation_index_points=observation_index_points,
          observations=observations,
          index_points=index_points,
          observation_noise_variance=observation_noise_variance,
          predictive_noise_variance=predictive_noise_variance,
          cholesky_fn=cholesky_fn,
          jitter=jitter,
          _conditional_kernel=conditional_kernel,
          _conditional_mean_fn=conditional_mean_fn,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=name)
      # pylint: disable=protected-access
      gprm._precomputed_divisor_matrix_cholesky = (
          conditional_kernel._precomputed_divisor_matrix_cholesky)
      gprm._precomputed_solve_on_observation = solve_on_observation
      # pylint: enable=protected-access

    return gprm

  @property
  def observation_index_points(self):
    return self._observation_index_points

  @property
  def observations(self):
    return self._observations

  @property
  def predictive_noise_variance(self):
    return self._predictive_noise_variance

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    def _event_ndims_fn(self):
      return tf.nest.map_structure(lambda nd: nd + 1, self.kernel.feature_ndims)
    return dict(
        index_points=parameter_properties.ParameterProperties(
            event_ndims=_event_ndims_fn,
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED,
        ),
        observations=parameter_properties.ParameterProperties(
            event_ndims=1,
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED),
        observation_index_points=parameter_properties.ParameterProperties(
            event_ndims=_event_ndims_fn,
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED,
        ),
        observations_is_missing=parameter_properties.ParameterProperties(
            event_ndims=1,
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED,
        ),
        kernel=parameter_properties.BatchedComponentProperties(),
        _conditional_kernel=parameter_properties.BatchedComponentProperties(),
        observation_noise_variance=parameter_properties.ParameterProperties(
            event_ndims=0,
            shape_fn=lambda sample_shape: sample_shape[:-1],
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        predictive_noise_variance=parameter_properties.ParameterProperties(
            event_ndims=0,
            shape_fn=lambda sample_shape: sample_shape[:-1],
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))

  def __getitem__(self, slices) -> 'GaussianProcessRegressionModel':
    # _conditional_mean_fn is a closure over possibly-sliced values, but will
    # be rebuilt by the constructor.
    return slicing.batch_slice(self, dict(_conditional_mean_fn=None), slices)
