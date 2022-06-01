# Copyright 2021 The TensorFlow Probability Authors.
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
"""Sampler for sparse regression with spike-and-slab prior."""

import collections

import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import bernoulli
from tensorflow_probability.python.distributions import inverse_gamma
from tensorflow_probability.python.distributions import joint_distribution_auto_batched
from tensorflow_probability.python.distributions import sample as sample_dist
from tensorflow_probability.python.experimental.distributions import MultivariateNormalPrecisionFactorLinearOperator
from tensorflow_probability.python.internal import broadcast_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import vectorization_util
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

__all__ = ['SpikeSlabSampler']


class InverseGammaWithSampleUpperBound(inverse_gamma.InverseGamma):
  """Inverse gamma distribution with an upper bound on sampled values."""

  def __init__(self, concentration, scale, upper_bound, **kwargs):
    self._upper_bound = upper_bound
    super().__init__(concentration=concentration, scale=scale, **kwargs)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        concentration=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        upper_bound=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))

  def _sample_n(self, n, seed=None):
    xs = super()._sample_n(n, seed=seed)
    if self._upper_bound is not None:
      xs = tf.minimum(xs, self._upper_bound)
    return xs


class MVNPrecisionFactorHardZeros(
    MultivariateNormalPrecisionFactorLinearOperator):
  """Multivariate normal that forces some sample dimensions to zero.

  This is equivalent to setting `loc[d] = 0.` and `precision_factor[d, d]=`inf`
  in the zeroed dimensions, but is numerically better behaved.
  """

  def __init__(self, loc, precision_factor, nonzeros, **kwargs):
    self._nonzeros = nonzeros
    super().__init__(loc=loc, precision_factor=precision_factor, **kwargs)

  def _call_sample_n(self, *args, **kwargs):
    xs = super()._call_sample_n(*args, **kwargs)
    return tf.where(self._nonzeros, xs, 0.)

  def _log_prob(self, *args, **kwargs):
    raise NotImplementedError('Log prob is not currently implemented.')

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        loc=parameter_properties.ParameterProperties(event_ndims=1),
        precision_factor=parameter_properties.BatchedComponentProperties(),
        precision=parameter_properties.BatchedComponentProperties(),
        nonzeros=parameter_properties.BatchedComponentProperties(event_ndims=1))


class SpikeSlabSamplerState(
    collections.namedtuple('SpikeSlabSamplerState', [
        'x_transpose_y',
        'nonzeros',
        'conditional_prior_precision_chol',
        'conditional_posterior_precision_chol',
        'conditional_weights_mean',
        'weights_posterior_precision',
        'observation_noise_variance_posterior_scale',
        'unnormalized_log_prob'
    ])):
  """Quantities maintained during a sweep of the spike and slab sampler.

    This state is generated and consumed by internal sampler methods. It is not
    intended to be publicly exposed.

    Elements:
      x_transpose_y: (batch of) float `Tensor`(s) of shape `[num_features]`,
        encoding the current regression targets. Equal to
        `matvec(design_matrix, targets, adjoint_a=True)`. Note that this is
        does not depend on the sparsity pattern and so is constant during a
        given sweep.
      nonzeros: (batch of) boolean `Tensor`(s) of shape `[num_features]`
        indicating the current sparsity pattern (`gamma` in [1]). A value of
        `True` indicates that the corresponding feature has nonzero weight.
      conditional_prior_precision_chol: (batch of) float `Tensor`(s) of shape
        `[num_features, num_features]`, giving the Cholesky factor of the
        prior precision matrix 'restricted' to the current nonzero weights
        (`Omega_gamma^{-1}` in [1]). Note that the matrix shape does not vary
        with the sparsity pattern; the restriction is implemented by replacing
        the unused entries by the identity matrix (see `_select_nonzero_block`).
      conditional_posterior_precision_chol: (batch of) float `Tensor`(s) of
        shape `[num_features, num_features]`, giving the Cholesky factor of the
        posterior precision matrix restricted to the current nonzero weights
        (`V_gamma^{-1}` in [1]). Note that the matrix shape does not vary with
        the sparsity pattern; the restriction is implemented by replacing the
        unused entries by the identity matrix (see `_select_nonzero_block`).
      conditional_posterior_mean: (batch of) float `Tensor`(s) of shape
        `[num_features]`, giving the posterior mean weight vector (`beta_gamma`
        in [1]). This has nonzero values in locations where `nonzeros` is True,
        and zeros elsewhere.
      weights_posterior_precision: (batch of) float `Tensor`(s) of shape
        `[num_features]`. This may optionally vary with the observation noise,
        so is stored in the state, rather than the class. (`V^-1` in [1])
        sampled posterior (`SS_gamma / 2` in [1]).
      observation_noise_variance_posterior_scale: (batch of) scalar float
        `Tensor`s representing the scale parameter of the inverse gamma
        posterior on the observation noise variance (`SS_gamma / 2` in [1]).
        Note that the concentration parameter is fixed given `num_outputs` and
        so does not appear in the sampler state.
      unnormalized_log_prob: (batch of) scale float `Tensor`(s) score for
        the sparsity pattern represented by this state (eqn (8) in [1]).

    All quantities in the state tuple may have batch dimensions, which must
    be the same across all components of the tuple.

  #### References

  [1] Steven L. Scott and Hal Varian. Predicting the Present with Bayesian
      Structural Time Series. __International Journal of Mathematical Modelling
      and Numerical Optimisation 5.1-2 (2014): 4-23.__
      https://people.ischool.berkeley.edu/~hal/Papers/2013/pred-present-with-bsts.pdf
  """
  pass


class SpikeSlabSampler(object):
  """Sampler for Bayesian regression with a spike-and-slab prior on weights.

  This implementation follows the sampler described in section 3.2
  of Scott and Varian, 2013 [1].

  ### Model

  This sampler assumes the regression model

  ```
  y ~ Normal(loc=matvec(design_matrix,  # `X` in [1].
                        weights),       # `beta` in `[1]`.
             scale=observation_noise_scale)  # `sigma_epsilon` in [1].
  ```

  where the design matrix has shape `[num_outputs, num_features]`, with a
  conjugate InverseGamma prior on the noise variance (eqn (6) of [1]):

  ```
  observation_noise_scale**2 ~ InverseGamma(
    concentration=observation_noise_variance_prior_concentration,
    scale=observation_noise_variance_prior_scale)
  ```

  and a spike-and-slab prior on the weights (eqns (5) and (6) of [1]):

  ```
  slab_weights ~ MultivariateNormal(
     loc=0.,  # `b` from [1].
     precision=(weights_prior_precision  # `Omega^{-1}` from [1].
                / observation_noise_scale**2))
  nonzeros ~ Bernoulli(probs=nonzero_prior_prob)  # `gamma` from [1].
  weights = slab_weights * nonzeros
  ```

  ### Example

  Constructing a sampler instance specifies the model priors:

  ```python
  sampler = spike_and_slab.SpikeSlabSampler(
    design_matrix=design_matrix,
    observation_noise_variance_prior_concentration=1.,
    observation_noise_variance_prior_scale=1.
    nonzero_prior_prob=0.1)
  ```

  The sampler instance itself is stateless, though some internal methods take
  or accept `SpikeSlabSamplerState` tuples representing posterior quantities
  maintained within a sampling pass. The sampler is
  invoked by passing the regression targets (`y`) and the initial sparsity
  pattern (`nonzeros`):

  ```
  (observation_noise_variance,
   weights) = sampler.sample_noise_variance_and_weights(
     targets=y, initial_nonzeros=tf.ones([num_features], dtype=tf.bool))
  ```

  This implements the stochastic search variable selection (SSVS) algorithm [2],
  sweeping over the features in random order to resample their sparsity
  indicators one by one. It then returns a sample from the joint posterior
  on the regression weights and the observation noise variance, conditioned
  on the resampled sparsity pattern.

  #### References

  [1] Steven L. Scott and Hal Varian. Predicting the Present with Bayesian
      Structural Time Series. __International Journal of Mathematical Modelling
      and Numerical Optimisation 5.1-2 (2014): 4-23.__
      https://people.ischool.berkeley.edu/~hal/Papers/2013/pred-present-with-bsts.pdf

  [2] George, E. I. and McCulloch, R. E. Approaches for Bayesian variable
      selection. __Statistica Sinica 7, 339–374 (1997)__.
  """

  def __init__(self,
               design_matrix,
               nonzero_prior_prob=0.5,
               weights_prior_precision=None,
               default_pseudo_observations=1.,
               observation_noise_variance_prior_concentration=0.005,
               observation_noise_variance_prior_scale=0.0025,
               observation_noise_variance_upper_bound=None,
               num_missing=0.):
    """Initializes priors for the spike and slab sampler.

    Args:
      design_matrix: (batch of) float `Tensor`(s) regression design matrix (`X`
        in [1]) having shape `[num_outputs, num_features]`.
      nonzero_prior_prob: scalar float `Tensor` prior probability of the 'slab',
        i.e., prior probability that any given feature has nonzero weight (`pi`
        in [1]). Default value: `0.5`.
      weights_prior_precision: (batch of) float `Tensor` complete prior
        precision matrix(s) over the weights, of shape `[num_features,
        num_features]`. If not specified, defaults to the Zellner g-prior
        specified in `[1]` as `Omega^{-1} = kappa * (X'X + diag(X'X)) / (2 *
        num_outputs)`, in which we've plugged in the suggested default of `w =
        0.5`. The parameter `kappa` is controlled by the
        `default_pseudo_observations` argument. Default value: `None`.
      default_pseudo_observations: scalar float `Tensor` Controls the number of
        pseudo-observations for the prior precision matrix over the weights.
        Corresponds to `kappa` in [1]. See also `weights_prior_precision`.
      observation_noise_variance_prior_concentration: scalar float `Tensor`
        concentration parameter of the inverse gamma prior on the noise
        variance. Corresponds to `nu / 2` in [1]. Default value: 0.005.
      observation_noise_variance_prior_scale: scalar float `Tensor` scale
        parameter of the inverse gamma prior on the noise variance. Corresponds
        to `ss / 2` in [1]. Default value: 0.0025.
      observation_noise_variance_upper_bound: optional scalar float `Tensor`
        maximum value of sampled observation noise variance. Specifying a bound
        can help avoid divergence when the sampler is initialized far from the
        posterior. Default value: `None`.
      num_missing: Optional scalar float `Tensor`. Corrects for how many missing
        values are are coded as zero in the design matrix.
    """
    with tf.name_scope('spike_slab_sampler'):
      dtype = dtype_util.common_dtype([
          design_matrix, nonzero_prior_prob, weights_prior_precision,
          observation_noise_variance_prior_concentration,
          observation_noise_variance_prior_scale,
          observation_noise_variance_upper_bound, num_missing
      ],
                                      dtype_hint=tf.float32)
      design_matrix = tf.convert_to_tensor(design_matrix, dtype=dtype)
      nonzero_prior_prob = tf.convert_to_tensor(nonzero_prior_prob, dtype=dtype)
      observation_noise_variance_prior_concentration = tf.convert_to_tensor(
          observation_noise_variance_prior_concentration, dtype=dtype)
      observation_noise_variance_prior_scale = tf.convert_to_tensor(
          observation_noise_variance_prior_scale, dtype=dtype)
      num_missing = tf.convert_to_tensor(num_missing, dtype=dtype)
      if observation_noise_variance_upper_bound is not None:
        observation_noise_variance_upper_bound = tf.convert_to_tensor(
            observation_noise_variance_upper_bound, dtype=dtype)

      design_shape = ps.shape(design_matrix)
      num_outputs = tf.cast(design_shape[-2], dtype=dtype) - num_missing
      num_features = design_shape[-1]

      x_transpose_x = tf.matmul(design_matrix, design_matrix, adjoint_a=True)
      if weights_prior_precision is None:
        # Default prior: 'Zellner’s g−prior' from section 3.2.1 of [1]:
        #   `omega^{-1} = kappa * (w X'X + (1 − w) diag(X'X))/n`
        # with default `w = 0.5`.
        padded_inputs = broadcast_util.left_justified_expand_dims_like(
            num_outputs, x_transpose_x)
        weights_prior_precision = default_pseudo_observations * tf.linalg.set_diag(
            0.5 * x_transpose_x,
            tf.linalg.diag_part(x_transpose_x)) / padded_inputs

      observation_noise_variance_posterior_concentration = (
          observation_noise_variance_prior_concentration +
          tf.convert_to_tensor(num_outputs / 2., dtype=dtype))

      self.num_outputs = num_outputs
      self.num_features = num_features
      self.design_matrix = design_matrix
      self.x_transpose_x = x_transpose_x
      self.dtype = dtype
      self.nonzeros_prior = sample_dist.Sample(
          bernoulli.Bernoulli(probs=nonzero_prior_prob),
          sample_shape=[num_features])
      self.weights_prior_precision = weights_prior_precision
      self.observation_noise_variance_prior_concentration = (
          observation_noise_variance_prior_concentration)
      self.observation_noise_variance_prior_scale = (
          observation_noise_variance_prior_scale)
      self.observation_noise_variance_upper_bound = (
          observation_noise_variance_upper_bound)
      self.observation_noise_variance_posterior_concentration = (
          observation_noise_variance_posterior_concentration)

  def sample_noise_variance_and_weights(self,
                                        targets,
                                        initial_nonzeros,
                                        seed,
                                        previous_observation_noise_variance=1.):
    """(Re)samples regression parameters under the spike-and-slab model.

    Args:
      targets: (batch of) float Tensor regression targets (y-values), of shape
        `[num_outputs]`.
      initial_nonzeros: (batch of) boolean Tensor vector(s) of shape
        `[num_features]`.
      seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
      previous_observation_noise_variance: Optional float to scale the
        `weights_prior_precision`. This behavior is not recommended.

    Returns:
      observation_noise_variance: (batch of) scalar float Tensor posterior
        sample(s) of the observation noise variance, given the resampled
        sparsity pattern.
      weights: (batch of) float Tensor posterior sample(s) of the weight vector
        given the resampled sparsity pattern (encoded as zeros in the
        weight vector) *and* the sampled observation noise variance. Has
        shape `[num_features]`.
    """
    previous_observation_noise_variance = tf.convert_to_tensor(
        previous_observation_noise_variance, dtype=self.dtype)
    feature_sweep_seed, resample_seed = samplers.split_seed(seed, n=2)
    initial_state = self._initialize_sampler_state(
        targets=targets,
        observation_noise_variance=previous_observation_noise_variance,
        nonzeros=initial_nonzeros)
    # Loop over the features to update their sparsity indicators.
    final_state = self._resample_all_features(
        initial_state, seed=feature_sweep_seed)
    # Finally, sample parameters given the updated sparsity indicators.
    return self._get_conditional_posterior(final_state).sample(
        seed=resample_seed)

  def _initialize_sampler_state(self, targets, nonzeros,
                                observation_noise_variance):
    """Precompute quantities needed to sample with given targets.

    This method computes a sampler state (including factorized precision
    matrices) from scratch for a given sparsity pattern. This requires
    time proportional to `num_features**3`. If a sampler state is already
    available for an off-by-one sparsity pattern, the `_flip_feature` method
    (which takes time proportional to `num_features**2`) is
    generally more efficient.

    Args:
      targets: (batch of) float Tensor regression outputs of shape
        `[num_outputs]`.
      nonzeros: (batch of) boolean Tensor vectors of shape `[num_features]`.
      observation_noise_variance: float Tensor of to scale the posterior
        precision.

    Returns:
      sampler_state: instance of `SpikeSlabSamplerState` collecting (potentially
        batched) Tensor quantities relevant to the sampler. See
        `SpikeSlabSamplerState` for details.
    """
    with tf.name_scope('initialize_sampler_state'):
      targets = tf.convert_to_tensor(targets, dtype=self.dtype)
      nonzeros = tf.convert_to_tensor(nonzeros, dtype=tf.bool)

      x_transpose_y = tf.linalg.matvec(
          self.design_matrix, targets, adjoint_a=True)

      # Ensure that `nonzeros` has full batch shape.
      batch_shape = ps.shape(x_transpose_y)[:-1]
      nonzeros = tf.broadcast_to(
          nonzeros,
          ps.broadcast_shape(
              ps.shape(nonzeros), ps.concat([batch_shape, [1]], axis=0)))

      weights_posterior_precision = self.x_transpose_x + self.weights_prior_precision * observation_noise_variance
      conditional_prior_precision_chol = tf.linalg.cholesky(
          _select_nonzero_block(self.weights_prior_precision, nonzeros))
      conditional_posterior_precision_chol = tf.linalg.cholesky(
          _select_nonzero_block(weights_posterior_precision,
                                nonzeros))
      conditional_weights_mean = tf.where(
          nonzeros,
          tf.linalg.cholesky_solve(conditional_posterior_precision_chol,
                                   x_transpose_y[..., tf.newaxis])[..., 0], 0.)
      return self._compute_log_prob(
          x_transpose_y=x_transpose_y,
          nonzeros=nonzeros,
          conditional_prior_precision_chol=conditional_prior_precision_chol,
          conditional_posterior_precision_chol=conditional_posterior_precision_chol,
          weights_posterior_precision=weights_posterior_precision,
          conditional_weights_mean=conditional_weights_mean,
          observation_noise_variance_posterior_scale=(
              # SS_gamma / 2 from eqn (7) of [1].
              self.observation_noise_variance_prior_scale +  # ss / 2
              (
                  tf.reduce_sum(targets**2, axis=-1) -  # y'y
                  tf.reduce_sum(  # beta_gamma' V_gamma^{-1} beta_gamma
                      conditional_weights_mean * x_transpose_y,
                      axis=-1)) / 2))

  def _flip_feature(self, sampler_state, idx):
    """Proposes flipping the sparsity indicator of the `idx`th feature.

    This method computes the sampler state (including factorized precision
    matrices) for a given sparsity pattern, given the state for a
    related sparsity pattern that differs in a single position. This is
    achieved using rank-1 Cholesky updates running in time
    proportional to `num_features**2`, and so is typically more efficient than
    recomputing the equivalent state from scratch using
    `_initialize_sampler_state`.

    Args:
      sampler_state: instance of `SpikeSlabSamplerState` collecting (potentially
        batched) Tensor quantities relevant to the sampler. See the
        `SpikeSlabSamplerState` definition for details.
      idx: scalar int `Tensor` index in `[0, num_features)`. This is a single
        value shared across all batch elements.

    Returns:
      updated_sampler_state: instance of `SpikeSlabSamplerState` equivalent to
        `self._initialize_sampler_state(targets, new_nonzeros)`, where
        `new_nonzeros` is equal to `nonzeros` with the `idx`th entry
        negated.
    """
    with tf.name_scope('flip_feature_indicator'):
      was_nonzero = tf.gather(sampler_state.nonzeros, idx, axis=-1)
      new_nonzeros = _set_vector_index(sampler_state.nonzeros, idx,
                                       tf.logical_not(was_nonzero))

      # Update the weight posterior mean and precision for the new nonzeros.
      # (and also update the prior, used to compute the marginal likelihood).
      new_conditional_prior_precision_chol = _update_nonzero_block_chol(
          # Low-rank update equivalent to
          # `tf.linalg.cholesky(
          #    _select_nonzero_block(self.weights_prior_precision, nonzeros))`.
          chol=sampler_state.conditional_prior_precision_chol,
          idx=idx,
          psd_matrix=self.weights_prior_precision,
          new_nonzeros=new_nonzeros,
          previous_nonzeros=sampler_state.nonzeros)
      new_conditional_posterior_precision_chol = _update_nonzero_block_chol(
          chol=sampler_state.conditional_posterior_precision_chol,
          idx=idx,
          psd_matrix=sampler_state.weights_posterior_precision,
          new_nonzeros=new_nonzeros,
          previous_nonzeros=sampler_state.nonzeros)
      new_conditional_weights_mean = tf.where(
          new_nonzeros,
          tf.linalg.cholesky_solve(new_conditional_posterior_precision_chol,
                                   sampler_state.x_transpose_y[...,
                                                               tf.newaxis])[...,
                                                                            0],
          0.)
      return self._compute_log_prob(
          nonzeros=new_nonzeros,
          conditional_prior_precision_chol=(
              new_conditional_prior_precision_chol),
          conditional_posterior_precision_chol=(
              new_conditional_posterior_precision_chol),
          conditional_weights_mean=new_conditional_weights_mean,
          weights_posterior_precision=sampler_state.weights_posterior_precision,
          observation_noise_variance_posterior_scale=(
              sampler_state.observation_noise_variance_posterior_scale -
              tf.reduce_sum(
                  (new_conditional_weights_mean -
                   sampler_state.conditional_weights_mean) *
                  sampler_state.x_transpose_y,
                  axis=-1) / 2),
          x_transpose_y=sampler_state.x_transpose_y)

  def _resample_all_features(self, initial_sampler_state, seed):
    """Loops over all features to resample their sparsity indicators.

    The sampler loops over the features in random order, where each iteration
    updates the `nonzeros` indicator for that particular (single) feature
    weight. This update is a collapsed Gibbs sampling step, i.e., it samples
    from the posterior on the current sparsity indicator given the remaining
    indicators, after marginalizing (collapsing) out the observation noise
    variance and the continuous regression weights under their conjugate priors.

    Args:
      initial_sampler_state: instance of `SpikeSlabSamplerState` collecting
        (potentially batched) Tensor quantities relevant to the sampler. See
        `SpikeSlabSamplerState` for details.
      seed: PRNG seed; see `tfp.random.sanitize_seed` for details.

    Returns:
      final sampler_state: instance of `SpikeSlabSamplerState` in which the
        sparsity indicators for all features have been resampled.
    """
    with tf.name_scope('resample_all_features'):
      feature_seed, loop_seed = samplers.split_seed(seed, n=2)

      # Visit features in random order.
      feature_permutation = tf.argsort(
          tf.random.stateless_uniform([self.num_features], seed=feature_seed))

      @tf.function(autograph=False)
      def resample_one_feature(step, seed, sampler_state):
        seed, next_seed = samplers.split_seed(seed, n=2)
        idx = tf.gather(feature_permutation, step)

        # Maybe flip this weight's sparsity indicator.
        proposed_sampler_state = self._flip_feature(sampler_state, idx=idx)
        should_flip = bernoulli.Bernoulli(
            logits=(proposed_sampler_state.unnormalized_log_prob -
                    sampler_state.unnormalized_log_prob),
            dtype=tf.bool).sample(seed=seed)
        return step + 1, next_seed, mcmc_util.choose(should_flip,
                                                     proposed_sampler_state,
                                                     sampler_state)

      _, _, final_sampler_state = tf.while_loop(
          cond=lambda step, *args: step < self.num_features,
          body=resample_one_feature,
          loop_vars=(0, loop_seed, initial_sampler_state))
      return final_sampler_state

  def _compute_log_prob(self, x_transpose_y, nonzeros,
                        conditional_prior_precision_chol,
                        conditional_posterior_precision_chol,
                        conditional_weights_mean,
                        weights_posterior_precision,
                        observation_noise_variance_posterior_scale):  # pylint: disable=g-doc-args
    """Computes an unnormalized log prob of a sampler state.

    This corresponds to equation (8) in [1]. It scores a sparsity pattern by
    the marginal likelihood of the observed targets (ignoring constant terms
    that do not depend on the sparsity pattern) multiplied by the prior
    probability of the sparsity pattern.

    Args: See `SpikeSlabSamplerState`.

    Returns:
      sampler_state: a `SpikeSlabSamplerState` instance containing the given
        args and the corresponding unnormalized log prob.
    """
    return SpikeSlabSamplerState(
        x_transpose_y=x_transpose_y,
        nonzeros=nonzeros,
        conditional_prior_precision_chol=conditional_prior_precision_chol,
        conditional_posterior_precision_chol=conditional_posterior_precision_chol,
        conditional_weights_mean=conditional_weights_mean,
        weights_posterior_precision=weights_posterior_precision,
        observation_noise_variance_posterior_scale=(
            observation_noise_variance_posterior_scale),
        unnormalized_log_prob=(  # Equation (8) of [1].
            _half_logdet(conditional_prior_precision_chol) -
            _half_logdet(conditional_posterior_precision_chol) +
            self.nonzeros_prior.log_prob(nonzeros) -
            (self.observation_noise_variance_posterior_concentration - 1) *
            tf.math.log(2 * observation_noise_variance_posterior_scale)))

  def _get_conditional_posterior(self, sampler_state):
    """Builds the joint posterior for a sparsity pattern (eqn (7) from [1])."""

    @joint_distribution_auto_batched.JointDistributionCoroutineAutoBatched
    def posterior_jd():
      observation_noise_variance = yield InverseGammaWithSampleUpperBound(
          concentration=(
              self.observation_noise_variance_posterior_concentration),
          scale=sampler_state.observation_noise_variance_posterior_scale,
          upper_bound=self.observation_noise_variance_upper_bound,
          name='observation_noise_variance')
      yield MVNPrecisionFactorHardZeros(
          loc=sampler_state.conditional_weights_mean,
          # Note that the posterior precision varies inversely with the
          # noise variance: in worlds with high noise we're also
          # more uncertain about the values of the weights.
          # TODO(colcarroll): Tests pass even without a square root on the
          # observation_noise_variance. Should add a test that would fail.
          precision_factor=tf.linalg.LinearOperatorLowerTriangular(
              sampler_state.conditional_posterior_precision_chol /
              tf.sqrt(observation_noise_variance[..., tf.newaxis, tf.newaxis])),
          nonzeros=sampler_state.nonzeros,
          name='weights')

    return posterior_jd


def _select_nonzero_block(matrix, nonzeros):
  """Replaces the `i`th row & col with the identity if not `nonzeros[i]`.

  This function effectively selects the 'slab' rows (corresponding to
  features with nonzero weight) of a prior or posterior precision matrix. To
  guarantee static shapes (and to support batching over different sparsity
  patterns), it returns a matrix of the same shape as the input, in which the
  rows not selected are replaced with the corresponding row of an identity
  matrix. The result can thus be viewed as a permutation of a block-diagonal
  matrix:

  ```
  | matrix[nonzeros, nonzeros]    0 |
  | 0                             I |
  ```

  and this view justifies the correctness of matrix operations such as
  Cholesky factorization on the selected submatrix, although in actuality
  the features are left at their original indices (not permuted).

  Args:
    matrix: (batch of) float Tensor matrix(s) of shape `[num_features,
      num_features]`.
    nonzeros: (batch of) boolean Tensor vectors of shape `[num_features]`.

  Returns:
    block_matrix: (batch of) float Tensor matrix(s) of the same shape as
      `matrix`, in which `block_matrix[i, j] = matrix[i, j] if
      `(nonzeros[i] and nonzeros[j]) else eye(num_features)[i, j]`. This is
      positive semidefinite if `matrix` is, since it is a permutation
      of a matrix with PSD blocks.
  """
  # Zero out all entries in the not-selected rows.
  masked = tf.where(nonzeros[..., tf.newaxis],
                    tf.where(nonzeros[..., tf.newaxis, :], matrix, 0.), 0.)
  # Restore a value of 1 on the diagonal of the not-selected rows. This avoids
  # numerical issues by ensuring that the matrix still has full rank.
  return tf.linalg.set_diag(masked,
                            tf.where(nonzeros, tf.linalg.diag_part(masked), 1.))


def _update_nonzero_block_chol(chol, idx, psd_matrix, new_nonzeros,
                               previous_nonzeros):
  """Efficient update to the cholesky factor of the 'slab' (nonzero) submatrix.

  This performs an efficient update when `nonzeros` changes by a single entry.
  It is equivalent to

  ```
  updated_chol = cholesky(_select_nonzero_block(psd_matrix, new_nonzeros))
  ```

  but requires time only quadratic in the dimension `num_features`, where
  a naive recomputation would take cubic time.


  Args:
    chol: (batch of) float Tensor lower-triangular Cholesky factor(s) of
      `select_nonzero_block(psd_matrix, previous_nonzeros)`, where
      `previous_nonzeros` differs from `new_nonzeros` in the `idx`th entry only.
      This has shape `[num_features, num_features]`.
    idx: scalar int `Tensor` feature index in `[0, num_features)`.
    psd_matrix: (batch of) float Tensor positive semidefinite matrix(s) of shape
      `[num_features, num_features]`.
    new_nonzeros: (batch of) boolean Tensor vectors of shape `[num_features]`.
    previous_nonzeros: (batch of) boolean Tensor vectors of shape
      `[num_features]`.

  Returns:
    updated_chol: (batch of) float Tensor lower-triangular Cholesky factor(s) of
      `select_nonzero_block(psd_matrix, new_nonzeros)`.
  """
  psd_row = tf.where(new_nonzeros, psd_matrix[..., idx, :], 0.)
  eye_row = _set_vector_index(tf.zeros_like(psd_row), idx, 1.)
  new_row = tf.where(new_nonzeros[..., idx, tf.newaxis], psd_row, eye_row)
  # NOTE: We could also compute `old_row` from `chol`, but we believe it is
  # more numerically accurate to use `psd_matrix`, as `chol` may have
  # accumulated errors over multiple calls to `_update_nonzero_block_chol`.
  old_row = _select_nonzero_block(psd_matrix, previous_nonzeros)[..., idx, :]
  return _symmetric_increment_chol(
      chol,
      idx=idx,
      # Set the `idx`th row/col to its target value if the `idx`th feature is
      # now nonzero; otherwise set it to the identity.
      increment=new_row - old_row)


def _symmetric_increment_chol(chol, idx, increment):
  """Effectly increments a row and column of a Cholesky-factorized matrix.

  Let `M = chol @ chol.T` be an `[m, m]` symmetric matrix. This function
  implements the symmetry-preserving operation:

  ```
  M[idx, :] += increment
  M[:, idx] += increment
  # Correct for double-incrementing the diagonal.
  M[idx, idx] -= increment[idx]
  ```

  in Cholesky space, but in an optimized form as 2 steps, where `increment` is
  a vector of length `m`.

  That is, this function adds `increment` to the `idx`th row, and
  (by symmetry) also to the `idx`th column. For example:

  ```python
  M = [[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]]
  new_chol = _symmetric_increment_chol(tf.linalg.cholesky(M),
                                       idx=2,
                                       increment=[0., -0.3, 1.])
  new_M = tf.matmul(new_chol, new_chol, adjoint_b=True)
  # ==> [[1., 0., 0.],
  #      [0., 1., -0.3],
  #      [0., -0.3, 2.]]
  ```

  This is implemented efficiently as two consecutive rank-1 updates of
  `chol(M)`.

  Args:
    chol: (batch of) float `Tensor` lower-triangular Cholesky factor(s) of a
      matrix `M`.
    idx: (batch of) int `Tensor` scalar index(s) of the row and column to
      update.
    increment: (batch of) float `Tensor` vector(s) to add to the given row and
      column of `M`.

  Returns:
    updated_chol: float `Tensor` lower-triangular Cholesky factor of the
      symmetric matrix resulting from adding `increment` to the
      given row and column of `M`.
  """
  with tf.name_scope('symmetric_increment_chol'):
    # TODO(jburnim): Can we make this more numerically accurate by doing both
    # rank-1 Cholesky updates in a single pass?
    chol = tf.convert_to_tensor(chol, name='chol')
    increment = tf.convert_to_tensor(increment, name='increment')
    orig_chol = chol

    # This does an update of the row and column in 2 rank-1 updates.
    # Consider an example update vector of v = [x, y, z]. Thus v @ v.T is:
    # [[x^2, xy, xz],
    #  [xy, y^2, yz],
    #  [xz, yz, z^2]]
    # cholesky_update will compute the return the updated cholesky given
    # this being added to the original matrix.
    #
    # Say we want update row and column 1, then the needed offset matrix is:
    # [[0, x, 0],
    #  [x, y, z],
    #  [0, z, 0]]
    # which is rank 2 and will require at least two rank 1 operations.
    #
    # If we do two updates, by adding v1 and subtracting v2, where
    #  v1 = [x, (y + 1)/2, z]
    #  v2 = [x, (y - 1)/2, z]
    # this accomplishes the goal, since:
    # [[0, x, 0],
    #  [x, y, z],   = v1 @ v1.T - v2 @ v2.T
    #  [0, z, 0]]
    a = (increment[..., idx] + 1.) / 2.
    b = (increment[..., idx] - 1.) / 2.
    chol = tfp_math.cholesky_update(
        chol, update_vector=_set_vector_index(increment, idx, a), multiplier=1)
    chol = tfp_math.cholesky_update(
        chol, update_vector=_set_vector_index(increment, idx, b), multiplier=-1)

    # There Cholesky decomposition should be unchanged in rows/cols before idx.
    #
    # TODO(b/229298550): Investigate whether this is really necessary, or if the
    # test failures we see without this line are due to an underlying bug.
    return tf.where((tf.range(chol.shape[-1]) < idx)[..., tf.newaxis],
                    orig_chol, chol)


def _set_vector_index_unbatched(v, idx, x):
  """Mutation-free equivalent of `v[idx] = x."""
  return tf.tensor_scatter_nd_update(v, indices=[[idx]], updates=[x])


_set_vector_index = vectorization_util.make_rank_polymorphic(
    _set_vector_index_unbatched, core_ndims=[1, 0, 0])


def _half_logdet(chol):
  return tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)), axis=-1)
