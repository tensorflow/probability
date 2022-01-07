"""Meta-distribution to apply importance resampling to a proposal dist."""
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

import tensorflow as tf
from tensorflow_probability.python.distributions import categorical
from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import nest_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.mcmc.internal import util as mcmc_util


class ImportanceResample(distribution_lib.Distribution):
  """Models the distribution of finitely many importance-reweighted samples.

  This wrapper adapts a proposal distribution towards a target density using
  [importance sampling](https://en.wikipedia.org/wiki/Importance_sampling).
  Given a proposal `q`, a target density `p` (which may be unnormalized), and
  an integer `importance_sample_size`, it models the result of the following
  sampling process:

  1. Draw `importance_sample_size` samples `z[k] ~ q` from the proposal.
  2. Compute an importance weight `w[k] = p(z[k]) / q(z[k])` for each sample.
  3. Return a sample `z[k*]` selected with probability proportional to
     the importance weights, i.e., with `k* ~ Categorical(probs=w/sum(w))`.

  In the limit where `importance_sample_size -> inf`, the result `z[k*]` of this
  procedure would be distributed according to the target density `p`. On the
  other hand, if `importance_sample_size == 1`, then the reweighting has no
  effect and the result `z[k*]` is simply a sample from `q`. Finite values
  of `importance_sample_size` describe distributions that are intermediate
  between `p` and `q`.

  This distribution may also be understood as an explicit representation of the
  surrogate posterior that is implicitly assumed by importance-weighted
  variational objectives. [1, 2]

  #### Examples

  This distribution can be used directly for posterior inference via importance
  sampling:

  ```python
  tfd = tfp.distributions
  tfed = tfp.experimental.distributions

  def target_log_prob_fn(x):
    prior = tfd.Normal(loc=0., scale=1.).log_prob(x)
    # Multimodal likelihood.
    likelihood = tf.reduce_logsumexp(
      tfd.Normal(loc=x, scale=0.1).log_prob([-1., 1.]))
    return prior + likelihood

  # Use importance sampling to infer an approximate posterior.
  approximate_posterior = tfed.ImportanceResample(
    proposal_distribution=tfd.Normal(loc=0., scale=2.),
    target_log_prob_fn=target_log_prob_fn,
    importance_sample_size=100)
  ```

  We can estimate posterior expectations directly using an importance-weighted
  sum of proposal samples:

  ```python
  # Directly compute expectations under the posterior via importance weights.
  posterior_mean = approximate_posterior.self_normalized_expectation(
    lambda x: x, importance_sample_size=1000)
  posterior_variance = approximate_posterior.self_normalized_expectation(
    lambda x: (x - posterior_mean)**2, importance_sample_size=1000)
  ```

  Alternately, the same expectations can be estimated from explicit (unweighted)
  samples. Note that sampling may be expensive because it performs resampling
  internally. For example, to produce `sample_size` samples requires first
  proposing values of shape `[sample_size, importance_sample_size]`
  (`[1000, 100]` in the code below) and then resampling down to `[sample_size]`,
  throwing most of the proposals away. For this reason you should prefer calling
  `self_normalized_expectation` over naive sampling to compute expectations.

  ```python
  posterior_samples = approximate_posterior.sample(1000)
  posterior_mean_inefficient = tf.reduce_mean(posterior_samples)
  posterior_variance_inefficient = tf.math.reduce_variance(posterior_samples)

  # Calling `self_normalized_expectation` allows for a much lower `sample_size`
  # because it uses the full set of `importance_sample_size` proposal samples to
  # approximate the expectation at each of the `sample_size` Monte Carlo
  # evaluations.
  posterior_mean_efficient = approximate_posterior.self_normalized_expectation(
    lambda x: x, sample_size=10)
  posterior_variance_efficient = (
    approximate_posterior.self_normalized_expectation(
      lambda x: (x - posterior_mean_efficient)**2, sample_size=10))
  ```

  The posterior (log-)density cannot be computed directly, but may be
  stochastically approximated. The `prob` and `log_prob` methods accept
  arguments `seed` and `sample_size` to control the variance of the
  approximation.

  ```python
  # Plot the posterior density.
  from matplotlib import pylab as plt
  xs = tf.linspace(-3., 3., 101)
  probs = approximate_posterior.prob(xs, sample_size=10, seed=(42, 42))
  plt.plot(xs, probs)
  ```

  #### Connections to importance-weighted variational inference

  Optimizing an importance-weighted variational bound provides a natural
  approach to choose a proposal distribution for importance sampling.
  Importance-weighted bounds are available directly in TFP via the
  `importance_sample_size` argument to `tfp.vi.monte_carlo_variational_loss`
  and `tfp.vi.fit_surrogate_posterior`. For example, we might improve on the
  example above by replacing the fixed proposal distribution with a learned
  proposal:

  ```python
  proposal_distribution = tfp.experimental.util.make_trainable(tfd.Normal)
  importance_sample_size = 100
  importance_weighted_losses = tfp.vi.fit_surrogate_posterior(
    target_log_prob_fn,
    surrogate_posterior=proposal_distribution,
    optimizer=tf.optimizers.Adam(0.1),
    num_steps=200,
    importance_sample_size=importance_sample_size)
  approximate_posterior = tfed.ImportanceResample(
    proposal_distribution=proposal_distribution,
    target_log_prob_fn=target_log_prob_fn,
    importance_sample_size=importance_sample_size)
  ```

  Note that although the importance-resampled `approximate_posterior` serves
  ultimately as the surrogate posterior, only the bare proposal distribution
  is passed as the `surrogate_posterior` argument to `fit_surrogate_posterior`.
  This is because the `importance_sample_size` argument tells
  `fit_surrogate_posterior` to compute an importance-weighted bound directly
  from the proposal distribution. Mathematically, it would be equivalent to omit
  the `importance_sample_size` argument and instead pass an `ImportanceResample`
  distribution as the surrogate posterior:

  ```python
  equivalent_but_less_efficient_losses = tfp.vi.fit_surrogate_posterior(
    target_log_prob_fn,
    surrogate_posterior=tfed.ImportanceResample(
      proposal_distribution=proposal_distribution,
      target_log_prob_fn=target_log_prob_fn,
      importance_sample_size=importance_sample_size),
    optimizer=tf.optimizers.Adam(0.1),
    num_steps=200)
  ```

  but this approach is not recommended, because it performs redundant
  evaluations of the `target_log_prob_fn` compared to the direct bound shown
  above.

  #### References

  [1] Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov. Importance Weighted
      Autoencoders. In _International Conference on Learning
      Representations_, 2016. https://arxiv.org/abs/1509.00519
  [2] Chris Cremer, Quaid Morris, and David Duvenaud. Reinterpreting
      Importance-Weighted Autoencoders. In _International Conference on Learning
      Representations_, Workshop track, 2017. https://arxiv.org/abs/1704.02916

  """

  def __init__(self,
               proposal_distribution,
               target_log_prob_fn,
               importance_sample_size,
               sample_size=1,
               stochastic_approximation_seed=None,
               validate_args=False,
               name=None):
    """Initialize an importance-resampled distribution.

    Args:
      proposal_distribution: Instance of `tfd.Distribution` used to generate
        proposals. This may be a joint distribution.
      target_log_prob_fn: Python `callable` representation of a (potentially
        unnormalized) target log-density. This should accept samples from the
        proposal, i.e.,
        `lp = target_log_prob_fn(proposal_distribution.sample())`.
      importance_sample_size: integer `Tensor` number of proposals used in
        the distribution of a single sample. Larger values better
        approximate the target distribution, at the cost of increased
        computation and memory usage.
      sample_size: integer `Tensor` number of Monte Carlo samples used
        to reduce variance in stochastic methods such as `log_prob`, `prob`,
        and `self_normalized_expectation`. Note that increasing
        `importance_sample_size` leads to a more accurate approximation of the
        target distribution (reducing bias and variance), while increasing
        `sample_size` improves the precision of estimates under the intermediate
        distribution corresponding to a particular finite
        `importance_sample_size` (i.e., it reduces variance only and does not
        affect the sampling distribution). If unsure, it's generally safe to
        leave `sample_size` at its default value of `1` and focus on increasing
        `importance_sample_size` instead.
        Default value: `1`.
      stochastic_approximation_seed: optional PRNG key used in stochastic
        approximations for methods such as `log_prob`, `prob`,
        and `self_normalized_expectation`. This seed does not affect sampling.
        Default value: `None`.
      validate_args: Python `bool`.  Whether to validate input with asserts.
        If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
        Default value: `False`.
      name: Python `str` name for this distribution. If `None`, defaults to
        'importance_resample'.
        Default value: `None`.
    """
    with tf.name_scope(name or 'importance_resample') as name:
      self._proposal_distribution = proposal_distribution
      self._target_log_prob_fn = target_log_prob_fn
      self._importance_sample_size = ps.convert_to_shape_tensor(
          importance_sample_size, dtype=tf.int32)
      self._sample_size = ps.convert_to_shape_tensor(
          sample_size, dtype=tf.int32)
      self._stochastic_approximation_seed = stochastic_approximation_seed

      super(ImportanceResample, self).__init__(
          dtype=proposal_distribution.dtype,
          reparameterization_type=proposal_distribution.reparameterization_type,
          allow_nan_stats=proposal_distribution.allow_nan_stats,
          validate_args=validate_args,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        proposal_distribution=parameter_properties.BatchedComponentProperties(),
        importance_sample_size=parameter_properties.ShapeParameterProperties(),
        sample_size=parameter_properties.ShapeParameterProperties(),
        stochastic_approximation_seed=parameter_properties.ParameterProperties(
            event_ndims=None))

  @property
  def proposal_distribution(self):
    return self._proposal_distribution

  @property
  def importance_sample_size(self):
    return self._importance_sample_size

  @property
  def sample_size(self):
    return self._sample_size

  @property
  def stochastic_approximation_seed(self):
    return self._stochastic_approximation_seed

  @property
  def target_log_prob_fn(self):
    return self._target_log_prob_fn

  def _event_shape(self):
    return self.proposal_distribution.event_shape

  def _event_shape_tensor(self):
    return self.proposal_distribution.event_shape_tensor()

  def _call_target_log_prob_fn(self, x):
    return nest_util.call_fn(self.target_log_prob_fn, x)

  def _get_importance_sample_size(self, importance_sample_size):
    if importance_sample_size is None:
      importance_sample_size = self.importance_sample_size
    if importance_sample_size is None:
      raise ValueError(
          'Required argument `importance_sample_size` was not specified.')
    return importance_sample_size

  def _get_sample_size(self, sample_size):
    if sample_size is None:
      sample_size = self.sample_size
    if sample_size is None:
      raise ValueError(
          'Required argument `sample_size` was not specified.')
    return sample_size

  def _get_stochastic_approximation_seed(self, stochastic_approximation_seed):
    if stochastic_approximation_seed is None:
      stochastic_approximation_seed = self.stochastic_approximation_seed
    return stochastic_approximation_seed

  def _check_weights_shape(self, log_weights, sample_shape):
    assertions = []
    message = ('Shape of importance weights does not match the batch shape '
               'of `self.proposal_distribution`. This implies that '
               'the proposal is not producing independent samples for some '
               'batch dimension(s) expected by `self.target_log_prob_fn`.')
    sample_and_batch_shape = ps.concat(
        [sample_shape, self.proposal_distribution.batch_shape_tensor()], axis=0)
    sample_and_batch_shape_ = tf.get_static_value(sample_and_batch_shape)
    if (sample_and_batch_shape_ is not None and not
        tensorshape_util.is_compatible_with(log_weights.shape,
                                            sample_and_batch_shape_)):
      raise ValueError(
          message + ' Saw: weights shape {} vs proposal sample and batch '
          'shape {}.'.format(log_weights.shape, sample_and_batch_shape))
    elif self.validate_args:
      assertions += [assert_util.assert_equal(
          tf.shape(log_weights),
          sample_and_batch_shape,
          message=message)]
    return assertions

  def _propose_with_log_weights(self, sample_shape, seed=None):
    particles, proposal_log_prob = (
        self.proposal_distribution.experimental_sample_and_log_prob(
            sample_shape, seed=seed))
    log_weights = self._call_target_log_prob_fn(particles) - proposal_log_prob
    # Ensure that the target fn didn't add batch dimensions.
    with tf.control_dependencies(
        self._check_weights_shape(log_weights, sample_shape=sample_shape)):
      log_weights = tf.identity(log_weights)

    return particles, log_weights

  def _resample(self, particles, log_weights, seed=None):
    """Chooses one out of `importance_sample_size` many weighted proposals."""
    sampled_indices = categorical.Categorical(
        logits=distribution_util.move_dimension(log_weights, 0, -1)).sample(
            sample_shape=[1], seed=seed)
    return tf.nest.map_structure(
        lambda x: (  # pylint: disable=g-long-lambda
            mcmc_util.index_remapping_gather(
                x, sampled_indices, axis=0)[0, ...]),
        particles)

  @distribution_util.AppendDocstring(
      additional_note="""The density of an importance-resampled distribution is
not generally available in closed form. This method follows algorithm (2) of
Cremer et al. [2] to compute an unbiased estimate of `prob(x)`, which
corresponds by [Jensen's inequality](
https://en.wikipedia.org/wiki/Jensen%27s_inequality) to a stochastic lower
bound on `log_prob(x)`. The estimation variance decreases, and the corresponding
bound tightens, as `sample_size` increases; an infinitely large `sample_size`
would recover the true (log-)density.""",
      kwargs_dict={
          'importance_sample_size': (
              'optional integer `Tensor` number of proposals used to define '
              'the distribution. If not specified, defaults to '
              '`self.importance_sample_size`.'),
          'sample_size': (
              'int `Tensor` number of samples used to reduce variance in the '
              'estimated density for a given `importance_sample_size`. If '
              '`None`, defaults to `self.sample_size`.'),
          'seed': 'PRNG seed; see `tfp.random.sanitize_seed` for details.'
      })
  def _log_prob(self,
                x,
                importance_sample_size=None,
                sample_size=None,
                seed=None):
    importance_sample_size = self._get_importance_sample_size(
        importance_sample_size)
    sample_size = self._get_sample_size(sample_size)
    seed = self._get_stochastic_approximation_seed(seed)

    # To have sampled `x`, we would have needed to propose `x` (along with
    # some other values) and then selected `x` from among the proposals.
    # Taking `k = importance_sample_size` and denoting the `k - 1` unchosen
    # proposal samples as `x_2, ..., x_k` with weights `w_i = p(x_i) / q(x_i)`,
    # we can compute:
    #   prob(x) = integral wrt x_2, ..., x_k of
    #              (q(x, x_2, ..., x_k)  # Propose `x` in any of `k` positions.
    #               + q(x_2, x, ..., x_k)
    #               + ...
    #               + q(x_2, x_3, ..., x)) *
    #               Categorical(w_x, w_2, ..., w_k).prob(0)  # Select `x`.
    #           = integral wrt x_2, ..., x_k of
    #              k * q(x, x_2, ..., x_k) * w_x / (w_x + w_2 + ... + w_k)
    #           = E_q[w_x / mean(weights)]
    #          ~= q(x) * w_x / mean(weights)
    #               with 1 / mean(weights) from samples x_2, ..., x_k ~ q
    #           = p(x) / mean(weights)
    # as an unbiased estimate. It can be seen that this agrees with the
    # calculation presented in Algorithm 2 of Cremer et al. [2].

    # Compute importance weight(s) `w_x` of the observed value(s) `x`.
    x_log_p = self._call_target_log_prob_fn(x)
    x_log_q = self.proposal_distribution.log_prob(x)
    x_log_weight = x_log_p - x_log_q
    log_unbiased_reciprocal_mean_weight = -x_log_weight
    dtype = log_unbiased_reciprocal_mean_weight.dtype

    if tf.get_static_value(importance_sample_size) != 1:
      # Sampling `x` implies that we also proposed `importance_sample_size - 1`
      # other values that were *not* chosen during resampling.
      # Estimate the total weight of these discarded proposals using
      # `sample_size` Monte Carlo draws.
      x_sample_ndims = (ps.rank(x_log_weight) -
                        ps.rank_from_shape(self.batch_shape_tensor()))
      _, log_weights_of_proposals_not_chosen = self._propose_with_log_weights(
          sample_shape=ps.concat(
              [
                  [sample_size],
                  [importance_sample_size - 1],
                  ps.ones([x_sample_ndims], dtype=tf.int32),
              ], axis=0),
          seed=seed)
      log_total_weight_not_chosen = tf.reduce_logsumexp(
          log_weights_of_proposals_not_chosen,
          axis=1)

      broadcast_weight_shape = ps.broadcast_shape(
          ps.shape(x_log_weight),
          ps.shape(log_total_weight_not_chosen))
      broadcast_x_log_weight = tf.broadcast_to(x_log_weight,
                                               broadcast_weight_shape)
      broadcast_log_weights_of_proposals_not_chosen = tf.broadcast_to(
          log_total_weight_not_chosen, broadcast_weight_shape)

      # Average importance weight of `x` with the other unchosen samples.
      # `mean(weights) = (w_x + w_2 + ..., w_k) / k`
      log_mean_weight = tf.reduce_logsumexp(
          [
              broadcast_x_log_weight,
              broadcast_log_weights_of_proposals_not_chosen
          ], axis=0) - tf.math.log(
              tf.cast(importance_sample_size, dtype=dtype))

      # Average over `sample_size` many unbiased estimates of the
      # random term `1 / mean(weights)`, i.e.,
      # `mean(1 / mean(weights) for _ in range(sample_size))`.
      log_unbiased_reciprocal_mean_weight = tf.reduce_logsumexp(
          -log_mean_weight, axis=0) - tf.math.log(
              tf.cast(sample_size, dtype=dtype))

    return x_log_p + log_unbiased_reciprocal_mean_weight

  @distribution_util.AppendDocstring(
      kwargs_dict={
          'importance_sample_size':
              'optional integer `Tensor` number of proposals used to define '
              'the distribution. If `None`, defaults to '
              '`self.importance_sample_size`.'
      })
  def _sample_n(self, n, importance_sample_size=None, seed=None):
    particle_seed, resample_seed = samplers.split_seed(seed, n=2)
    importance_sample_size = self._get_importance_sample_size(
        importance_sample_size)
    return self._resample(
        *self._propose_with_log_weights(
            sample_shape=[importance_sample_size, n],
            seed=particle_seed),
        seed=resample_seed)

  @distribution_util.AppendDocstring(
      additional_note="""
    Note: this method reuses the same proposal samples `z[k]` for both sampling
    and approximate `log_prob` evaluation. Thus, calling `sample_and_log_prob`
    is *not* equivalent to calling `sample` followed by `log_prob`, which would
    use two independent sets of proposal samples and in general return a
    different stochastic approximation to the log-density of the sampled points.

    In particular, `log_prob` returns a stochastic lower bound (which becomes
    tighter as `sample_size` increases) on the log-density of the
    importance-resampled distribution , while this method returns a
    single-sample stochastic *upper* bound. This guarantees that plugging an
    `ImportanceResample` surrogate posterior into a variational evidence lower
    bound (ELBO) preserves a valid lower bound---in fact, the IWAE bound [1]---
    which would otherwise not be the case for `log_prob` with finite values of
    `sample_size`. (This said, explicitly computing an IWAE bound via
    `tfp.vi.monte_carlo_variational_loss` is more efficient and stable than
    this implicit construction using an `ImportanceResample` surrogate, and so
    should be the preferred approach in general.)

    #### Mathematical details

    The `log_prob` estimate computed in this method is given by

    ```
    surrogate_log_prob(x) = target_log_prob_fn(x) - log(mean(weights(z)))
    ```

    where
    `weights(z)[k] = exp(target_log_prob_fn(z[k]) - proposal.log_prob(z[k]))`
    are the importance weights of the proposal samples `z[k]` from which `x` was
    selected. Since we know that we selected `x` from among these
    proposal samples, we may conclude that these samples are more likely
    to lead to us selecting `x` than would be the case for 'typical' proposal
    samples in the absence of such knowledge. The implied estimate of
    `prob(x)` is therefore biased upwards.

    The motivation for this estimate is that plugging it into the ELBO recovers
    the IWAE objective:

    ```
      ELBO = target_log_prob(x) - surrogate_log_prob(x)
               (for x ~ surrogate)
           = target_log_prob(x) - (target_log_prob(x) - log(mean(weights(z))))
               (for z[k] ~ proposal)
           = log(mean(weights(z)))
           = IWAE
    ```

    Because the IWAE objective lower-bounds the *true* ELBO
    of the importance-resampled distribution (i.e., the ELBO that we would
    compute using
    `surrogate_log_prob(x) = ImportanceResample.log_prob(x, sample_size=inf)`;
    see section 5.3 of Cremer et al. [2]),
    it follows that the quantity `surrogate_log_prob(x)` estimated here is an
    upper bound on the *true* log_prob of the importance-resampled distribution.
    """,
      kwargs_dict={
          'importance_sample_size':
              'optional integer `Tensor` number of proposals used to define '
              'the distribution. If `None`, defaults to '
              '`self.importance_sample_size`.'
      })
  def _sample_and_log_prob(self,
                           sample_shape,
                           importance_sample_size=None,
                           seed=None):
    """Re-use proposal samples for lower variance in the log-prob estimate."""
    particle_seed, resample_seed = samplers.split_seed(seed, n=2)
    importance_sample_size = self._get_importance_sample_size(
        importance_sample_size)
    sample_shape, _ = self._expand_sample_shape_to_vector(
        sample_shape, name='sample_shape')
    particles, log_weights = self._propose_with_log_weights(
        sample_shape=ps.concat([[importance_sample_size],
                                sample_shape], axis=0),
        seed=particle_seed)
    x = self._resample(particles=particles,
                       log_weights=log_weights,
                       seed=resample_seed)
    # Estimate `prob(x)` using the *same* proposal samples that were used to
    # sample `x`.
    return x, (self._call_target_log_prob_fn(x) -
               tf.reduce_logsumexp(log_weights, axis=0) +
               tf.math.log(tf.cast(importance_sample_size,
                                   dtype=log_weights.dtype)))

  def self_normalized_expectation(self,
                                  fn,
                                  importance_sample_size=None,
                                  sample_size=None,
                                  seed=None,
                                  name='self_normalized_expectation'):
    """Approximates the expectation of fn(x).

    This function applies self-normalized importance sampling with the given
    proposal distribution to approximate expectations under the target
    distribution. By using all of the `importance_sample_size` proposal
    samples to approximate the expectation, this will in general give
    lower-variance estimates than those obtained by explicit sampling
    (`tf.reduce_sum(fn(self.sample(sample_size)), axis=0)`), since the latter
    returns only one point from each set of `importance_sample_size` proposals.

    Concretely, this function draws `importance_sample_size` samples
    `x[1], x[2], ...` from
    `self.proposal_distribution`, computes their importance weights
    `w[k] = target_log_prob_fn(x[k]) / proposal_distribution.log_prob(x[k])`,
    and returns the weighted sum
    `sum(w[k]/sum(w) * fn(x[k]) for k in range(importance_sample_size))`. If
    `sample_size > 1` is specified, the previous procedure is performed
    multiple times and the results averaged to reduce variance.

    Note: to approximate expectations under the target distribution you should
    prefer to increase `importance_sample_size` (which reduces
    both bias and variance) rather than `sample_size` (which reduces variance
    only). Values of `sample_size > 1` are needed only if you specifically want
    expectations under the intermediate distribution that arises from
    considering a particular finite number of importance samples.

    Args:
      fn: Python `callable` that takes samples from `self.proposal_distribution`
        and returns a (structure of) `Tensor` value(s). This may represent a
        prediction derived from a posterior sample, or even a simple statistic;
        for example, the expectation of `fn = lambda x: x` is the posterior
        mean.
      importance_sample_size: int `Tensor` number of samples used to define the
        distribution under which the expectation is taken. If `None`, defaults
        to `self.importance_sample_size`.
        Default value: `None`.
      sample_size: int `Tensor` number of samples used to reduce variance in the
        expectation for a given `importance_sample_size`. If `None`, defaults
        to `self.sample_size`.
        Default value: `None`.
      seed: PRNG seed; see `tfp.random.sanitize_seed` for details. If `None`,
        defaults to `self.stochastic_approximation_seed`.
        Default value: `None`.
      name: Python string name for ops created by this function.
        Default value: `self_normalized_expectation`.
    Returns:
      expected_value: (structure of) `Tensor` value(s) estimate of the
        expectation of `fn(x)` under the target distribution.
    """
    with tf.name_scope(name):
      importance_sample_size = self._get_importance_sample_size(
          importance_sample_size)
      sample_size = self._get_sample_size(sample_size)
      seed = self._get_stochastic_approximation_seed(seed)
      particles, log_weights = self._propose_with_log_weights(
          [sample_size, importance_sample_size], seed=seed)

      weighted_reduce_sum = _make_weighted_reduce_sum(
          weights=tf.nn.softmax(log_weights, axis=1))
      return tf.nest.map_structure(
          # Average over the `sample_size` axis to reduce variance.
          lambda x: tf.reduce_mean(weighted_reduce_sum(x, axis=1), axis=0),
          fn(particles))


def _make_weighted_reduce_sum(weights):
  """Builds a function to compute weighted sums."""

  def weighted_reduce_sum(x, axis=0):
    """Weighted sum over an axis of `x`."""
    # Extend the weights to broadcast over any event dimensions of `x`.
    # This assumes that `weights` and `x` have the same sample and batch
    # dimensions, e.g., that they come from the same `sample_and_log_prob` call.
    event_ndims = ps.rank(x) - ps.rank(weights)
    aligned_weights = tf.reshape(weights,
                                 ps.concat([ps.shape(weights),
                                            ps.ones([event_ndims],
                                                    dtype=tf.int32)],
                                           axis=0))
    return tf.reduce_sum(aligned_weights * tf.cast(x, weights.dtype), axis=axis)

  return weighted_reduce_sum
