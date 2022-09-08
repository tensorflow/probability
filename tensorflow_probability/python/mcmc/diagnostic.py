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
"""Utilities for Markov Chain Monte Carlo (MCMC) sampling.

@@effective_sample_size
@@potential_scale_reduction
"""

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import stats
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import nest_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'effective_sample_size',
    'potential_scale_reduction',
]


def effective_sample_size(states,
                          filter_threshold=0.,
                          filter_beyond_lag=None,
                          filter_beyond_positive_pairs=False,
                          cross_chain_dims=None,
                          validate_args=False,
                          name=None):
  """Estimate a lower bound on effective sample size for each independent chain.

  Roughly speaking, "effective sample size" (ESS) is the size of an iid sample
  with the same variance as `state`.

  More precisely, given a stationary sequence of possibly correlated random
  variables `X_1, X_2, ..., X_N`, identically distributed, ESS is the
  number such that

  ```
  Variance{ N**-1 * Sum{X_i} } = ESS**-1 * Variance{ X_1 }.
  ```

  If the sequence is uncorrelated, `ESS = N`.  If the sequence is positively
  auto-correlated, `ESS` will be less than `N`. If there are negative
  correlations, then `ESS` can exceed `N`.

  Some math shows that, with `R_k` the auto-correlation sequence,
  `R_k := Covariance{X_1, X_{1+k}} / Variance{X_1}`, we have

  ```
  ESS(N) =  N / [ 1 + 2 * ( (N - 1) / N * R_1 + ... + 1 / N * R_{N-1}  ) ]
  ```

  This function estimates the above by first estimating the auto-correlation.
  Since `R_k` must be estimated using only `N - k` samples, it becomes
  progressively noisier for larger `k`.  For this reason, the summation over
  `R_k` should be truncated at some number `filter_beyond_lag < N`. This
  function provides two methods to perform this truncation.

  * `filter_threshold` -- since many MCMC methods generate chains where `R_k >
    0`, a reasonable criterion is to truncate at the first index where the
    estimated auto-correlation becomes negative. This method does not estimate
    the `ESS` of super-efficient chains (where `ESS > N`) correctly.

  * `filter_beyond_positive_pairs` -- reversible MCMC chains produce
    an auto-correlation sequence with the property that pairwise sums of the
    elements of that sequence are positive [Geyer][1], i.e.
    `R_{2k} + R_{2k + 1} > 0` for `k in {0, ..., N/2}`. Deviations are only
    possible due to noise. This method truncates the auto-correlation sequence
    where the pairwise sums become non-positive.

  The arguments `filter_beyond_lag`, `filter_threshold` and
  `filter_beyond_positive_pairs` are filters intended to remove noisy tail terms
  from `R_k`.  You can combine `filter_beyond_lag` with `filter_threshold` or
  `filter_beyond_positive_pairs. E.g., combining `filter_beyond_lag` and
  `filter_beyond_positive_pairs` means that terms are removed if they were to be
  filtered under the `filter_beyond_lag` OR `filter_beyond_positive_pairs`
  criteria.

  This function can also compute cross-chain ESS following
  [Vehtari et al. (2021)][2] by specifying the `cross_chain_dims` argument.
  Cross-chain ESS takes into account the cross-chain variance to reduce the ESS
  in cases where the chains are not mixing well. In general, this will be a
  smaller number than computing the ESS for individual chains and then summing
  them. In an extreme case where the chains have fallen into K non-mixing modes,
  this function will return ESS ~ K. Even when chains are mixing well it is
  still preferrable to compute cross-chain ESS via this method because it will
  reduce the noise in the estimate of `R_k`, reducing the need for truncation.

  Args:
    states: `Tensor` or Python structure of `Tensor` objects.  Dimension zero
      should index identically distributed states.
    filter_threshold: `Tensor` or Python structure of `Tensor` objects.  Must
      broadcast with `state`.  The sequence of auto-correlations is truncated
      after the first appearance of a term less than `filter_threshold`.
      Setting to `None` means we use no threshold filter.  Since `|R_k| <= 1`,
      setting to any number less than `-1` has the same effect. Ignored if
      `filter_beyond_positive_pairs` is `True`.
    filter_beyond_lag: `Tensor` or Python structure of `Tensor` objects.  Must
      be `int`-like and scalar valued.  The sequence of auto-correlations is
      truncated to this length.  Setting to `None` means we do not filter based
      on the size of lags.
    filter_beyond_positive_pairs: Python boolean. If `True`, only consider the
      initial auto-correlation sequence where the pairwise sums are positive.
    cross_chain_dims: An integer `Tensor` or a structure of integer `Tensors`
      corresponding to each state component. If a list of `states` is provided,
      then this argument should also be a list of the same length. Which
      dimensions of `states` to treat as independent chains that ESS will be
      summed over.  If `None`, no summation is performed. Note this requires at
      least 2 chains.
    validate_args: Whether to add runtime checks of argument validity. If False,
      and arguments are incorrect, correct behavior is not guaranteed.
    name:  `String` name to prepend to created ops.

  Returns:
    ess: `Tensor` structure parallel to `states`.  The effective sample size of
      each component of `states`.  If `cross_chain_dims` is None, the shape will
      be `states.shape[1:]`. Otherwise, the shape is `tf.reduce_mean(states,
      cross_chain_dims).shape[1:]`.

  Raises:
    ValueError: If `states` and `filter_threshold` or `states` and
      `filter_beyond_lag` are both structures of different shapes.
    ValueError: If `cross_chain_dims` is not `None` and there are less than 2
      chains.

  #### Examples

  We use ESS to estimate standard error.

  ```
  import tensorflow as tf
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  target = tfd.MultivariateNormalDiag(scale_diag=[1., 2.])

  # Get 1000 states from one chain.
  states = tfp.mcmc.sample_chain(
      num_burnin_steps=200,
      num_results=1000,
      current_state=tf.constant([0., 0.]),
      trace_fn=None,
      kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target.log_prob,
        step_size=0.05,
        num_leapfrog_steps=20))
  print(states.shape)
  ==> (1000, 2)

  ess = effective_sample_size(states, filter_beyond_positive_pairs=True)
  print(ess.shape)
  ==> (2,)

  mean, variance = tf.nn.moments(states, axes=0)
  standard_error = tf.sqrt(variance / ess)
  ```

  #### References

  [1]: Charles J. Geyer, Practical Markov chain Monte Carlo (with discussion).
       Statistical Science, 7:473-511, 1992.

  [2]: Aki Vehtari, Andrew Gelman, Daniel Simpson, Bob Carpenter, Paul-Christian
       Bürkner. Rank-normalization, folding, and localization: An improved R-hat
       for assessing convergence of MCMC, 2021. Bayesian analysis,
       16(2):667-718.
  """
  if cross_chain_dims is None:
    cross_chain_dims = nest_util.broadcast_structure(states, None)
  filter_beyond_lag = nest_util.broadcast_structure(states, filter_beyond_lag)
  filter_threshold = nest_util.broadcast_structure(states, filter_threshold)
  filter_beyond_positive_pairs = nest_util.broadcast_structure(
      states, filter_beyond_positive_pairs)

  # Process items, one at a time.
  def single_state(*args):
    return _effective_sample_size_single_state(
        *args, validate_args=validate_args)
  with tf.name_scope('effective_sample_size' if name is None else name):
    return nest.map_structure_up_to(
        states,
        single_state,
        states, filter_beyond_lag, filter_threshold,
        filter_beyond_positive_pairs, cross_chain_dims)


def _effective_sample_size_single_state(states, filter_beyond_lag,
                                        filter_threshold,
                                        filter_beyond_positive_pairs,
                                        cross_chain_dims,
                                        validate_args):
  """ESS computation for one single Tensor argument."""

  with tf.name_scope('effective_sample_size_single_state'):

    states = tf.convert_to_tensor(states, name='states')
    dt = states.dtype

    # filter_beyond_lag == None ==> auto_corr is the full sequence.
    auto_cov = stats.auto_correlation(
        states, axis=0, max_lags=filter_beyond_lag, normalize=False)
    n = _axis_size(states, axis=0)

    if cross_chain_dims is not None:
      num_chains = _axis_size(states, cross_chain_dims)
      num_chains_ = tf.get_static_value(num_chains)

      assertions = []
      msg = ('When `cross_chain_dims` is not `None`, there must be > 1 chain '
             'in `states`.')
      if num_chains_ is not None:
        if num_chains_ < 2:
          raise ValueError(msg)
      elif validate_args:
        assertions.append(
            assert_util.assert_greater(num_chains, 1., message=msg))

      with tf.control_dependencies(assertions):
        # We're computing the R[k] from equation 10 of Vehtari et al.
        # (2021):
        #
        # R[k] := 1 - (W - 1/C * Sum_{c=1}^C s_c**2 R[k, c]) / (var^+),
        #
        # where:
        #   C := number of chains
        #   N := length of chains
        #   x_hat[c] := 1 / N Sum_{n=1}^N x[n, c], chain mean.
        #   x_hat := 1 / C Sum_{c=1}^C x_hat[c], overall mean.
        #   W := 1/C Sum_{c=1}^C s_c**2, within-chain variance.
        #   B := N / (C - 1) Sum_{c=1}^C (x_hat[c] - x_hat)**2, between chain
        #     variance.
        #   s_c**2 := 1 / (N - 1) Sum_{n=1}^N (x[n, c] - x_hat[c])**2, chain
        #       variance
        #   R[k, m] := auto_corr[k, m, ...], auto-correlation indexed by chain.
        #   var^+ := (N - 1) / N * W + B / N

        cross_chain_dims = ps.non_negative_axis(
            cross_chain_dims, ps.rank(states))
        # B / N
        between_chain_variance_div_n = _reduce_variance(
            tf.reduce_mean(states, axis=0),
            biased=False,  # This makes the denominator be C - 1.
            axis=cross_chain_dims - 1)
        # W * (N - 1) / N
        biased_within_chain_variance = tf.reduce_mean(auto_cov[0],
                                                      cross_chain_dims - 1)
        # var^+
        approx_variance = (
            biased_within_chain_variance + between_chain_variance_div_n)
        # 1/C * Sum_{c=1}^C s_c**2 R[k, c]
        mean_auto_cov = tf.reduce_mean(auto_cov, cross_chain_dims)
        auto_corr = 1. - (biased_within_chain_variance -
                          mean_auto_cov) / approx_variance
    else:
      auto_corr = auto_cov / auto_cov[:1]
      num_chains = 1

    # With R[k] := auto_corr[k, ...],
    # ESS = N / {1 + 2 * Sum_{k=1}^N R[k] * (N - k) / N}
    #     = N / {-1 + 2 * Sum_{k=0}^N R[k] * (N - k) / N} (since R[0] = 1)
    #     approx N / {-1 + 2 * Sum_{k=0}^M R[k] * (N - k) / N}
    # where M is the filter_beyond_lag truncation point chosen above.

    # Get the factor (N - k) / N, and give it shape [M, 1,...,1], having total
    # ndims the same as auto_corr
    k = tf.range(0., _axis_size(auto_corr, axis=0))
    nk_factor = (n - k) / n
    if tensorshape_util.rank(auto_corr.shape) is not None:
      new_shape = [-1] + [1] * (tensorshape_util.rank(auto_corr.shape) - 1)
    else:
      new_shape = tf.concat(
          ([-1],
           tf.ones([tf.rank(auto_corr) - 1], dtype=tf.int32)),
          axis=0)
    nk_factor = tf.reshape(nk_factor, new_shape)
    weighted_auto_corr = nk_factor * auto_corr

    if filter_beyond_positive_pairs:
      def _sum_pairs(x):
        x_len = ps.shape(x)[0]
        # For odd sequences, we drop the final value.
        x = x[:x_len - x_len % 2]
        new_shape = ps.concat([[x_len // 2, 2], ps.shape(x)[1:]], axis=0)
        return tf.reduce_sum(tf.reshape(x, new_shape), 1)

      # Pairwise sums are all positive for auto-correlation spectra derived from
      # reversible MCMC chains.
      # E.g. imagine the pairwise sums are [0.2, 0.1, -0.1, -0.2]
      # Step 1: mask = [False, False, True, True]
      mask = _sum_pairs(auto_corr) < 0.
      # Step 2: mask = [0, 0, 1, 1]
      mask = tf.cast(mask, dt)
      # Step 3: mask = [0, 0, 1, 2]
      mask = tf.cumsum(mask, axis=0)
      # Step 4: mask = [1, 1, 0, 0]
      mask = tf.maximum(1. - mask, 0.)

      # N.B. this reduces the length of weighted_auto_corr by a factor of 2.
      # It still works fine in the formula below.
      weighted_auto_corr = _sum_pairs(weighted_auto_corr) * mask
    elif filter_threshold is not None:
      filter_threshold = tf.convert_to_tensor(
          filter_threshold, dtype=dt, name='filter_threshold')
      # Get a binary mask to zero out values of auto_corr below the threshold.
      #   mask[i, ...] = 1 if auto_corr[j, ...] > threshold for all j <= i,
      #   mask[i, ...] = 0, otherwise.
      # So, along dimension zero, the mask will look like [1, 1, ..., 0, 0,...]
      # Building step by step,
      #   Assume auto_corr = [1, 0.5, 0.0, 0.3], and filter_threshold = 0.2.
      # Step 1:  mask = [False, False, True, False]
      mask = auto_corr < filter_threshold
      # Step 2:  mask = [0, 0, 1, 0]
      mask = tf.cast(mask, dtype=dt)
      # Step 3:  mask = [0, 0, 1, 1]
      mask = tf.cumsum(mask, axis=0)
      # Step 4:  mask = [1, 1, 0, 0]
      mask = tf.maximum(1. - mask, 0.)
      weighted_auto_corr *= mask

    return num_chains * n / (-1 + 2 * tf.reduce_sum(weighted_auto_corr, axis=0))


def potential_scale_reduction(chains_states,
                              independent_chain_ndims=1,
                              split_chains=False,
                              validate_args=False,
                              name=None):
  """Gelman and Rubin (1992)'s potential scale reduction for chain convergence.

  Given `N > 1` states from each of `C > 1` independent chains, the potential
  scale reduction factor, commonly referred to as R-hat, measures convergence of
  the chains (to the same target) by testing for equality of means.
  Specifically, R-hat measures the degree to which variance (of the means)
  between chains exceeds what one would expect if the chains were identically
  distributed. See [Gelman and Rubin (1992)][1]; [Brooks and Gelman (1998)][2].

  Some guidelines:

  * The initial state of the chains should be drawn from a distribution
    overdispersed with respect to the target.
  * If all chains converge to the target, then as `N --> infinity`, R-hat --> 1.
    Before that, R-hat > 1 (except in pathological cases, e.g. if the chain
    paths were identical).
  * The above holds for any number of chains `C > 1`.  Increasing `C` does
    improve effectiveness of the diagnostic.
  * Sometimes, R-hat < 1.2 is used to indicate approximate convergence, but of
    course this is problem-dependent. See [Brooks and Gelman (1998)][2].
  * R-hat only measures non-convergence of the mean. If higher moments, or
    other statistics are desired, a different diagnostic should be used. See
    [Brooks and Gelman (1998)][2].

  Args:
    chains_states:  `Tensor` or Python structure of `Tensor`s representing the
      states of a Markov Chain at each result step.  The `ith` state is
      assumed to have shape `[Ni, Ci1, Ci2,...,CiD] + A`.
      Dimension `0` indexes the `Ni > 1` result steps of the Markov Chain.
      Dimensions `1` through `D` index the `Ci1 x ... x CiD` independent
      chains to be tested for convergence to the same target.
      The remaining dimensions, `A`, can have any shape (even empty).
    independent_chain_ndims: Integer type `Tensor` with value `>= 1` giving the
      number of dimensions, from `dim = 1` to `dim = D`, holding independent
      chain results to be tested for convergence.
    split_chains: Python `bool`. If `True`, divide samples from each chain into
      first and second halves, treating these as separate chains.  This makes
      R-hat more robust to non-stationary chains, and is recommended in [3].
    validate_args: Whether to add runtime checks of argument validity. If False,
      and arguments are incorrect, correct behavior is not guaranteed.
    name: `String` name to prepend to created tf.  Default:
      `potential_scale_reduction`.

  Returns:
    `Tensor` structure parallel to `chains_states` representing the
    R-hat statistic for the state(s).  Same `dtype` as `state`, and
    shape equal to `state.shape[1 + independent_chain_ndims:]`.

  Raises:
    ValueError:  If `independent_chain_ndims < 1`.

  #### Examples

  Diagnosing convergence by monitoring 10 chains that each attempt to
  sample from a 2-variate normal.

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  target = tfd.MultivariateNormalDiag(scale_diag=[1., 2.])

  # Get 10 (2x) overdispersed initial states.
  initial_state = target.sample(10) * 2.
  ==> (10, 2)

  # Get 1000 samples from the 10 independent chains.
  chains_states = tfp.mcmc.sample_chain(
      num_burnin_steps=200,
      num_results=1000,
      current_state=initial_state,
      trace_fn=None,
      kernel=tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=target.log_prob,
          step_size=0.05,
          num_leapfrog_steps=20))
  chains_states.shape
  ==> (1000, 10, 2)

  rhat = tfp.mcmc.diagnostic.potential_scale_reduction(
      chains_states, independent_chain_ndims=1)

  # The second dimension needed a longer burn-in.
  rhat.eval()
  ==> [1.05, 1.3]
  ```

  To see why R-hat is reasonable, let `X` be a random variable drawn uniformly
  from the combined states (combined over all chains).  Then, in the limit
  `N, C --> infinity`, with `E`, `Var` denoting expectation and variance,

  ```R-hat = ( E[Var[X | chain]] + Var[E[X | chain]] ) / E[Var[X | chain]].```

  Using the law of total variance, the numerator is the variance of the combined
  states, and the denominator is the total variance minus the variance of the
  the individual chain means.  If the chains are all drawing from the same
  distribution, they will have the same mean, and thus the ratio should be one.

  #### References

  [1]: Stephen P. Brooks and Andrew Gelman. General Methods for Monitoring
       Convergence of Iterative Simulations. _Journal of Computational and
       Graphical Statistics_, 7(4), 1998.

  [2]: Andrew Gelman and Donald B. Rubin. Inference from Iterative Simulation
       Using Multiple Sequences. _Statistical Science_, 7(4):457-472, 1992.

  [3]: Aki Vehtari, Andrew Gelman, Daniel Simpson, Bob Carpenter, Paul-Christian
       Bürkner. Rank-normalization, folding, and localization: An improved R-hat
       for assessing convergence of MCMC, 2021. Bayesian analysis,
       16(2):667-718.
  """
  # tf.get_static_value returns None iff a constant value (as a numpy
  # array) is not efficiently computable.  Therefore, we try constant_value then
  # check for None.
  icn_const_ = tf.get_static_value(
      ps.convert_to_shape_tensor(independent_chain_ndims))
  if icn_const_ is not None:
    independent_chain_ndims = icn_const_
    if icn_const_ < 1:
      raise ValueError(
          'Argument `independent_chain_ndims` must be `>= 1`, found: {}'.format(
              independent_chain_ndims))

  def single_state(s):
    return _potential_scale_reduction_single_state(
        s, independent_chain_ndims, split_chains, validate_args)
  with tf.name_scope('potential_scale_reduction' if name is None else name):
    return tf.nest.map_structure(single_state, chains_states)


def _potential_scale_reduction_single_state(state, independent_chain_ndims,
                                            split_chains, validate_args):
  """potential_scale_reduction for one single state `Tensor`."""
  # casting integers to floats for floating-point division
  # check to see if the `state` is a numpy object for the numpy test suite
  if dtype_util.as_numpy_dtype(state.dtype) is np.int64:
    state = tf.cast(state, tf.float64)
  elif dtype_util.is_integer(state.dtype):
    state = tf.cast(state, tf.float32)
  with tf.name_scope('potential_scale_reduction_single_state'):
    # We assume exactly one leading dimension indexes e.g. correlated samples
    # from each Markov chain.
    state = tf.convert_to_tensor(state, name='state')

    n_samples_ = tf.compat.dimension_value(state.shape[0])
    if n_samples_ is not None:  # If available statically.
      if split_chains and n_samples_ < 4:
        raise ValueError(
            'Must provide at least 4 samples when splitting chains. '
            'Found {}'.format(n_samples_))
      if not split_chains and n_samples_ < 2:
        raise ValueError(
            'Must provide at least 2 samples.  Found {}'.format(n_samples_))
    elif validate_args:
      if split_chains:
        assertions = [assert_util.assert_greater(
            ps.shape(state)[0], 4,
            message='Must provide at least 4 samples when splitting chains.')]
        with tf.control_dependencies(assertions):
          state = tf.identity(state)
      else:
        assertions = [assert_util.assert_greater(
            ps.shape(state)[0], 2,
            message='Must provide at least 2 samples.')]
        with tf.control_dependencies(assertions):
          state = tf.identity(state)

    # Define so it's not a magic number.
    # Warning!  `if split_chains` logic assumes this is 1!
    sample_ndims = 1

    if split_chains:
      # Split the sample dimension in half, doubling the number of
      # independent chains.

      # For odd number of samples, keep all but the last sample.
      state_shape = ps.shape(state)
      n_samples = state_shape[0]
      state = state[:n_samples - n_samples % 2]

      # Suppose state = [0, 1, 2, 3, 4, 5]
      # Step 1: reshape into [[0, 1, 2], [3, 4, 5]]
      # E.g. reshape states of shape [a, b] into [2, a//2, b].
      state = tf.reshape(
          state,
          ps.concat([[2, n_samples // 2], state_shape[1:]], axis=0)
      )
      # Step 2: Put the size `2` dimension in the right place to be treated as a
      # chain, changing [[0, 1, 2], [3, 4, 5]] into [[0, 3], [1, 4], [2, 5]],
      # reshaping [2, a//2, b] into [a//2, 2, b].
      state = tf.transpose(
          a=state,
          perm=ps.concat(
              [[1, 0], ps.range(2, ps.rank(state))], axis=0))

      # We're treating the new dim as indexing 2 chains, so increment.
      independent_chain_ndims += 1

    sample_axis = ps.range(0, sample_ndims)
    chain_axis = ps.range(sample_ndims,
                          sample_ndims + independent_chain_ndims)
    sample_and_chain_axis = ps.range(
        0, sample_ndims + independent_chain_ndims)

    n = _axis_size(state, sample_axis)
    m = _axis_size(state, chain_axis)

    # In the language of Brooks and Gelman (1998),
    # B / n is the between chain variance, the variance of the chain means.
    # W is the within sequence variance, the mean of the chain variances.
    b_div_n = _reduce_variance(
        tf.reduce_mean(state, axis=sample_axis, keepdims=True),
        sample_and_chain_axis,
        biased=False)
    w = tf.reduce_mean(
        _reduce_variance(state, sample_axis, keepdims=True, biased=False),
        axis=sample_and_chain_axis)

    # sigma^2_+ is an estimate of the true variance, which would be unbiased if
    # each chain was drawn from the target.  c.f. "law of total variance."
    sigma_2_plus = ((n - 1) / n) * w + b_div_n
    return ((m + 1.) / m) * sigma_2_plus / w - (n - 1.) / (m * n)


# TODO(b/72873233) Move some variant of this to tfd.sample_stats.
def _reduce_variance(x, axis=None, biased=True, keepdims=False):
  with tf.name_scope('reduce_variance'):
    x = tf.convert_to_tensor(x, name='x')
    mean = tf.reduce_mean(x, axis=axis, keepdims=True)
    biased_var = tf.reduce_mean(
        tf.math.squared_difference(x, mean), axis=axis, keepdims=keepdims)
    if biased:
      return biased_var
    n = _axis_size(x, axis)
    return (n / (n - 1.)) * biased_var


def _axis_size(x, axis=None):
  """Get number of elements of `x` in `axis`, as type `x.dtype`."""
  if axis is None:
    return ps.cast(ps.size(x), x.dtype)
  return ps.cast(
      ps.reduce_prod(
          ps.gather(ps.shape(x), axis)), x.dtype)
