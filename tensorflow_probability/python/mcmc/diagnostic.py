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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import stats
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import prefer_static

__all__ = [
    'effective_sample_size',
    'potential_scale_reduction',
]


def effective_sample_size(states,
                          filter_threshold=0.,
                          filter_beyond_lag=None,
                          filter_beyond_positive_pairs=False,
                          name=None):
  """Estimate a lower bound on effective sample size for each independent chain.

  Roughly speaking, "effective sample size" (ESS) is the size of an iid sample
  with the same variance as `state`.

  More precisely, given a stationary sequence of possibly correlated random
  variables `X_1, X_2,...,X_N`, each identically distributed ESS is the number
  such that

  ```Variance{ N**-1 * Sum{X_i} } = ESS**-1 * Variance{ X_1 }.```

  If the sequence is uncorrelated, `ESS = N`.  If the sequence is positively
  auto-correlated, `ESS` will be less than `N`. If there are negative
  correlations, then `ESS` can exceed `N`.

  Args:
    states:  `Tensor` or list of `Tensor` objects.  Dimension zero should index
      identically distributed states.
    filter_threshold:  `Tensor` or list of `Tensor` objects.
      Must broadcast with `state`.  The auto-correlation sequence is truncated
      after the first appearance of a term less than `filter_threshold`.
      Setting to `None` means we use no threshold filter.  Since `|R_k| <= 1`,
      setting to any number less than `-1` has the same effect. Ignored if
      `filter_beyond_positive_pairs` is `True`.
    filter_beyond_lag:  `Tensor` or list of `Tensor` objects.  Must be
      `int`-like and scalar valued.  The auto-correlation sequence is truncated
      to this length.  Setting to `None` means we do not filter based on number
      of lags.
    filter_beyond_positive_pairs: Python boolean. If `True`, only consider the
      initial auto-correlation sequence where the pairwise sums are positive.
    name:  `String` name to prepend to created ops.

  Returns:
    ess:  `Tensor` or list of `Tensor` objects.  The effective sample size of
      each component of `states`.  Shape will be `states.shape[1:]`.

  Raises:
    ValueError:  If `states` and `filter_threshold` or `states` and
      `filter_beyond_lag` are both lists with different lengths.

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
      kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target.log_prob,
        step_size=0.05,
        num_leapfrog_steps=20))
  states.shape
  ==> (1000, 2)

  ess = effective_sample_size(states, filter_beyond_positive_pairs=True)
  ==> Shape (2,) Tensor

  mean, variance = tf.nn.moments(states, axis=0)
  standard_error = tf.sqrt(variance / ess)
  ```

  Some math shows that, with `R_k` the auto-correlation sequence,
  `R_k := Covariance{X_1, X_{1+k}} / Variance{X_1}`, we have

  ```ESS(N) =  N / [ 1 + 2 * ( (N - 1) / N * R_1 + ... + 1 / N * R_{N-1}  ) ]```

  This function estimates the above by first estimating the auto-correlation.
  Since `R_k` must be estimated using only `N - k` samples, it becomes
  progressively noisier for larger `k`.  For this reason, the summation over
  `R_k` should be truncated at some number `filter_beyond_lag < N`. This
  function provides two methods to perform this truncation.

  * `filter_threshold` -- since many MCMC methods generate chains where `R_k >
    0`, a reasonable criteria is to truncate at the first index where the
    estimated auto-correlation becomes negative. This method does not estimate
    the `ESS` of super-efficient chains (where `ESS > N`) correctly.

  * `filter_beyond_positive_pairs` -- reversible MCMC chains produce
    auto-correlation sequence with the property that pairwise sums of the
    elements of that sequence are positive [1] (i.e. `R_{2k} + R_{2k + 1} > 0`
    for `k in {0, ..., N/2}`). Deviations are only possible due to noise. This
    method truncates the auto-correlation sequence where the pairwise sums
    become non-positive.

  The arguments `filter_beyond_lag`, `filter_threshold` and
  `filter_beyond_positive_pairs` are filters intended to remove noisy tail terms
  from `R_k`.  You can combine `filter_beyond_lag` with `filter_threshold` or
  `filter_beyond_positive_pairs. E.g. combining `filter_beyond_lag` and
  `filter_beyond_positive_pairs` means that terms are removed if they were to be
  filtered under the `filter_beyond_lag` OR `filter_beyond_positive_pairs`
  criteria.

  #### References

  [1]: Geyer, C. J. Practical Markov chain Monte Carlo (with discussion).
       Statistical Science, 7:473-511, 1992.
  """
  states_was_list = _is_list_like(states)

  # Convert all args to lists.
  if not states_was_list:
    states = [states]

  filter_beyond_lag = _broadcast_maybelist_arg(states, filter_beyond_lag,
                                               'filter_beyond_lag')
  filter_threshold = _broadcast_maybelist_arg(states, filter_threshold,
                                              'filter_threshold')
  filter_beyond_positive_pairs = _broadcast_maybelist_arg(
      states, filter_beyond_positive_pairs, 'filter_beyond_positive_pairs')

  # Process items, one at a time.
  with tf.name_scope('effective_sample_size' if name is None else name):
    ess_list = [
        _effective_sample_size_single_state(s, fbl, ft, fbpp)  # pylint: disable=g-complex-comprehension
        for (s, fbl, ft,
             fbpp) in zip(states, filter_beyond_lag, filter_threshold,
                          filter_beyond_positive_pairs)
    ]

  if states_was_list:
    return ess_list
  return ess_list[0]


def _effective_sample_size_single_state(states, filter_beyond_lag,
                                        filter_threshold,
                                        filter_beyond_positive_pairs):
  """ESS computation for one single Tensor argument."""

  with tf.name_scope('effective_sample_size_single_state'):

    states = tf.convert_to_tensor(states, name='states')
    dt = states.dtype

    # filter_beyond_lag == None ==> auto_corr is the full sequence.
    auto_corr = stats.auto_correlation(
        states, axis=0, max_lags=filter_beyond_lag)

    # With R[k] := auto_corr[k, ...],
    # ESS = N / {1 + 2 * Sum_{k=1}^N (N - k) / N * R[k]}
    #     = N / {-1 + 2 * Sum_{k=0}^N (N - k) / N * R[k]} (since R[0] = 1)
    #     approx N / {-1 + 2 * Sum_{k=0}^M (N - k) / N * R[k]}
    # where M is the filter_beyond_lag truncation point chosen above.

    # Get the factor (N - k) / N, and give it shape [M, 1,...,1], having total
    # ndims the same as auto_corr
    n = _axis_size(states, axis=0)
    k = tf.range(0., _axis_size(auto_corr, axis=0))
    nk_factor = (n - k) / n
    if auto_corr.shape.ndims is not None:
      new_shape = [-1] + [1] * (auto_corr.shape.ndims - 1)
    else:
      new_shape = tf.concat(
          ([-1],
           tf.ones([tf.rank(auto_corr) - 1], dtype=tf.int32)),
          axis=0)
    nk_factor = tf.reshape(nk_factor, new_shape)
    weighted_auto_corr = nk_factor * auto_corr

    if filter_beyond_positive_pairs:
      def _sum_pairs(x):
        x_len = tf.shape(x)[0]
        # For odd sequences, we drop the final value.
        x = x[:x_len - x_len % 2]
        new_shape = tf.concat([[x_len // 2, 2], tf.shape(x)[1:]], axis=0)
        return tf.reduce_sum(tf.reshape(x, new_shape), 1)

      # Pairwise sums are all positive for auto-correlation spectra derived from
      # reversible MCMC chains.
      # E.g. imagine the pairwise sums are [0.2, 0.1, -0.1, -0.2]
      # Step 1: mask = [False, False, True, False]
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
      # Step 2:  mask = [0, 0, 1, 1]
      mask = tf.cast(mask, dtype=dt)
      # Step 3:  mask = [0, 0, 1, 2]
      mask = tf.cumsum(mask, axis=0)
      # Step 4:  mask = [1, 1, 0, 0]
      mask = tf.maximum(1. - mask, 0.)
      weighted_auto_corr *= mask

    return n / (-1 + 2 * tf.reduce_sum(weighted_auto_corr, axis=0))


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
    chains_states:  `Tensor` or Python `list` of `Tensor`s representing the
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
    `Tensor` or Python `list` of `Tensor`s representing the R-hat statistic for
    the state(s).  Same `dtype` as `state`, and shape equal to
    `state.shape[1 + independent_chain_ndims:]`.

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
  chains_states, _ = tfp.mcmc.sample_chain(
      num_burnin_steps=200,
      num_results=1000,
      current_state=initial_state,
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
  [3]: Vehtari et al.  Rank-normalization, folding, and localization: An
       improved Rhat for assessing convergence of MCMC.
  """
  chains_states_was_list = _is_list_like(chains_states)
  if not chains_states_was_list:
    chains_states = [chains_states]

  # tf.get_static_value returns None iff a constant value (as a numpy
  # array) is not efficiently computable.  Therefore, we try constant_value then
  # check for None.
  icn_const_ = tf.get_static_value(
      tf.convert_to_tensor(independent_chain_ndims))
  if icn_const_ is not None:
    independent_chain_ndims = icn_const_
    if icn_const_ < 1:
      raise ValueError(
          'Argument `independent_chain_ndims` must be `>= 1`, found: {}'.format(
              independent_chain_ndims))

  with tf.name_scope('potential_scale_reduction' if name is None else name):
    rhat_list = [
        _potential_scale_reduction_single_state(s, independent_chain_ndims,
                                                split_chains, validate_args)
        for s in chains_states
    ]

  if chains_states_was_list:
    return rhat_list
  return rhat_list[0]


def _potential_scale_reduction_single_state(state, independent_chain_ndims,
                                            split_chains, validate_args):
  """potential_scale_reduction for one single state `Tensor`."""
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
        state = distribution_util.with_dependencies([
            tf1.assert_greater(
                tf.shape(state)[0], 4,
                message='Must provide at least 4 samples when splitting chains.'
            )], state)
      else:
        state = distribution_util.with_dependencies([
            tf1.assert_greater(
                tf.shape(state)[0], 2,
                message='Must provide at least 2 samples.')], state)

    # Define so it's not a magic number.
    # Warning!  `if split_chains` logic assumes this is 1!
    sample_ndims = 1

    if split_chains:
      # Split the sample dimension in half, doubling the number of
      # independent chains.

      # For odd number of samples, keep all but the last sample.
      state_shape = prefer_static.shape(state)
      n_samples = state_shape[0]
      state = state[:n_samples - n_samples % 2]

      # Suppose state = [0, 1, 2, 3, 4, 5]
      # Step 1: reshape into [[0, 1, 2], [3, 4, 5]]
      # E.g. reshape states of shape [a, b] into [2, a//2, b].
      state = tf.reshape(
          state,
          prefer_static.concat([[2, n_samples // 2], state_shape[1:]], axis=0)
      )
      # Step 2: Put the size `2` dimension in the right place to be treated as a
      # chain, changing [[0, 1, 2], [3, 4, 5]] into [[0, 3], [1, 4], [2, 5]],
      # reshaping [2, a//2, b] into [a//2, 2, b].
      state = tf.transpose(
          a=state,
          perm=prefer_static.concat(
              [[1, 0], tf.range(2, tf.rank(state))], axis=0))

      # We're treating the new dim as indexing 2 chains, so increment.
      independent_chain_ndims += 1

    sample_axis = tf.range(0, sample_ndims)
    chain_axis = tf.range(sample_ndims,
                          sample_ndims + independent_chain_ndims)
    sample_and_chain_axis = tf.range(
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
        _reduce_variance(state, sample_axis, keepdims=True, biased=True),
        axis=sample_and_chain_axis)

    # sigma^2_+ is an estimate of the true variance, which would be unbiased if
    # each chain was drawn from the target.  c.f. "law of total variance."
    sigma_2_plus = w + b_div_n

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
    return tf.cast(tf.size(x), x.dtype)
  return tf.cast(tf.reduce_prod(tf.gather(tf.shape(x), axis)), x.dtype)


def _is_list_like(x):
  """Helper which returns `True` if input is `list`-like."""
  return isinstance(x, (tuple, list))


def _broadcast_maybelist_arg(states, secondary_arg, name):
  """Broadcast a listable secondary_arg to that of states."""
  if _is_list_like(secondary_arg):
    if len(secondary_arg) != len(states):
      raise ValueError('Argument `%s` was a list of different length ({}) than '
                       '`states` ({})'.format(name, len(states)))
  else:
    secondary_arg = [secondary_arg] * len(states)

  return secondary_arg
