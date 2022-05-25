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
"""Utilities for setting tolerances for stochastic TFP tests."""

import collections

import numpy as np
import scipy.optimize as optimize
import scipy.stats as stats

__all__ = [
    'binomial_confidence_interval',
    'brief_report',
    'full_report',
]


# Confidence interval one particular error rate that is justified by a
# _BootstrapResult.
_BootstrapConfidenceInterval = collections.namedtuple(
    '_BootstrapConfidenceInterval',
    ['low_p', 'high_p'])


class _BootstrapResult(collections.namedtuple(
    '_BootstrapResult',
    ['failures', 'trials'])):
  """Result of bootstrapping test success or failure."""
  __slots__ = ()

  def for_error_rate(self, error_rate):
    """Confidence interval at the given `error_rate` on the true p(fail)."""
    low_p, high_p = binomial_confidence_interval(
        self.failures, self.trials, error_rate)
    return _BootstrapConfidenceInterval(low_p, high_p)


# Suggested new test parameters to achieve a desired p(fail) for the
# test, assuming Gaussianity.
_GaussianOneErrorRateResult = collections.namedtuple(
    '_GaussianOneErrorRateResult',
    ['new_atol',
     'new_rtol',
     'out_of_bounds',
     'too_tight',
     'too_broad',
     'samples_factor',])


class _GaussianResult(collections.namedtuple(
    '_GaussianResult',
    ['loc',
     'scale',
     'k_s_limit',
     'too_low_p',
     'too_high_p',
     'expected',
     'lb',
     'ub',])):
  """Result of fitting a Gaussian for mean(empirical distribution)."""
  __slots__ = ()

  def for_error_rate(self, error_rate):
    """Specific recommendations to achieve the given `error_rate`."""
    dist = stats.norm(loc=self.loc, scale=self.scale)
    # Predict a reasonable tolerance
    if self.too_high_p > self.too_low_p:
      # Assume the upper limit is binding
      new_ub = dist.isf(error_rate / 2.0)
      new_atol = new_ub - self.expected
    else:
      # Assume the lower limit is binding
      new_lb = dist.ppf(error_rate / 2.0)
      new_atol = self.expected - new_lb
    new_rtol = new_atol / np.abs(self.expected)

    # Predict a reasonable number of samples
    if self.loc >= self.ub or self.loc <= self.lb:
      # No increase in the sample size will make this pass, unless it
      # also shifts the mean, which we cannot predict.
      return _GaussianOneErrorRateResult(
          new_atol, new_rtol, out_of_bounds=True,
          too_tight=False, too_broad=False, samples_factor=None)
    else:
      # Given that loc is in bounds, this function is monotonically
      # increasing in the scale factor (because the too_low_p and
      # too_high_p terms are individually increasing toward 0.5).
      def error_rate_good(scale_factor):
        dist = stats.norm(loc=self.loc, scale=self.scale * scale_factor)
        too_low_p = dist.cdf(self.lb)
        too_high_p = dist.sf(self.ub)
        return too_low_p + too_high_p - error_rate
      if error_rate_good(1000.) < 0.:
        # Even a million times fewer samples do not risk exiting the tolerances.
        return _GaussianOneErrorRateResult(
            new_atol, new_rtol, out_of_bounds=False,
            too_tight=True, too_broad=False, samples_factor=None)
      elif error_rate_good(0.001) > 0.:
        # Even a huge number of samples predicts a bad error rate.
        return _GaussianOneErrorRateResult(
            new_atol, new_rtol, out_of_bounds=False,
            too_tight=False, too_broad=True, samples_factor=None)
      else:
        scale_factor = optimize.brentq(error_rate_good, 0.001, 1000., rtol=1e-9)
        return _GaussianOneErrorRateResult(
            new_atol, new_rtol, out_of_bounds=False,
            too_tight=False, too_broad=False,
            samples_factor=scale_factor ** -2.)


# All results for one batch member, suitable for rendering.
_Result = collections.namedtuple(
    '_Result',
    ['batch_indices',
     'reduction_size',
     'expected',
     'tolerance',
     'bootstrap',
     'gaussian',])


# TODO(cgs): Test this independently of its use in
# distributions/internal/correlation_matrix_volumes_lib
def binomial_confidence_interval(successes, trials, error_rate):
  """Computes a confidence interval on the true p of a binomial.

  Assumes:
  - The given `successes` count outcomes of an iid Bernoulli trial
    with unknown probability p, that was repeated `trials` times.

  Guarantees:
  - The probability (over the randomness of drawing the given sample)
    that the true p is outside the returned interval is no more than
    the given `error_rate`.

  Args:
    successes: Python or numpy `int` number of successes.
    trials: Python or numpy `int` number of trials.
    error_rate: Python `float` admissible rate of mistakes.

  Returns:
    low_p: Lower bound of confidence interval.
    high_p: Upper bound of confidence interval.

  Raises:
    ValueError: If scipy is not available.

  """
  def p_small_enough(p):
    # This is positive iff p is smaller than the desired upper bound.
    log_prob = stats.binom.logcdf(successes, trials, p)
    return log_prob - np.log(error_rate / 2.)
  def p_big_enough(p):
    # This is positive iff p is larger than the desired lower bound.
    # Scipy's survival function for discrete random variables excludes
    # the argument, but I want it included for this purpose.
    log_prob = stats.binom.logsf(successes-1, trials, p)
    return log_prob - np.log(error_rate / 2.)
  if successes < trials:
    high_p = optimize.brentq(
        p_small_enough, successes / float(trials), 1., rtol=1e-9)
  else:
    high_p = 1.
  if successes > 0:
    low_p = optimize.brentq(
        p_big_enough, 0., successes / float(trials), rtol=1e-9)
  else:
    low_p = 0.
  return low_p, high_p


def _choose_num_bootstraps(sample_size, mean_size, fuel):
  """Choose how many bootstraps to do.

  The fuel is the total number floats we get to draw from
  `np.random.choice`, which is a proxy for the amount of time we're
  allowed to spend bootstrapping.  We choose how many bootstraps to
  do to make sure we fit in our budget.

  Args:
    sample_size: Size of sample we are bootstrapping from.
    mean_size: Number of samples reduced to form each mean estimate.
    fuel: Total number of floats chosen in the bootstrap, as a proxy
      for total work spent bootstrapping.

  Returns:
    num_bootstraps: Number of bootstraps to do.
  """
  # We ignore the sample size here because the asymptotics of
  # np.random.choice should be O(n log n + k) where n is the input
  # size and k is the output size, and the fuel limit is only binding
  # in the regime where k >> n.
  num_bootstraps = int(fuel / mean_size)
  # However, we always have at least as many bootstraps as we've
  # already drawn samples, because in that limit we can just slice the
  # existing array.
  if num_bootstraps * mean_size <= sample_size:
    num_bootstraps = int(sample_size / mean_size)
  return num_bootstraps


def _bootstrap_means(samples, mean_size, fuel):
  """Compute bootstrapped means."""
  num_bootstraps = _choose_num_bootstraps(len(samples), mean_size, fuel)
  if num_bootstraps * mean_size <= len(samples):
    # Inputs are huge relative to fuel; fake a bootstrap by just slicing
    # the input array
    return np.mean(np.reshape(samples, newshape=(-1, mean_size)), axis=-1)
  # Compute this in batches to never materialize an over-large
  # intermediate array.
  n_batches = 10
  if n_batches > num_bootstraps:
    n_batches = num_bootstraps
  batches = []
  for _ in range(n_batches):
    batch_size = int(num_bootstraps / n_batches)
    batch = np.mean(
        np.random.choice(samples, size=(batch_size, mean_size), replace=True),
        axis=-1)
    batches.append(batch)
  return np.concatenate(batches, axis=0)


def _evaluate_bootstrap(means, lb, ub):
  in_bounds = (means > lb) & (means < ub)
  trials = len(in_bounds)
  successes = np.count_nonzero(in_bounds)
  failures = trials - successes
  return _BootstrapResult(failures, trials)


def _fit_gaussian(samples, mean_size):
  """Fit a Gaussian to represent the mean of `mean_size` of the `samples`."""
  loc = np.mean(samples)
  scale_one = np.sqrt(np.mean((samples - loc)**2))
  scale = scale_one / np.sqrt(mean_size)
  rho = np.mean(np.abs(samples - loc) ** 3)
  # Upper bound from Wikipedia
  # https://en.wikipedia.org/wiki/Berry%E2%80%93Esseen_theorem
  berry_esseen_c = 0.4748
  # From the Berry-Esseen theorem
  k_s_limit = berry_esseen_c * rho / (np.sqrt(mean_size) * (scale_one ** 3))
  return loc, scale, k_s_limit


def _evaluate_assuming_gaussian(loc, scale, k_s_limit, expected, lb, ub):
  dist = stats.norm(loc=loc, scale=scale)
  too_low_p = dist.cdf(lb)
  too_high_p = dist.sf(ub)
  return _GaussianResult(loc, scale, k_s_limit, too_low_p, too_high_p,
                         expected, lb, ub)


def _evaluate_one_batch_member(to_reduce, expected, reduction_size, tolerance):
  """As `_evaluate_means_assertion` but for one batch member."""
  lb = expected - tolerance
  ub = expected + tolerance

  # This is our budget of draws from np.random.choice.  Chosen to
  # avoid the bootstraps being much more expensive than the code under
  # test.
  fuel = 30000000
  bootstrapped_means = _bootstrap_means(to_reduce, reduction_size, fuel)
  bootstrap_result = _evaluate_bootstrap(bootstrapped_means, lb, ub)
  loc, scale, k_s_limit = _fit_gaussian(to_reduce, reduction_size)
  gauss_result = _evaluate_assuming_gaussian(
      loc, scale, k_s_limit, expected, lb, ub)
  return bootstrap_result, gauss_result


def _evaluate_means_assertion(to_reduce, expected, axis, atol, rtol):
  """Evaluates `assertAllMeansClose` assertion quality.

  The results of the evaluation are packed into a list of `_Result`
  objects (one per batch member), which can then be post-processed for
  presentation depending on the medium.

  Args:
    to_reduce: Numpy array of samples, presumed IID along `axis`.
      Other dimensions are taken to be batch dimensions.
    expected: Numpy array of expected mean values.  Must broadcast
      with the reduction of `to_reduce` along `axis`.
    axis: Python `int` giving the reduction axis.
    atol: Python `float`, absolute tolerance for the means.
    rtol: Python `float`, relative tolerance for the means.

  Returns:
    results: A Python list of `_Result` objects, one for each batch member.
  """
  assert isinstance(axis, int), 'TODO(cgs): Handle multiple reduction axes'

  # Normalize so we correctly handle negative axes below.
  axis = axis if axis >= 0 else axis + len(to_reduce.shape)

  # Compute reduction size
  reduction_size = to_reduce.shape[axis]

  # Pull the axis to be reduced to the front, so we can iterate over
  # the others
  permutation = np.concatenate([
      [axis],
      np.arange(0, axis, dtype=np.int32),
      np.arange(axis + 1, len(to_reduce.shape), dtype=np.int32),
  ], axis=0)
  to_reduce = np.transpose(to_reduce, permutation)

  # Broadcast everything.  The subtlety is that we protect the top
  # axis of to_reduce, because that's now the reduction axis.  For
  # that we need to broadcast underneath it.
  batch_example = np.zeros(to_reduce.shape[1:], dtype=to_reduce.dtype)
  for array in [expected, atol, rtol]:
    batch_example = batch_example + np.zeros_like(
        array, dtype=batch_example.dtype)
  # Rank-expand to_reduce, in case the batch had more dimensions.
  # Doing this explicitly to keep the reduction axis on top and to
  # avoid broadcasting it.
  while len(to_reduce.shape) <= len(batch_example.shape):
    to_reduce = to_reduce[:, np.newaxis, ...]

  to_reduce = to_reduce + batch_example
  expected = expected + batch_example
  atol = atol + batch_example
  rtol = rtol + batch_example

  # Iterate over batch members
  results = []
  for indices in np.ndindex(to_reduce.shape[1:]):
    sub_to_reduce = to_reduce
    sub_expected = expected
    sub_atol = atol
    sub_rtol = rtol

    for index in indices:
      sub_to_reduce = sub_to_reduce[:, index, ...]
      sub_expected = sub_expected[index, ...]
      sub_atol = sub_atol[index, ...]
      sub_rtol = sub_rtol[index, ...]

    tolerance = sub_atol + sub_rtol * np.abs(sub_expected)
    bootstrap, gaussian = _evaluate_one_batch_member(
        sub_to_reduce, sub_expected, reduction_size, tolerance)
    results.append(_Result(
        indices, reduction_size, sub_expected, tolerance, bootstrap, gaussian))

  return results


def _format_bootstrap_report(result):
  """Formats a `_BootstrapResult` as a complete report str.

  The information presented includes:
  - The results of a bootstrap to test how often the test fails
  - A confidence interval on the probability of the
    `assertAllMeansClose` failing, assuming the empirical distribution
    is close to the true distribution

  This differs conceptually from the `_gaussian_report` because it
  doesn't assume that the distribution on means Gaussianizes, so
  remains valid in a low-sample setting.  On the other hand, when
  trustworthy, `_gaussian_report` is more informative.

  Args:
    result: A `_BootstrapResult` capturing the fit.

  Returns:
    report: A Python  `str` suitable for being printed or added to an
      assertion message.

  """
  report = (
      f'\n{result.failures} of {result.trials} bootstrapped trials fail.'
      '\nAssuming that the empirical distribution is the true distribution:')
  for rate in [1e-3, 1e-9]:
    one_rate = result.for_error_rate(rate)
    report += (
        f'\n- With confidence 1 - {rate}, p(fail) >= {one_rate.low_p:.3g}.'
        f'\n- With confidence 1 - {rate}, p(fail) <= {one_rate.high_p:.3g}.')
  return report


def _format_gaussian_report(result):
  """Formats a `_GaussianResult` as a complete report str.

  The information presented includes:
  - The parameters of the Gaussian fit
  - The extrapolated probability of the `assertAllMeansClose` failing,
    assuming the distribution of means is Gaussian
  - Suggested changes to the tolerance and, if possible, number of
    samples that should bring the failure rate to a desired point

  Args:
    result: A `_GaussianResult` capturing the fit.

  Returns:
    report: A Python  `str` suitable for being printed or added to an
      assertion message.
  """
  report = '\nAssuming also that the true distribution on means is Gaussian:'
  report += f'\n- Mean ~ N(loc={result.loc:.3g}, scale={result.scale:.3g}).'
  p_fail = result.too_low_p + result.too_high_p
  report += f'\n- p(fail) = {p_fail:.3g}.'
  for error_rate in [1e-3, 1e-9]:
    one_rate = result.for_error_rate(error_rate)
    if p_fail > error_rate:
      report += f'\n- To lower to {error_rate}, try '
    else:
      report += f'\n- To raise to {error_rate}, try '
    report += f'atol {one_rate.new_atol:.3g} or rtol {one_rate.new_rtol:.3g}'

    if one_rate.out_of_bounds:
      # No increase in the sample size will make this pass, unless it
      # also shifts the mean, which we cannot predict.
      report += '.'
    else:
      if one_rate.too_tight:
        # Even a million times fewer samples do not risk exiting the tolerances.
        report += '; the Gaussian is too tight on the scale of the tolerances.'
      elif one_rate.too_broad:
        # Even a huge number of samples predicts a bad error rate.
        report += '; the Gaussian is too broad on the scale of the tolerances.'
      else:
        report += (
            f', or {one_rate.samples_factor:.3g} times the samples.')
  return report


def _format_full_report(results):
  """Formats the given list of `_Result`s as a complete report str.

  This version is more prolix than `_format_brief_report`, but does not
  suppress any information, and does not suffer from any statistical
  bias.

  Args:
    results: A list of `_Result` for a batch `assertAllMeansClose`
      evaluation.

  Returns:
    report: A Python `str` suitable for being printed or added to an
      assertion message.
  """
  report = ''
  for result in results:
    report += f'\n\nAt index {result.batch_indices}'
    report += f'\nExpected mean({result.reduction_size} draws)'
    report += f' in {result.expected:.3g} +- {result.tolerance:.3g};'
    report += _format_bootstrap_report(result.bootstrap)
    report += _format_gaussian_report(result.gaussian)

  return report


def _format_brief_report_one(result):
  """Formats the given `Result` as a condensed report str."""
  report = f'\nExpected mean({result.reduction_size} draws)'
  report += f' in {result.expected:.3g} +- {result.tolerance:.3g};'
  report += f' got mean ~ N(loc={result.gaussian.loc:.3g}, scale={result.gaussian.scale:.3g}),'
  p_fail = result.gaussian.too_low_p + result.gaussian.too_high_p
  report += f' p(fail) = {p_fail:.3g}.'
  for target_rate in [0.001, 1e-9]:
    for_rate = result.gaussian.for_error_rate(target_rate)
    if p_fail > target_rate:
      report += f'\nTo lower to {target_rate},'
    else:
      report += f'\nTo raise to {target_rate},'
    report += f' try atol {for_rate.new_atol:.3g} or rtol {for_rate.new_rtol:.3g}'
    if for_rate.out_of_bounds or for_rate.too_tight or for_rate.too_broad:
      report += '.'
    else:
      report += f', or {for_rate.samples_factor:.3g} times the samples.'
  return report


def _format_brief_report(results):
  """Formats the gives list of `_Result`s as a condensed report str.

  Part of the condensing is that `brief_report` only describes the
  most likely to fail batch member.  This introduces some statistical
  skew: the samples being bootstrapped from are not for the true
  distribution of the batch member under test, but from the
  distribution conditioned on that batch member having turned to be
  the worst in the batch.  This may grow problematic for a large batch
  of `assertAllMeansClose`, where multiple members have a
  non-negligible true probability of failure.  If that source of error
  is not worth the brevity, there's always `full_report`.

  Args:
    results: A list of `_Result` for a batch `assertAllMeansClose`
      evaluation.

  Returns:
    report: A Python `str` suitable for being printed or added to an
      assertion message.
  """
  worst = results[0]
  report = ''
  if len(results) > 1:
    for result in results:
      p_fail = result.gaussian.too_low_p + result.gaussian.too_high_p
      if p_fail > worst.gaussian.too_low_p + worst.gaussian.too_high_p:
        worst = result
    report += f'At index {worst.batch_indices}'
  return report + _format_brief_report_one(worst)


def brief_report(to_reduce, expected, axis, atol, rtol):
  """Evaluates `assertAllMeansClose` assertion quality and reports briefly.

  Specifically, uses a Gaussian approximation to estimate
  - The probability of the assertion failing
  - How to change the parameters to control the failure probability

  The evaluation assumes that the elements of `to_reduce` are IID
  along the given `axis`, and that the empirical distribution they
  represent is a good approximation of the true one-element
  distribution.  The analysis also assumes that the reduction axis is
  large enough that the distribution on the computed mean is well
  approximated as a Gaussian.

  The brief report only describes the most likely to fail batch
  member.  This introduces some statistical skew: the samples being
  bootstrapped from are not for the true distribution of the batch
  member under test, but from the distribution conditioned on that
  batch member having turned to be the worst in the batch.  This may
  grow problematic for a large batch of `assertAllMeansClose`, where
  multiple members have a non-negligible true probability of failure.
  If that source of error is not worth the brevity, there's always
  `full_report`.

  Args:
    to_reduce: Numpy array of samples, presumed IID along `axis`.
      Other dimensions are taken to be batch dimensions.
    expected: Numpy array of expected mean values.  Must broadcast
      with the reduction of `to_reduce` along `axis`.
    axis: Python `int` giving the reduction axis.
    atol: Python `float`, absolute tolerance for the means.
    rtol: Python `float`, relative tolerance for the means.

  Returns:
    report: A Python `str` suitable for being printed or added to an
      assertion message.
  """
  result = _evaluate_means_assertion(to_reduce, expected, axis, atol, rtol)
  return _format_brief_report(result)


def full_report(to_reduce, expected, axis, atol, rtol):
  """Evaluates `assertAllMeansClose` assertion quality and reports.

  This version is more prolix than `brief_report`, but does not
  suppress any information, and does not suffer from any statistical
  bias.

  Specifically, uses both a bootstrap and a Gaussian approximation to
  estimate
  - The probability of the assertion failing
  - How to change the parameters to control the failure probability

  The difference between the bootstrap and the Gaussian is that the
  bootstrap does not assume the distribution on means Gaussianizes, so
  remains more trustworthy in low-sample settings.

  The evaluation assumes that the elements of `to_reduce` are IID
  along the given `axis`, and that the empirical distribution they
  represent is a good approximation of the true one-element
  distribution.  The Gaussian analysis also assumes that the reduction
  axis is large enough that the distribution on the computed mean is
  well approximated as a Gaussian.

  The full report describes all members of a batch.

  Args:
    to_reduce: Numpy array of samples, presumed IID along `axis`.
      Other dimensions are taken to be batch dimensions.
    expected: Numpy array of expected mean values.  Must broadcast
      with the reduction of `to_reduce` along `axis`.
    axis: Python `int` giving the reduction axis.
    atol: Python `float`, absolute tolerance for the means.
    rtol: Python `float`, relative tolerance for the means.

  Returns:
    report: A Python `str` suitable for being printed or added to an
      assertion message.
  """
  result = _evaluate_means_assertion(to_reduce, expected, axis, atol, rtol)
  return _format_full_report(result)
