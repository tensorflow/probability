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
"""Statistical test assertions calibrated for their error rates.

Statistical tests have an inescapable probability of error: a correct
sampler can still fail a test by chance, and an incorrect sampler can
still pass a test by chance.  This library is about bounding both of
those error rates.  This requires admitting a task-specific notion of
"discrepancy": Correct code will fail rarely, code that misbehaves by
more than the discrepancy will pass rarely, and nothing reliable can
be said about code that misbehaves, but misbehaves by less than the
discrepancy.

# Example

Consider testing that the mean of a scalar probability distribution P
is some expected constant.  Suppose the support of P is the interval
`[0, 1]`.  Then you might do this:

```python
  from tensorflow_probability.python.distributions.internal import statistical_testing

  expected_mean = ...
  num_samples = 5000
  samples = ... draw 5000 samples from P

  # Check that the mean looks right
  check1 = statistical_testing.assert_true_mean_equal_by_dkwm(
      samples, low=0., high=1., expected=expected_mean,
      false_fail_rate=1e-6)

  # Check that the difference in means detectable with 5000 samples is
  # small enough
  check2 = tf.compat.v1.assert_less(
      statistical_testing.min_discrepancy_of_true_means_detectable_by_dkwm(
          num_samples, low=0., high=1.0,
          false_fail_rate=1e-6, false_pass_rate=1e-6),
      0.01)

  # Be sure to execute both assertion ops
  sess.run([check1, check2])
```

The second assertion is an instance of experiment design.  It's a
deterministic computation (independent of the code under test) that
checks that `5000` samples is enough to reliably resolve mean
differences of `0.01` or more.  Here "reliably" means that if the code
under test is correct, the probability of drawing an unlucky sample
that causes this test to fail is at most 1e-6; and if the code under
test is incorrect enough that its true mean is 0.01 more or less than
expected, then the probability of drawing a "lucky" sample that causes
the test to false-pass is also at most 1e-6.

# Overview

Every function in this library can be characterized in terms of:

- The property being tested, such as the full density of the
  distribution under test, or just its true mean, or a single
  Bernoulli probability, etc.

- The relation being asserted, e.g., whether the mean is less, more,
  or equal to the given expected value.

- The stochastic bound being relied upon, such as the
  [Dvoretzky-Kiefer-Wolfowitz-Massart inequality]
  (https://en.wikipedia.org/wiki/CDF-based_nonparametric_confidence_interval)
  or the CDF of the binomial distribution (for assertions about
  Bernoulli probabilities).

- The number of sample sets in the statistical test.  For example,
  testing equality of means has a one-sample variant, where the
  expected mean is given exactly, and a two-sample variant, where the
  expected mean is itself given by a set of samples (e.g., from an
  alternative algorithm).

- What operation(s) of the test are to be performed.  Each test has
  three of these:

  1. `assert` executes the test.  Specifically, it creates a TF op that
     produces an error if it has enough evidence to prove that the
     property under test is violated.  These functions depend on the
     desired false failure rate, because that determines the sizes of
     appropriate confidence intervals, etc.

  2. `min_discrepancy` computes the smallest difference reliably
     detectable by that test, given the sample count and error rates.
     What it's a difference of is test-specific.  For example, a test
     for equality of means would make detection guarantees about the
     difference of the true means.

  3. `min_num_samples` computes the minimum number of samples needed
     to reliably detect a given discrepancy with given error rates.

  The latter two are for experimental design, and are meant to be
  usable either interactively or inline in the overall test method.

This library follows a naming convention, to make room for every
combination of the above.  A name mentions the operation first, then
the property, then the relation, then the bound, then, if the test
takes more than one set of samples, a token indicating this.  For
example, `assert_true_mean_equal_by_dkwm` (which is implicitly
one-sample).  Each name is a grammatically sound noun phrase (or verb
phrase, for the asserts).

# Asymptotic properties

The number of samples needed tends to scale as `O(1/discrepancy**2)` and
as `O(log(1/error_rate))`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import itertools

import tensorflow as tf
from tensorflow_probability.python.distributions import seed_stream
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util

__all__ = [
    'assert_true_cdf_equal_by_dkwm',
    'min_discrepancy_of_true_cdfs_detectable_by_dkwm',
    'min_num_samples_for_dkwm_cdf_test',
    'kolmogorov_smirnov_distance',
    'kolmogorov_smirnov_distance_two_sample',
    'empirical_cdfs',
    'assert_true_cdf_equal_by_dkwm_two_sample',
    'min_discrepancy_of_true_cdfs_detectable_by_dkwm_two_sample',
    'min_num_samples_for_dkwm_cdf_two_sample_test',
    'true_mean_confidence_interval_by_dkwm',
    'assert_true_mean_equal_by_dkwm',
    'min_discrepancy_of_true_means_detectable_by_dkwm',
    'min_num_samples_for_dkwm_mean_test',
    'assert_true_mean_in_interval_by_dkwm',
    'assert_true_mean_equal_by_dkwm_two_sample',
    'min_discrepancy_of_true_means_detectable_by_dkwm_two_sample',
    'min_num_samples_for_dkwm_mean_two_sample_test',
    'assert_multivariate_true_cdf_equal_on_projections_two_sample',
]


def assert_true_cdf_equal_by_dkwm(
    samples, cdf, left_continuous_cdf=None, false_fail_rate=1e-6, name=None):
  """Asserts the full CDF of the given distribution is as expected.

  More precisely, fails if there is enough evidence (using the
  [Dvoretzky-Kiefer-Wolfowitz-Massart inequality]
  (https://en.wikipedia.org/wiki/CDF-based_nonparametric_confidence_interval))
  that the true CDF of some distribution from which the given samples are
  drawn is _not_ the given expected CDF with statistical significance
  `false_fail_rate` or stronger, otherwise passes.  If you also want to
  check that you are gathering enough evidence that a pass is not
  spurious, see `min_num_samples_for_dkwm_cdf_test` and
  `min_discrepancy_of_true_cdfs_detectable_by_dkwm`.

  If the distribution in question has atoms (e.g., is discrete), computing this
  test requires CDF values for both sides of the discontinuity.  In this case,
  the `cdf` argument is assumed to compute the CDF inclusive of the atom, i.e.,
  cdf(x) = Pr(X <= x).  The user must also supply the `left_continuous_cdf`,
  which must compute the cdf exclusive of the atom, i.e., left_continuous_cdf(x)
  = Pr(X < x).  Invariant: cdf(x) - left_continuous_cdf(x) = pmf(x).

  For example, the two required cdfs of the degenerate distribution that places
  all the mass at 0 can be given as
  ```
  cdf=lambda x: tf.where(x < 0, 0., 1.)
  left_continuous_cdf=lambda x: tf.where(x <= 0, 0., 1.)
  ```

  Note that `false_fail_rate` is a total false failure rate for all
  the assertions in the batch.  As such, if the batch is nontrivial,
  the assertion will insist on stronger evidence to fail any one member.

  Args:
    samples: Tensor of shape [n] + B.  Samples from some (batch of) scalar-event
      distribution(s) of interest, giving a (batch of) empirical CDF(s).
      Assumed IID across the 0 dimension.
    cdf: Analytic cdf inclusive of any atoms, as a function that can compute CDF
      values in batch.  Must accept a Tensor of shape B + [n] and the same dtype
      as `samples` and return a Tensor of shape B + [n] of CDF values.  For each
      sample x, `cdf(x) = Pr(X <= x)`.
    left_continuous_cdf: Analytic left-continuous cdf, as a function that can
      compute CDF values in batch.  Must accept a Tensor of shape B + [n] and
      the same dtype as `samples` and return a Tensor of shape B + [n] of CDF
      values.  For each sample x, `left_continuous_cdf(x) = Pr(X < x)`.  If the
      distribution under test has no atoms (i.e., the CDF is continuous), this
      is redundant and may be omitted.  Conversely, if this argument is omitted,
      the test assumes the distribution is atom-free.
    false_fail_rate: *Scalar* floating-point `Tensor` admissible total
      rate of mistakes.
    name: A name for this operation (optional).

  Returns:
    check: Op that raises `InvalidArgumentError` if any expected CDF is
      outside the corresponding confidence envelope.
  """
  with tf.compat.v1.name_scope(name, 'assert_true_cdf_equal_by_dkwm',
                               [samples, false_fail_rate]):
    dtype = dtype_util.common_dtype([samples, false_fail_rate], tf.float32)
    samples = tf.convert_to_tensor(value=samples, name='samples', dtype=dtype)
    false_fail_rate = tf.convert_to_tensor(
        value=false_fail_rate, name='false_fail_rate', dtype=dtype)
    tf.compat.v1.assert_scalar(false_fail_rate)  # Static shape
    itemwise_false_fail_rate = _itemwise_error_rate(
        total_rate=false_fail_rate,
        param_tensors=[], samples_tensor=samples)
    n = tf.shape(input=samples)[0]
    envelope = _dkwm_cdf_envelope(n, itemwise_false_fail_rate)
    distance = kolmogorov_smirnov_distance(samples, cdf, left_continuous_cdf)
    return tf.compat.v1.assert_less_equal(
        distance, envelope, message='Empirical CDF outside K-S envelope')


def min_discrepancy_of_true_cdfs_detectable_by_dkwm(
    n, false_fail_rate, false_pass_rate, name=None):
  """Returns the minimum CDF discrepancy that a DKWM-based test can detect.

  DKWM is the [Dvoretzky-Kiefer-Wolfowitz-Massart inequality]
  (https://en.wikipedia.org/wiki/CDF-based_nonparametric_confidence_interval).

  Note that `false_fail_rate` is a total false failure rate for all
  the tests in the batch.  As such, if the batch is nontrivial, each
  member will demand more samples.  The `false_pass_rate` is also
  interpreted as a total, but is treated asymmetrically: If each test
  in the batch detects its corresponding discrepancy with probability
  at least `1 - false_pass_rate`, then running all those tests and
  failing if any one fails will jointly detect all those discrepancies
  with the same `false_pass_rate`.

  Args:
    n: `Tensor` of numbers of samples to be drawn from the distributions
      of interest.
    false_fail_rate: *Scalar* floating-point `Tensor` admissible total
      rate of false failures.
    false_pass_rate: *Scalar* floating-point `Tensor` admissible rate
      of false passes.
    name: A name for this operation (optional).

  Returns:
    discr: `Tensor` of lower bounds on the K-S distances between true
       CDFs detectable by a DKWM-based test.

  For each batch member `i`, of `K` total, drawing `n[i]` samples from some
  scalar distribution is enough to detect a K-S distance in CDFs of size
  `discr[i]` or more.  Specifically, we guarantee that (a) if the true CDF is
  the expected CDF, then `assert_true_cdf_equal_by_dkwm` will fail with
  probability at most `false_fail_rate / K` (which amounts to `false_fail_rate`
  if applied to the whole batch at once), and (b) if the true CDF differs from
  the expected CDF by at least `discr[i]`, `assert_true_cdf_equal_by_dkwm` will
  pass with probability at most `false_pass_rate`.

  The detectable discrepancy scales as

  - `O(1 / sqrt(n[i]))`,
  - `O(-log(false_fail_rate/K))`, and
  - `O(-log(false_pass_rate))`.
  """
  with tf.compat.v1.name_scope(
      name, 'min_discrepancy_of_true_cdfs_detectable_by_dkwm',
      [n, false_fail_rate, false_pass_rate]):
    dtype = dtype_util.common_dtype(
        [n, false_fail_rate, false_pass_rate], tf.float32)
    n = tf.convert_to_tensor(value=n, name='n', dtype=dtype)
    false_fail_rate = tf.convert_to_tensor(
        value=false_fail_rate, name='false_fail_rate', dtype=dtype)
    false_pass_rate = tf.convert_to_tensor(
        value=false_pass_rate, name='false_pass_rate', dtype=dtype)
    # Algorithm: Assume a true CDF F.  The DKWM inequality gives a
    # stochastic bound on how far the observed empirical CDF F_n can be.
    # Then, using the DKWM inequality again gives a stochastic bound on
    # the farthest candidate true CDF F' that
    # true_mean_confidence_interval_by_dkwm might consider.  At worst, these
    # errors may go in the same direction, so the distance between F and
    # F' is bounded by the sum.
    # On batching: false fail rates sum, so I need to reduce
    # the input to account for the batching.  False pass rates
    # max, so I don't.
    sampling_envelope = _dkwm_cdf_envelope(n, false_pass_rate)
    itemwise_false_fail_rate = _itemwise_error_rate(
        total_rate=false_fail_rate, param_tensors=[n])
    analysis_envelope = _dkwm_cdf_envelope(n, itemwise_false_fail_rate)
    return sampling_envelope + analysis_envelope


def min_num_samples_for_dkwm_cdf_test(
    discrepancy, false_fail_rate=1e-6, false_pass_rate=1e-6, name=None):
  """Returns how many samples suffice for a one-sample DKWM CDF test.

  To wit, returns an upper bound on the number of samples necessary to
  guarantee detecting a K-S distance of CDFs of at least the given
  `discrepancy`, with the given `false_fail_rate` and `false_pass_rate`,
  using the [Dvoretzky-Kiefer-Wolfowitz-Massart inequality]
  (https://en.wikipedia.org/wiki/CDF-based_nonparametric_confidence_interval)
  on a scalar distribution.

  Args:
    discrepancy: Floating-point `Tensor` of desired upper limits on K-S
      distances that may go undetected with probability higher than
      `1 - false_pass_rate`.
    false_fail_rate: *Scalar* floating-point `Tensor` admissible total
      rate of false failures.
    false_pass_rate: *Scalar* floating-point `Tensor` admissible rate
      of false passes.
    name: A name for this operation (optional).

  Returns:
    n: `Tensor` of numbers of samples to be drawn from the distributions
      of interest.

  For each batch member `i`, of `K` total, drawing `n[i]` samples from some
  scalar distribution is enough to detect a K-S distribution of CDFs of size
  `discrepancy[i]` or more.  Specifically, we guarantee that (a) if the true CDF
  is the expected CDF, then `assert_true_cdf_equal_by_dkwm` will fail with
  probability at most `false_fail_rate / K` (which amounts to `false_fail_rate`
  if applied to the whole batch at once), and (b) if the true CDF differs from
  the expected CDF by at least `discrepancy[i]`, `assert_true_cdf_equal_by_dkwm`
  will pass with probability at most `false_pass_rate`.

  The required number of samples scales as

  - `O(-log(false_fail_rate/K))`,
  - `O(-log(false_pass_rate))`, and
  - `O(1 / discrepancy[i]**2)`.
  """
  with tf.compat.v1.name_scope(name, 'min_num_samples_for_dkwm_cdf_test',
                               [false_fail_rate, false_pass_rate, discrepancy]):
    dtype = dtype_util.common_dtype(
        [false_fail_rate, false_pass_rate, discrepancy], tf.float32)
    discrepancy = tf.convert_to_tensor(
        value=discrepancy, name='discrepancy', dtype=dtype)
    false_fail_rate = tf.convert_to_tensor(
        value=false_fail_rate, name='false_fail_rate', dtype=dtype)
    false_pass_rate = tf.convert_to_tensor(
        value=false_pass_rate, name='false_pass_rate', dtype=dtype)
    # Could choose to cleverly allocate envelopes, but this is sound.
    envelope1 = discrepancy / 2.
    envelope2 = envelope1
    itemwise_false_fail_rate = _itemwise_error_rate(
        total_rate=false_fail_rate, param_tensors=[discrepancy])
    n1 = -tf.math.log(itemwise_false_fail_rate / 2.) / (2. * envelope1**2)
    n2 = -tf.math.log(false_pass_rate / 2.) / (2. * envelope2**2)
    return tf.maximum(n1, n2)


def kolmogorov_smirnov_distance(
    samples, cdf, left_continuous_cdf=None, name=None):
  """Computes the Kolmogorov-Smirnov distance between the given CDFs.

  The (absolute) Kolmogorov-Smirnov distance is the maximum (absolute)
  discrepancy between the CDFs, i.e.,

    sup_x(|cdf1(x) - cdf2(x)|)

  This is tractable to compute exactly when at least one CDF in question is an
  empirical CDF given by samples, because the analytic one need only be queried
  at the sampled values.

  If the distribution in question has atoms (e.g., is discrete), computing the
  distance requires CDF values for both sides of the discontinuity.  In this
  case, the `cdf` argument is assumed to compute the CDF inclusive of the atom,
  i.e., cdf(x) = Pr(X <= x).  The user must also supply the
  `left_continuous_cdf`, which must compute the cdf exclusive of the atom, i.e.,
  left_continuous_cdf(x) = Pr(X < x).

  For example, the two required cdfs of the degenerate distribution that places
  all the mass at 0 can be given as
  ```
  cdf=lambda x: tf.where(x < 0, 0., 1.)
  left_continuous_cdf=lambda x: tf.where(x <= 0, 0., 1.)
  ```

  Args:
    samples: Tensor of shape [n] + B.  Samples from some (batch of) scalar-event
      distribution(s) of interest, giving a (batch of) empirical CDF(s).
      Assumed IID across the 0 dimension.
    cdf: Analytic cdf inclusive of any atoms, as a function that can compute CDF
      values in batch.  Must accept a Tensor of shape B + [n] and the same dtype
      as `samples` and return a Tensor of shape B + [n] of CDF values.  For each
      sample x, `cdf(x) = Pr(X <= x)`.
    left_continuous_cdf: Analytic left-continuous cdf, as a function that can
      compute CDF values in batch.  Must accept a Tensor of shape B + [n] and
      the same dtype as `samples` and return a Tensor of shape B + [n] of CDF
      values.  For each sample x, `left_continuous_cdf(x) = Pr(X < x)`.  If the
      distribution under test has no atoms (i.e., the CDF is continuous), this
      is redundant and may be omitted.  Conversely, if this argument is omitted,
      the test assumes the distribution is atom-free.
    name: A name for this operation (optional).

  Returns:
    distance: Tensor of shape B: (Absolute) Kolmogorov-Smirnov distance between
      the empirical and analytic CDFs.
  """
  with tf.compat.v1.name_scope(name, 'kolmogorov_smirnov_distance', [samples]):
    dtype = dtype_util.common_dtype([samples], tf.float32)
    samples = tf.convert_to_tensor(value=samples, name='samples', dtype=dtype)
    samples = _move_dim_and_sort(samples)

    # Compute analytic cdf values at each sample
    cdfs = cdf(samples)
    if left_continuous_cdf is None:
      left_continuous_cdfs = cdfs
    else:
      left_continuous_cdfs = left_continuous_cdf(samples)

    # Compute per-batch-member empirical cdf values at each sample
    # If any samples within a batch member are repeated, some of the entries
    # will be wrong:
    # - In low_empirical_cdfs, the first sample in a run of equal samples will
    #   have the correct cdf value, and the others will be too high; and
    # - In high_empirical_cdfs, the last sample in a run of equal samples will
    #   have the correct cdf value, and the others will be too low.
    # However, this is OK, because those errors do not change the maximums.
    # Could defensively use `empirical_cdfs` here, but those rely on the
    # relatively more expensive `searchsorted` operation.
    n = tf.cast(tf.shape(input=samples)[-1], dtype=cdfs.dtype)
    low_empirical_cdfs = tf.range(n, dtype=cdfs.dtype) / n
    high_empirical_cdfs = tf.range(1, n+1, dtype=cdfs.dtype) / n

    # Compute per-batch K-S distances on either side of each discontinuity in
    # the empirical CDF.  I only need one-sided comparisons in both cases,
    # because the empirical CDF is piecewise constant and the true CDF is
    # monotonic: The maximum of F(x) - F_n(x) occurs just before a
    # discontinuity, and the maximum of F_n(x) - F(x) occurs just after.
    low_distances = tf.reduce_max(
        input_tensor=left_continuous_cdfs - low_empirical_cdfs, axis=-1)
    high_distances = tf.reduce_max(
        input_tensor=high_empirical_cdfs - cdfs, axis=-1)
    return tf.maximum(low_distances, high_distances)


def kolmogorov_smirnov_distance_two_sample(samples1, samples2, name=None):
  """Computes the Kolmogorov-Smirnov distance between the given empirical CDFs.

  The (absolute) Kolmogorov-Smirnov distance is the maximum (absolute)
  discrepancy between the CDFs, i.e.,

    sup_x(|cdf1(x) - cdf2(x)|)

  This is tractable to compute exactly for empirical CDFs, because they are
  piecewise constant with known piece boundaries (the samples).

  This function works even if the samples have duplicates (e.g., if the
  underlying distribution is discrete).

  Args:
    samples1: Tensor of shape [n] + B.  Samples from some (batch of)
      scalar-event distribution(s) of interest, giving a (batch of) empirical
      CDF(s).  Assumed IID across the 0 dimension.
    samples2: Tensor of shape [m] + B.  Samples from some (batch of)
      scalar-event distribution(s) of interest, giving a (batch of) empirical
      CDF(s).  Assumed IID across the 0 dimension.
    name: A name for this operation (optional).

  Returns:
    distance: Tensor of shape B: (Absolute) Kolmogorov-Smirnov distance between
      the two empirical CDFs given by the samples.
  """
  with tf.compat.v1.name_scope(name, 'kolmogorov_smirnov_distance_two_sample',
                               [samples1, samples2]):
    dtype = dtype_util.common_dtype([samples1, samples2], tf.float32)
    samples1 = tf.convert_to_tensor(
        value=samples1, name='samples1', dtype=dtype)
    samples2 = tf.convert_to_tensor(
        value=samples2, name='samples2', dtype=dtype)
    samples2 = _move_dim_and_sort(samples2)

    cdf = functools.partial(
        empirical_cdfs, samples2,
        continuity='right', dtype=samples1.dtype)
    left_continuous_cdf = functools.partial(
        empirical_cdfs, samples2,
        continuity='left', dtype=samples1.dtype)
    return kolmogorov_smirnov_distance(samples1, cdf, left_continuous_cdf)


def _move_dim_and_sort(samples):
  """Internal helper for K-S distance computation."""
  # Move the batch dimension of `samples` to the rightmost position,
  # where the _batch_sort_vector function wants it.
  samples = distribution_util.move_dimension(samples, 0, -1)

  # Order the samples within each batch member
  samples = _batch_sort_vector(samples)
  return samples


def _batch_sort_vector(x, ascending=True, name=None):
  """Batch sort.  Sorts the -1 dimension of each batch member independently."""
  with tf.compat.v1.name_scope(name, '_batch_sort_vector', [x]):
    x = tf.convert_to_tensor(value=x, name='x')
    n = tf.shape(input=x)[-1]
    if ascending:
      y, _ = tf.nn.top_k(-x, k=n, sorted=True)
      y = -y
    else:
      y, _ = tf.nn.top_k(x, k=n, sorted=True)
    y.set_shape(x.shape)
    return y


def empirical_cdfs(samples, positions, continuity='right',
                   dtype=tf.float32, name=None):
  """Evaluates the empirical CDF of a batch of samples at a batch of positions.

  If elements of `positions` might be exactly equal to elements of `samples`
  (e.g., if the underlying distribution of interest is discrete), there is a
  difference between the conventional, right-continuous CDF (Pr[X <= x]) and a
  left-continuous variant (Pr[X < x]).  The latter can be accessed by setting
  `continuity='left'`.  The difference between the right-continuous and
  left-continuous CDFs is the empirical pmf at each point, i.e., how many times
  each element of `positions` occurs in its batch of `samples`.

  Note: Returns results parallel to `positions`, i.e., the values of the
  empirical CDF at those points.

  Note: The sample dimension is _last_, and the samples must be _sorted_ within
  each batch.

  Args:
    samples: Tensor of shape `batch + [num_samples]` of samples.  The samples
      must be in ascending order within each batch member.
    positions: Tensor of shape `batch + [m]` of positions where to evaluate the
      CDFs.  The positions need not be sorted.
    continuity: Whether to return a conventional, right-continuous CDF
      (`continuity = 'right'`, default) or a left-continuous CDF (`continuity =
      'left'`).  The value at each point `x` will be `F_n(X <= x)` or
      `F_n(X < x)`, respectively.
    dtype: dtype at which to evaluate the desired empirical CDFs.
    name: A name for this operation (optional).

  Returns:
    cdf: Tensor parallel to `positions`.  For each x in `positions`, gives the
      (right- or left-continuous, per the `continuity` argument) cdf at that
      position.  If `positions` contains duplicates, `cdf` will give each the
      same value.
  """
  if continuity not in ['left', 'right']:
    msg = 'Continuity value must be "left" or "right", got {}.'.format(
        continuity)
    raise ValueError(msg)
  with tf.compat.v1.name_scope(name, 'empirical_cdfs', [samples, positions]):
    n = tf.cast(tf.shape(input=samples)[-1], dtype=dtype)
    indexes = tf.searchsorted(
        sorted_sequence=samples, values=positions, side=continuity)
    return tf.cast(indexes, dtype=dtype) / n


def _do_maximum_mean(samples, envelope, high, name=None):
  """Common code between maximum_mean and minimum_mean."""
  with tf.compat.v1.name_scope(name, 'do_maximum_mean',
                               [samples, envelope, high]):
    dtype = dtype_util.common_dtype([samples, envelope, high], tf.float32)
    samples = tf.convert_to_tensor(value=samples, name='samples', dtype=dtype)
    envelope = tf.convert_to_tensor(
        value=envelope, name='envelope', dtype=dtype)
    high = tf.convert_to_tensor(value=high, name='high', dtype=dtype)
    n = tf.rank(samples)
    # Move the batch dimension of `samples` to the rightmost position,
    # where the _batch_sort_vector function wants it.
    perm = tf.concat([tf.range(1, n), [0]], axis=0)
    samples = tf.transpose(a=samples, perm=perm)

    samples = _batch_sort_vector(samples)

    # The maximum mean is given by taking `envelope`-worth of
    # probability from the smallest samples and moving it to the
    # maximum value.  This amounts to:
    # - ignoring the smallest k samples, where `k/n < envelope`
    # - taking a `1/n - (envelope - k/n)` part of the index k sample
    # - taking all the other samples
    # - and adding `envelope * high` at the end.
    # The following is a vectorized and batched way of computing this.
    # `max_mean_contrib` is a mask implementing the previous.
    batch_size = tf.shape(input=samples)[-1]
    batch_size = tf.cast(batch_size, dtype=dtype)
    step = 1. / batch_size
    cum_steps = step * tf.range(1, batch_size + 1, dtype=dtype)
    max_mean_contrib = tf.clip_by_value(
        cum_steps - envelope[..., tf.newaxis],
        clip_value_min=0.,
        clip_value_max=step)
    return tf.reduce_sum(
        input_tensor=samples * max_mean_contrib, axis=-1) + envelope * high


def assert_true_cdf_equal_by_dkwm_two_sample(
    samples1, samples2, false_fail_rate=1e-6, name=None):
  """Asserts the full CDFs of the two given distributions are equal.

  More precisely, fails if there is enough evidence (using the
  [Dvoretzky-Kiefer-Wolfowitz-Massart inequality]
  (https://en.wikipedia.org/wiki/CDF-based_nonparametric_confidence_interval))
  that the true CDF of the distribution from which `samples1` are drawn is _not_
  the true CDF of the distribution from which `samples2` are drawn, with
  statistical significance `false_fail_rate` or stronger, otherwise passes.  If
  you also want to check that you are gathering enough evidence that a pass is
  not spurious, see `min_num_samples_for_dkwm_cdf_two_sample_test` and
  `min_discrepancy_of_true_cdfs_detectable_by_dkwm_two_sample`.

  This test works as written even if the distribution in question has atoms
  (e.g., is discrete).

  Note that `false_fail_rate` is a total false failure rate for all
  the assertions in the batch.  As such, if the batch is nontrivial,
  the assertion will insist on stronger evidence to fail any one member.

  Args:
    samples1: Tensor of shape [n] + B.  Samples from some (batch of)
      scalar-event distribution(s) of interest, giving a (batch of) empirical
      CDF(s).  Assumed IID across the 0 dimension.
    samples2: Tensor of shape [m] + B.  Samples from some (batch of)
      scalar-event distribution(s) of interest, giving a (batch of) empirical
      CDF(s).  Assumed IID across the 0 dimension.
    false_fail_rate: *Scalar* floating-point `Tensor` admissible total
      rate of mistakes.
    name: A name for this operation (optional).

  Returns:
    check: Op that raises `InvalidArgumentError` if any expected CDF is
      outside the corresponding confidence envelope.
  """
  with tf.compat.v1.name_scope(name, 'assert_true_cdf_equal_by_dkwm_two_sample',
                               [samples1, samples2, false_fail_rate]):
    dtype = dtype_util.common_dtype(
        [samples1, samples2, false_fail_rate], tf.float32)
    samples1 = tf.convert_to_tensor(
        value=samples1, name='samples1', dtype=dtype)
    samples2 = tf.convert_to_tensor(
        value=samples2, name='samples2', dtype=dtype)
    false_fail_rate = tf.convert_to_tensor(
        value=false_fail_rate, name='false_fail_rate', dtype=dtype)
    tf.compat.v1.assert_scalar(false_fail_rate)  # Static shape
    compatible_samples = tf.compat.v1.assert_equal(
        tf.shape(input=samples1)[1:],
        tf.shape(input=samples2)[1:])
    with tf.control_dependencies([compatible_samples]):
      itemwise_false_fail_rate = _itemwise_error_rate(
          total_rate=false_fail_rate,
          param_tensors=[], samples_tensor=samples1)
      n1 = tf.shape(input=samples1)[0]
      envelope1 = _dkwm_cdf_envelope(n1, itemwise_false_fail_rate)
      n2 = tf.shape(input=samples2)[0]
      envelope2 = _dkwm_cdf_envelope(n2, itemwise_false_fail_rate)
      distance = kolmogorov_smirnov_distance_two_sample(samples1, samples2)
      return tf.compat.v1.assert_less_equal(
          distance, envelope1 + envelope2,
          message='Empirical CDFs outside joint K-S envelope')


def min_discrepancy_of_true_cdfs_detectable_by_dkwm_two_sample(
    n1, n2, false_fail_rate, false_pass_rate, name=None):
  """Returns the minimum CDF discrepancy that a two-sample DKWM test can detect.

  DKWM is the [Dvoretzky-Kiefer-Wolfowitz-Massart inequality]
  (https://en.wikipedia.org/wiki/CDF-based_nonparametric_confidence_interval).

  Note that `false_fail_rate` is a total false failure rate for all
  the tests in the batch.  As such, if the batch is nontrivial, each
  member will demand more samples.  The `false_pass_rate` is also
  interpreted as a total, but is treated asymmetrically: If each test
  in the batch detects its corresponding discrepancy with probability
  at least `1 - false_pass_rate`, then running all those tests and
  failing if any one fails will jointly detect all those discrepancies
  with the same `false_pass_rate`.

  Args:
    n1: `Tensor` of numbers of samples to be drawn from the distributions A.
    n2: `Tensor` of numbers of samples to be drawn from the distributions B.
    false_fail_rate: *Scalar* floating-point `Tensor` admissible total
      rate of false failures.
    false_pass_rate: *Scalar* floating-point `Tensor` admissible rate
      of false passes.
    name: A name for this operation (optional).

  Returns:
    discr: `Tensor` of lower bounds on the K-S distances between true
       CDFs detectable by a DKWM-based test.

  For each batch member `i`, of `K` total, drawing `n1[i]` samples from scalar
  distribution A and `n2[i]` samples from scalar distribution B is enough to
  detect a K-S distance in CDFs of size `discr[i]` or more.  Specifically, we
  guarantee that (a) if their true CDFs are the same, then
  `assert_true_cdf_equal_by_dkwm_two_sample` will fail with probability at most
  `false_fail_rate / K` (which amounts to `false_fail_rate` if applied to the
  whole batch at once), and (b) if their true CDFs differ by at least
  `discr[i]`, `assert_true_cdf_equal_by_dkwm_two_sample` will pass with
  probability at most `false_pass_rate`.

  The detectable discrepancy scales as

  - `O(1 / sqrt(n[i]))`,
  - `O(-log(false_fail_rate/K))`, and
  - `O(-log(false_pass_rate))`.
  """
  with tf.compat.v1.name_scope(
      name, 'min_discrepancy_of_true_cdfs_detectable_by_dkwm_two_sample',
      [n1, n2, false_fail_rate, false_pass_rate]):
    dtype = dtype_util.common_dtype(
        [n1, n2, false_fail_rate, false_pass_rate], tf.float32)
    n1 = tf.convert_to_tensor(value=n1, name='n1', dtype=dtype)
    n2 = tf.convert_to_tensor(value=n2, name='n2', dtype=dtype)
    false_fail_rate = tf.convert_to_tensor(
        value=false_fail_rate, name='false_fail_rate', dtype=dtype)
    false_pass_rate = tf.convert_to_tensor(
        value=false_pass_rate, name='false_pass_rate', dtype=dtype)
    # To fail to detect a discrepancy with the two-sample test, the given
    # samples must be close enough that they could have come from the same true
    # CDF.  That is, each must be close enough to a common CDF for the
    # one-sample test to be able to fail to detect the discrepancy.
    d1 = min_discrepancy_of_true_cdfs_detectable_by_dkwm(
        n1, false_fail_rate / 2., false_pass_rate / 2.)
    d2 = min_discrepancy_of_true_cdfs_detectable_by_dkwm(
        n2, false_fail_rate / 2., false_pass_rate / 2.)
    return d1 + d2


def min_num_samples_for_dkwm_cdf_two_sample_test(
    discrepancy, false_fail_rate=1e-6, false_pass_rate=1e-6, name=None):
  """Returns how many samples suffice for a two-sample DKWM CDF test.

  Args:
    discrepancy: Floating-point `Tensor` of desired upper limits on K-S
      distances that may go undetected with probability higher than
      `1 - false_pass_rate`.
    false_fail_rate: *Scalar* floating-point `Tensor` admissible total
      rate of false failures.
    false_pass_rate: *Scalar* floating-point `Tensor` admissible rate
      of false passes.
    name: A name for this operation (optional).

  Returns:
    n1: `Tensor` of numbers of samples to be drawn from the distributions A.
    n2: `Tensor` of numbers of samples to be drawn from the distributions B.

  For each batch member `i`, of `K` total, drawing `n1[i]` samples from scalar
  distribution A and `n2[i]` samples from scalar distribution B is enough to
  detect a K-S distance of CDFs of size `discrepancy[i]` or more.  Specifically,
  we guarantee that (a) if the true CDFs are equal, then
  `assert_true_cdf_equal_by_dkwm_two_sample` will fail with probability at most
  `false_fail_rate / K` (which amounts to `false_fail_rate` if applied to the
  whole batch at once), and (b) if the true CDFs differ from each other least
  `discrepancy[i]`, `assert_true_cdf_equal_by_dkwm_two_sample` will pass with
  probability at most `false_pass_rate`.

  The required number of samples scales as

  - `O(-log(false_fail_rate/K))`,
  - `O(-log(false_pass_rate))`, and
  - `O(1 / discrepancy[i]**2)`.
  """
  with tf.compat.v1.name_scope(name,
                               'min_num_samples_for_dkwm_cdf_two_sample_test',
                               [discrepancy, false_fail_rate, false_pass_rate]):
    dtype = dtype_util.common_dtype(
        [discrepancy, false_fail_rate, false_pass_rate], tf.float32)
    discrepancy = tf.convert_to_tensor(
        value=discrepancy, name='discrepancy', dtype=dtype)
    false_fail_rate = tf.convert_to_tensor(
        value=false_fail_rate, name='false_fail_rate', dtype=dtype)
    false_pass_rate = tf.convert_to_tensor(
        value=false_pass_rate, name='false_pass_rate', dtype=dtype)
    n = min_num_samples_for_dkwm_cdf_test(
        discrepancy / 2., false_fail_rate / 2., false_pass_rate / 2.)
    return n, n


def _maximum_mean(samples, envelope, high, name=None):
  """Returns a stochastic upper bound on the mean of a scalar distribution.

  The idea is that if the true CDF is within an `eps`-envelope of the
  empirical CDF of the samples, and the support is bounded above, then
  the mean is bounded above as well.  In symbols,

  ```none
  sup_x(|F_n(x) - F(x)|) < eps
  ```

  The 0th dimension of `samples` is interpreted as independent and
  identically distributed samples.  The remaining dimensions are
  broadcast together with `envelope` and `high`, and operated on
  separately.

  Args:
    samples: Floating-point `Tensor` of samples from the distribution(s)
      of interest.  Entries are assumed IID across the 0th dimension.
      The other dimensions must broadcast with `envelope` and `high`.
    envelope: Floating-point `Tensor` of sizes of admissible CDF
      envelopes (i.e., the `eps` above).
    high: Floating-point `Tensor` of upper bounds on the distributions'
      supports.  `samples <= high`.
    name: A name for this operation (optional).

  Returns:
    bound: Floating-point `Tensor` of upper bounds on the true means.

  Raises:
    InvalidArgumentError: If some `sample` is found to be larger than
      the corresponding `high`.
  """
  with tf.compat.v1.name_scope(name, 'maximum_mean', [samples, envelope, high]):
    dtype = dtype_util.common_dtype([samples, envelope, high], tf.float32)
    samples = tf.convert_to_tensor(value=samples, name='samples', dtype=dtype)
    envelope = tf.convert_to_tensor(
        value=envelope, name='envelope', dtype=dtype)
    high = tf.convert_to_tensor(value=high, name='high', dtype=dtype)

    xmax = tf.reduce_max(input_tensor=samples, axis=[0])
    msg = 'Given sample maximum value exceeds expectations'
    check_op = tf.compat.v1.assert_less_equal(xmax, high, message=msg)
    with tf.control_dependencies([check_op]):
      return tf.identity(_do_maximum_mean(samples, envelope, high))


def _minimum_mean(samples, envelope, low, name=None):
  """Returns a stochastic lower bound on the mean of a scalar distribution.

  The idea is that if the true CDF is within an `eps`-envelope of the
  empirical CDF of the samples, and the support is bounded below, then
  the mean is bounded below as well.  In symbols,

  ```none
  sup_x(|F_n(x) - F(x)|) < eps
  ```

  The 0th dimension of `samples` is interpreted as independent and
  identically distributed samples.  The remaining dimensions are
  broadcast together with `envelope` and `low`, and operated on
  separately.

  Args:
    samples: Floating-point `Tensor` of samples from the distribution(s)
      of interest.  Entries are assumed IID across the 0th dimension.
      The other dimensions must broadcast with `envelope` and `low`.
    envelope: Floating-point `Tensor` of sizes of admissible CDF
      envelopes (i.e., the `eps` above).
    low: Floating-point `Tensor` of lower bounds on the distributions'
      supports.  `samples >= low`.
    name: A name for this operation (optional).

  Returns:
    bound: Floating-point `Tensor` of lower bounds on the true means.

  Raises:
    InvalidArgumentError: If some `sample` is found to be smaller than
      the corresponding `low`.
  """
  with tf.compat.v1.name_scope(name, 'minimum_mean', [samples, envelope, low]):
    dtype = dtype_util.common_dtype([samples, envelope, low], tf.float32)
    samples = tf.convert_to_tensor(value=samples, name='samples', dtype=dtype)
    envelope = tf.convert_to_tensor(
        value=envelope, name='envelope', dtype=dtype)
    low = tf.convert_to_tensor(value=low, name='low', dtype=dtype)

    xmin = tf.reduce_min(input_tensor=samples, axis=[0])
    msg = 'Given sample minimum value falls below expectations'
    check_op = tf.compat.v1.assert_greater_equal(xmin, low, message=msg)
    with tf.control_dependencies([check_op]):
      return - _do_maximum_mean(-samples, envelope, -low)


def _dkwm_cdf_envelope(n, error_rate, name=None):
  """Computes the CDF envelope that the DKWM inequality licenses.

  The [Dvoretzky-Kiefer-Wolfowitz-Massart inequality]
  (https://en.wikipedia.org/wiki/CDF-based_nonparametric_confidence_interval)
  gives a stochastic bound on the distance between the true cumulative
  distribution function (CDF) of any distribution and its empirical
  CDF.  To wit, for `n` iid samples from any distribution with CDF F,

  ```none
  P(sup_x |F_n(x) - F(x)| > eps) < 2exp(-2n eps^2)
  ```

  This function computes the envelope size `eps` as a function of the
  number of samples `n` and the desired limit on the left-hand
  probability above.

  Args:
    n: `Tensor` of numbers of samples drawn.
    error_rate: Floating-point `Tensor` of admissible rates of mistakes.
    name: A name for this operation (optional).

  Returns:
    eps: `Tensor` of maximum distances the true CDF can be from the
      empirical CDF.  This scales as `O(sqrt(-log(error_rate)))` and
      as `O(1 / sqrt(n))`.  The shape is the broadcast of `n` and
      `error_rate`.
  """
  with tf.compat.v1.name_scope(name, 'dkwm_cdf_envelope', [n, error_rate]):
    n = tf.cast(n, dtype=error_rate.dtype)
    return tf.sqrt(-tf.math.log(error_rate / 2.) / (2. * n))


def _check_shape_dominates(samples, parameters):
  """Check that broadcasting `samples` against `parameters` does not expand it.

  Why?  To be very sure that the samples tensor is not accidentally enlarged by
  broadcasting against tensors that are supposed to be describing the
  distribution(s) sampled from, lest the sample counts end up inflated.

  Args:
    samples: A `Tensor` whose shape is to be protected against broadcasting.
    parameters: A list of `Tensor`s who are parameters for the statistical test.

  Returns:
    samples: Return original `samples` with control dependencies attached
      to ensure no broadcasting.
  """
  def check(t):
    samples_batch_shape = tf.shape(input=samples)[1:]
    broadcasted_batch_shape = tf.broadcast_dynamic_shape(
        samples_batch_shape, tf.shape(input=t))
    # This rank check ensures that I don't get a wrong answer from the
    # _shapes_ broadcasting against each other.
    samples_batch_ndims = tf.size(input=samples_batch_shape)
    ge = tf.compat.v1.assert_greater_equal(samples_batch_ndims, tf.rank(t))
    eq = tf.compat.v1.assert_equal(samples_batch_shape, broadcasted_batch_shape)
    return ge, eq
  checks = list(itertools.chain(*[check(t) for t in parameters]))
  with tf.control_dependencies(checks):
    return tf.identity(samples)


def true_mean_confidence_interval_by_dkwm(
    samples, low, high, error_rate=1e-6, name=None):
  """Computes a confidence interval for the mean of a scalar distribution.

  In batch mode, computes confidence intervals for all distributions
  in the batch (which need not be identically distributed).

  Relies on the [Dvoretzky-Kiefer-Wolfowitz-Massart inequality]
  (https://en.wikipedia.org/wiki/CDF-based_nonparametric_confidence_interval).

  The probability (over the randomness of drawing the given samples)
  that any true mean is outside the corresponding returned interval is
  no more than the given `error_rate`.  The size of the intervals
  scale as
  `O(1 / sqrt(#samples))`, as `O(high - low)`, and as `O(-log(error_rate))`.

  Note that `error_rate` is a total error rate for all the confidence
  intervals in the batch.  As such, if the batch is nontrivial, the
  error rate is not broadcast but divided (evenly) among the batch
  members.

  Args:
    samples: Floating-point `Tensor` of samples from the distribution(s)
      of interest.  Entries are assumed IID across the 0th dimension.
      The other dimensions must broadcast with `low` and `high`.
      The support is bounded: `low <= samples <= high`.
    low: Floating-point `Tensor` of lower bounds on the distributions'
      supports.
    high: Floating-point `Tensor` of upper bounds on the distributions'
      supports.
    error_rate: *Scalar* floating-point `Tensor` admissible total rate
      of mistakes.
    name: A name for this operation (optional).

  Returns:
    low: A floating-point `Tensor` of stochastic lower bounds on the
      true means.
    high: A floating-point `Tensor` of stochastic upper bounds on the
      true means.
  """
  with tf.compat.v1.name_scope(name, 'true_mean_confidence_interval_by_dkwm',
                               [samples, low, high, error_rate]):
    dtype = dtype_util.common_dtype(
        [samples, low, high, error_rate], tf.float32)
    samples = tf.convert_to_tensor(value=samples, name='samples', dtype=dtype)
    low = tf.convert_to_tensor(value=low, name='low', dtype=dtype)
    high = tf.convert_to_tensor(value=high, name='high', dtype=dtype)
    error_rate = tf.convert_to_tensor(
        value=error_rate, name='error_rate', dtype=dtype)
    samples = _check_shape_dominates(samples, [low, high])
    tf.compat.v1.assert_scalar(error_rate)  # Static shape
    itemwise_error_rate = _itemwise_error_rate(
        total_rate=error_rate, param_tensors=[low, high],
        samples_tensor=samples)
    n = tf.shape(input=samples)[0]
    envelope = _dkwm_cdf_envelope(n, itemwise_error_rate)
    min_mean = _minimum_mean(samples, envelope, low)
    max_mean = _maximum_mean(samples, envelope, high)
    return min_mean, max_mean


def _itemwise_error_rate(
    total_rate, param_tensors, samples_tensor=None, name=None):
  """Distributes a total error rate for a batch of assertions."""
  with tf.compat.v1.name_scope(name, 'itemwise_error_rate',
                               [total_rate, param_tensors, samples_tensor]):
    result_shape = [1]
    for p_tensor in param_tensors:
      result_shape = tf.broadcast_dynamic_shape(
          tf.shape(input=p_tensor), result_shape)
    if samples_tensor is not None:
      result_shape = tf.broadcast_dynamic_shape(
          tf.shape(input=samples_tensor)[1:], result_shape)
    num_items = tf.reduce_prod(input_tensor=result_shape)
    return total_rate / tf.cast(num_items, dtype=total_rate.dtype)


def assert_true_mean_equal_by_dkwm(
    samples, low, high, expected, false_fail_rate=1e-6, name=None):
  """Asserts the mean of the given distribution is as expected.

  More precisely, fails if there is enough evidence (using the
  [Dvoretzky-Kiefer-Wolfowitz-Massart inequality]
  (https://en.wikipedia.org/wiki/CDF-based_nonparametric_confidence_interval))
  that the true mean of some distribution from which the given samples are
  drawn is _not_ the given expected mean with statistical significance
  `false_fail_rate` or stronger, otherwise passes.  If you also want to
  check that you are gathering enough evidence that a pass is not
  spurious, see `min_num_samples_for_dkwm_mean_test` and
  `min_discrepancy_of_true_means_detectable_by_dkwm`.

  Note that `false_fail_rate` is a total false failure rate for all
  the assertions in the batch.  As such, if the batch is nontrivial,
  the assertion will insist on stronger evidence to fail any one member.

  Args:
    samples: Floating-point `Tensor` of samples from the distribution(s)
      of interest.  Entries are assumed IID across the 0th dimension.
      The other dimensions must broadcast with `low` and `high`.
      The support is bounded: `low <= samples <= high`.
    low: Floating-point `Tensor` of lower bounds on the distributions'
      supports.
    high: Floating-point `Tensor` of upper bounds on the distributions'
      supports.
    expected: Floating-point `Tensor` of expected true means.
    false_fail_rate: *Scalar* floating-point `Tensor` admissible total
      rate of mistakes.
    name: A name for this operation (optional).

  Returns:
    check: Op that raises `InvalidArgumentError` if any expected mean is
      outside the corresponding confidence interval.
  """
  with tf.compat.v1.name_scope(name, 'assert_true_mean_equal_by_dkwm',
                               [samples, low, high, expected, false_fail_rate]):
    return assert_true_mean_in_interval_by_dkwm(
        samples, low, high, expected, expected, false_fail_rate)


def min_discrepancy_of_true_means_detectable_by_dkwm(
    n, low, high, false_fail_rate, false_pass_rate, name=None):
  """Returns the minimum mean discrepancy that a DKWM-based test can detect.

  DKWM is the [Dvoretzky-Kiefer-Wolfowitz-Massart inequality]
  (https://en.wikipedia.org/wiki/CDF-based_nonparametric_confidence_interval).

  Note that `false_fail_rate` is a total false failure rate for all
  the tests in the batch.  As such, if the batch is nontrivial, each
  member will demand more samples.  The `false_pass_rate` is also
  interpreted as a total, but is treated asymmetrically: If each test
  in the batch detects its corresponding discrepancy with probability
  at least `1 - false_pass_rate`, then running all those tests and
  failing if any one fails will jointly detect all those discrepancies
  with the same `false_pass_rate`.

  Args:
    n: `Tensor` of numbers of samples to be drawn from the distributions
      of interest.
    low: Floating-point `Tensor` of lower bounds on the distributions'
      supports.
    high: Floating-point `Tensor` of upper bounds on the distributions'
      supports.
    false_fail_rate: *Scalar* floating-point `Tensor` admissible total
      rate of false failures.
    false_pass_rate: *Scalar* floating-point `Tensor` admissible rate
      of false passes.
    name: A name for this operation (optional).

  Returns:
    discr: `Tensor` of lower bounds on the distances between true
       means detectable by a DKWM-based test.

  For each batch member `i`, of `K` total, drawing `n[i]` samples from
  some scalar distribution supported on `[low[i], high[i]]` is enough
  to detect a difference in means of size `discr[i]` or more.
  Specifically, we guarantee that (a) if the true mean is the expected
  mean (resp. in the expected interval), then `assert_true_mean_equal_by_dkwm`
  (resp. `assert_true_mean_in_interval_by_dkwm`) will fail with
  probability at most `false_fail_rate / K` (which amounts to
  `false_fail_rate` if applied to the whole batch at once), and (b) if
  the true mean differs from the expected mean (resp. falls outside
  the expected interval) by at least `discr[i]`,
  `assert_true_mean_equal_by_dkwm`
  (resp. `assert_true_mean_in_interval_by_dkwm`) will pass with
  probability at most `false_pass_rate`.

  The detectable discrepancy scales as

  - `O(high[i] - low[i])`,
  - `O(1 / sqrt(n[i]))`,
  - `O(-log(false_fail_rate/K))`, and
  - `O(-log(false_pass_rate))`.
  """
  with tf.compat.v1.name_scope(
      name, 'min_discrepancy_of_true_means_detectable_by_dkwm',
      [n, low, high, false_fail_rate, false_pass_rate]):
    dtype = dtype_util.common_dtype(
        [n, low, high, false_fail_rate, false_pass_rate], tf.float32)
    n = tf.convert_to_tensor(value=n, name='n', dtype=dtype)
    low = tf.convert_to_tensor(value=low, name='low', dtype=dtype)
    high = tf.convert_to_tensor(value=high, name='high', dtype=dtype)
    false_fail_rate = tf.convert_to_tensor(
        value=false_fail_rate, name='false_fail_rate', dtype=dtype)
    false_pass_rate = tf.convert_to_tensor(
        value=false_pass_rate, name='false_pass_rate', dtype=dtype)
    cdf_discrepancy = min_discrepancy_of_true_cdfs_detectable_by_dkwm(
        n, false_fail_rate, false_pass_rate)
    return (high - low) * cdf_discrepancy


def min_num_samples_for_dkwm_mean_test(
    discrepancy, low, high,
    false_fail_rate=1e-6, false_pass_rate=1e-6, name=None):
  """Returns how many samples suffice for a one-sample DKWM mean test.

  To wit, returns an upper bound on the number of samples necessary to
  guarantee detecting a mean difference of at least the given
  `discrepancy`, with the given `false_fail_rate` and `false_pass_rate`,
  using the [Dvoretzky-Kiefer-Wolfowitz-Massart inequality]
  (https://en.wikipedia.org/wiki/CDF-based_nonparametric_confidence_interval)
  on a scalar distribution supported on `[low, high]`.

  Args:
    discrepancy: Floating-point `Tensor` of desired upper limits on mean
      differences that may go undetected with probability higher than
      `1 - false_pass_rate`.
    low: `Tensor` of lower bounds on the distributions' support.
    high: `Tensor` of upper bounds on the distributions' support.
    false_fail_rate: *Scalar* floating-point `Tensor` admissible total
      rate of false failures.
    false_pass_rate: *Scalar* floating-point `Tensor` admissible rate
      of false passes.
    name: A name for this operation (optional).

  Returns:
    n: `Tensor` of numbers of samples to be drawn from the distributions
      of interest.

  The `discrepancy`, `low`, and `high` tensors must have
  broadcast-compatible shapes.

  For each batch member `i`, of `K` total, drawing `n[i]` samples from
  some scalar distribution supported on `[low[i], high[i]]` is enough
  to detect a difference in means of size `discrepancy[i]` or more.
  Specifically, we guarantee that (a) if the true mean is the expected
  mean (resp. in the expected interval), then `assert_true_mean_equal_by_dkwm`
  (resp. `assert_true_mean_in_interval_by_dkwm`) will fail with
  probability at most `false_fail_rate / K` (which amounts to
  `false_fail_rate` if applied to the whole batch at once), and (b) if
  the true mean differs from the expected mean (resp. falls outside
  the expected interval) by at least `discrepancy[i]`,
  `assert_true_mean_equal_by_dkwm`
  (resp. `assert_true_mean_in_interval_by_dkwm`) will pass with
  probability at most `false_pass_rate`.

  The required number of samples scales
  as `O((high[i] - low[i])**2)`, `O(-log(false_fail_rate/K))`,
  `O(-log(false_pass_rate))`, and `O(1 / discrepancy[i]**2)`.
  """
  with tf.compat.v1.name_scope(
      name, 'min_num_samples_for_dkwm_mean_test',
      [low, high, false_fail_rate, false_pass_rate, discrepancy]):
    dtype = dtype_util.common_dtype(
        [low, high, false_fail_rate, false_pass_rate, discrepancy], tf.float32)
    discrepancy = tf.convert_to_tensor(
        value=discrepancy, name='discrepancy', dtype=dtype)
    low = tf.convert_to_tensor(value=low, name='low', dtype=dtype)
    high = tf.convert_to_tensor(value=high, name='high', dtype=dtype)
    false_fail_rate = tf.convert_to_tensor(
        value=false_fail_rate, name='false_fail_rate', dtype=dtype)
    false_pass_rate = tf.convert_to_tensor(
        value=false_pass_rate, name='false_pass_rate', dtype=dtype)
    cdf_discrepancy = discrepancy / (high - low)
    return min_num_samples_for_dkwm_cdf_test(
        cdf_discrepancy, false_fail_rate, false_pass_rate)


def assert_true_mean_in_interval_by_dkwm(
    samples, low, high, expected_low, expected_high,
    false_fail_rate=1e-6, name=None):
  """Asserts the mean of the given distribution is in the given interval.

  More precisely, fails if there is enough evidence (using the
  [Dvoretzky-Kiefer-Wolfowitz-Massart inequality]
  (https://en.wikipedia.org/wiki/CDF-based_nonparametric_confidence_interval))
  that the mean of the distribution from which the given samples are
  drawn is _outside_ the given interval with statistical significance
  `false_fail_rate` or stronger, otherwise passes.  If you also want
  to check that you are gathering enough evidence that a pass is not
  spurious, see `min_num_samples_for_dkwm_mean_test` and
  `min_discrepancy_of_true_means_detectable_by_dkwm`.

  Note that `false_fail_rate` is a total false failure rate for all
  the assertions in the batch.  As such, if the batch is nontrivial,
  the assertion will insist on stronger evidence to fail any one member.

  Args:
    samples: Floating-point `Tensor` of samples from the distribution(s)
      of interest.  Entries are assumed IID across the 0th dimension.
      The other dimensions must broadcast with `low` and `high`.
      The support is bounded: `low <= samples <= high`.
    low: Floating-point `Tensor` of lower bounds on the distributions'
      supports.
    high: Floating-point `Tensor` of upper bounds on the distributions'
      supports.
    expected_low: Floating-point `Tensor` of lower bounds on the
      expected true means.
    expected_high: Floating-point `Tensor` of upper bounds on the
      expected true means.
    false_fail_rate: *Scalar* floating-point `Tensor` admissible total
      rate of mistakes.
    name: A name for this operation (optional).

  Returns:
    check: Op that raises `InvalidArgumentError` if any expected mean
      interval does not overlap with the corresponding confidence
      interval.
  """
  args_list = [samples, low, high, expected_low, expected_high, false_fail_rate]
  with tf.compat.v1.name_scope(name, 'assert_true_mean_in_interval_by_dkwm',
                               args_list):
    dtype = dtype_util.common_dtype(args_list, tf.float32)
    samples = tf.convert_to_tensor(value=samples, name='samples', dtype=dtype)
    low = tf.convert_to_tensor(value=low, name='low', dtype=dtype)
    high = tf.convert_to_tensor(value=high, name='high', dtype=dtype)
    expected_low = tf.convert_to_tensor(
        value=expected_low, name='expected_low', dtype=dtype)
    expected_high = tf.convert_to_tensor(
        value=expected_high, name='expected_high', dtype=dtype)
    false_fail_rate = tf.convert_to_tensor(
        value=false_fail_rate, name='false_fail_rate', dtype=dtype)
    samples = _check_shape_dominates(
        samples, [low, high, expected_low, expected_high])
    min_mean, max_mean = true_mean_confidence_interval_by_dkwm(
        samples, low, high, false_fail_rate)
    # Assert that the interval [min_mean, max_mean] intersects the
    # interval [expected_low, expected_high].  This is true if
    #   max_mean >= expected_low and min_mean <= expected_high.
    # By DeMorgan's law, that's also equivalent to
    #   not (max_mean < expected_low or min_mean > expected_high),
    # which is a way of saying the two intervals are not disjoint.
    check_confidence_interval_can_intersect = tf.compat.v1.assert_greater_equal(
        max_mean,
        expected_low,
        message='Confidence interval does not '
        'intersect: true mean smaller than expected')
    with tf.control_dependencies([check_confidence_interval_can_intersect]):
      return tf.compat.v1.assert_less_equal(
          min_mean,
          expected_high,
          message='Confidence interval does not '
          'intersect: true mean greater than expected')


def assert_true_mean_equal_by_dkwm_two_sample(
    samples1, low1, high1, samples2, low2, high2,
    false_fail_rate=1e-6, name=None):
  """Asserts the means of the given distributions are equal.

  More precisely, fails if there is enough evidence (using the
  [Dvoretzky-Kiefer-Wolfowitz-Massart inequality]
  (https://en.wikipedia.org/wiki/CDF-based_nonparametric_confidence_interval))
  that the means of the distributions from which the given samples are
  drawn are _not_ equal with statistical significance `false_fail_rate`
  or stronger, otherwise passes.  If you also want to check that you
  are gathering enough evidence that a pass is not spurious, see
  `min_num_samples_for_dkwm_mean_two_sample_test` and
  `min_discrepancy_of_true_means_detectable_by_dkwm_two_sample`.

  Note that `false_fail_rate` is a total false failure rate for all
  the assertions in the batch.  As such, if the batch is nontrivial,
  the assertion will insist on stronger evidence to fail any one member.

  Args:
    samples1: Floating-point `Tensor` of samples from the
      distribution(s) A.  Entries are assumed IID across the 0th
      dimension.  The other dimensions must broadcast with `low1`,
      `high1`, `low2`, and `high2`.
      The support is bounded: `low1 <= samples1 <= high1`.
    low1: Floating-point `Tensor` of lower bounds on the supports of the
      distributions A.
    high1: Floating-point `Tensor` of upper bounds on the supports of
      the distributions A.
    samples2: Floating-point `Tensor` of samples from the
      distribution(s) B.  Entries are assumed IID across the 0th
      dimension.  The other dimensions must broadcast with `low1`,
      `high1`, `low2`, and `high2`.
      The support is bounded: `low2 <= samples2 <= high2`.
    low2: Floating-point `Tensor` of lower bounds on the supports of the
      distributions B.
    high2: Floating-point `Tensor` of upper bounds on the supports of
      the distributions B.
    false_fail_rate: *Scalar* floating-point `Tensor` admissible total
      rate of mistakes.
    name: A name for this operation (optional).

  Returns:
    check: Op that raises `InvalidArgumentError` if any pair of confidence
      intervals true for corresponding true means do not overlap.
  """
  args_list = [samples1, low1, high1, samples2, low2, high2, false_fail_rate]
  with tf.compat.v1.name_scope(
      name, 'assert_true_mean_equal_by_dkwm_two_sample', args_list):
    dtype = dtype_util.common_dtype(args_list, tf.float32)
    samples1 = tf.convert_to_tensor(
        value=samples1, name='samples1', dtype=dtype)
    low1 = tf.convert_to_tensor(value=low1, name='low1', dtype=dtype)
    high1 = tf.convert_to_tensor(value=high1, name='high1', dtype=dtype)
    samples2 = tf.convert_to_tensor(
        value=samples2, name='samples2', dtype=dtype)
    low2 = tf.convert_to_tensor(value=low2, name='low2', dtype=dtype)
    high2 = tf.convert_to_tensor(value=high2, name='high2', dtype=dtype)
    false_fail_rate = tf.convert_to_tensor(
        value=false_fail_rate, name='false_fail_rate', dtype=dtype)
    samples1 = _check_shape_dominates(samples1, [low1, high1])
    samples2 = _check_shape_dominates(samples2, [low2, high2])
    compatible_samples = tf.compat.v1.assert_equal(
        tf.shape(input=samples1)[1:],
        tf.shape(input=samples2)[1:])
    with tf.control_dependencies([compatible_samples]):
      # Could in principle play games with cleverly allocating
      # significance instead of the even split below.  It may be possible
      # to get tighter intervals, in order to obtain a higher power test.
      # Any allocation strategy that depends only on the support bounds
      # and sample counts should be valid; however, because the intervals
      # scale as O(-log(false_fail_rate)), there doesn't seem to be much
      # room to win.
      min_mean_2, max_mean_2 = true_mean_confidence_interval_by_dkwm(
          samples2, low2, high2, false_fail_rate / 2.)
      return assert_true_mean_in_interval_by_dkwm(
          samples1, low1, high1, min_mean_2, max_mean_2, false_fail_rate / 2.)


def min_discrepancy_of_true_means_detectable_by_dkwm_two_sample(
    n1, low1, high1, n2, low2, high2,
    false_fail_rate, false_pass_rate, name=None):
  """Returns the minimum mean discrepancy for a two-sample DKWM-based test.

  DKWM is the [Dvoretzky-Kiefer-Wolfowitz-Massart inequality]
  (https://en.wikipedia.org/wiki/CDF-based_nonparametric_confidence_interval).

  Note that `false_fail_rate` is a total false failure rate for all
  the tests in the batch.  As such, if the batch is nontrivial, each
  member will demand more samples.  The `false_pass_rate` is also
  interpreted as a total, but is treated asymmetrically: If each test
  in the batch detects its corresponding discrepancy with probability
  at least `1 - false_pass_rate`, then running all those tests and
  failing if any one fails will jointly detect all those discrepancies
  with the same `false_pass_rate`.

  Args:
    n1: `Tensor` of numbers of samples to be drawn from the distributions A.
    low1: Floating-point `Tensor` of lower bounds on the supports of the
      distributions A.
    high1: Floating-point `Tensor` of upper bounds on the supports of
      the distributions A.
    n2: `Tensor` of numbers of samples to be drawn from the distributions B.
    low2: Floating-point `Tensor` of lower bounds on the supports of the
      distributions B.
    high2: Floating-point `Tensor` of upper bounds on the supports of
      the distributions B.
    false_fail_rate: *Scalar* floating-point `Tensor` admissible total
      rate of false failures.
    false_pass_rate: *Scalar* floating-point `Tensor` admissible rate
      of false passes.
    name: A name for this operation (optional).

  Returns:
    discr: `Tensor` of lower bounds on the distances between true means
       detectable by a two-sample DKWM-based test.

  For each batch member `i`, of `K` total, drawing `n1[i]` samples
  from scalar distribution A supported on `[low1[i], high1[i]]` and `n2[i]`
  samples from scalar distribution B supported on `[low2[i], high2[i]]`
  is enough to detect a difference in their true means of size
  `discr[i]` or more.  Specifically, we guarantee that (a) if their
  true means are equal, `assert_true_mean_equal_by_dkwm_two_sample`
  will fail with probability at most `false_fail_rate/K` (which
  amounts to `false_fail_rate` if applied to the whole batch at once),
  and (b) if their true means differ by at least `discr[i]`,
  `assert_true_mean_equal_by_dkwm_two_sample` will pass with
  probability at most `false_pass_rate`.

  The detectable distribution scales as

  - `O(high1[i] - low1[i])`, `O(high2[i] - low2[i])`,
  - `O(1 / sqrt(n1[i]))`, `O(1 / sqrt(n2[i]))`,
  - `O(-log(false_fail_rate/K))`, and
  - `O(-log(false_pass_rate))`.
  """
  args_list = (
      [n1, low1, high1, n2, low2, high2, false_fail_rate, false_pass_rate])
  with tf.compat.v1.name_scope(
      name, 'min_discrepancy_of_true_means_detectable_by_dkwm_two_sample',
      args_list):
    dtype = dtype_util.common_dtype(args_list, tf.float32)
    n1 = tf.convert_to_tensor(value=n1, name='n1', dtype=dtype)
    low1 = tf.convert_to_tensor(value=low1, name='low1', dtype=dtype)
    high1 = tf.convert_to_tensor(value=high1, name='high1', dtype=dtype)
    n2 = tf.convert_to_tensor(value=n2, name='n2', dtype=dtype)
    low2 = tf.convert_to_tensor(value=low2, name='low2', dtype=dtype)
    high2 = tf.convert_to_tensor(value=high2, name='high2', dtype=dtype)
    false_fail_rate = tf.convert_to_tensor(
        value=false_fail_rate, name='false_fail_rate', dtype=dtype)
    false_pass_rate = tf.convert_to_tensor(
        value=false_pass_rate, name='false_pass_rate', dtype=dtype)
    det_disc1 = min_discrepancy_of_true_means_detectable_by_dkwm(
        n1, low1, high1, false_fail_rate / 2., false_pass_rate / 2.)
    det_disc2 = min_discrepancy_of_true_means_detectable_by_dkwm(
        n2, low2, high2, false_fail_rate / 2., false_pass_rate / 2.)
    return det_disc1 + det_disc2


def min_num_samples_for_dkwm_mean_two_sample_test(
    discrepancy, low1, high1, low2, high2,
    false_fail_rate=1e-6, false_pass_rate=1e-6, name=None):
  """Returns how many samples suffice for a two-sample DKWM mean test.

  DKWM is the [Dvoretzky-Kiefer-Wolfowitz-Massart inequality]
  (https://en.wikipedia.org/wiki/CDF-based_nonparametric_confidence_interval).

  Args:
    discrepancy: Floating-point `Tensor` of desired upper limits on mean
      differences that may go undetected with probability higher than
      `1 - false_pass_rate`.
    low1: Floating-point `Tensor` of lower bounds on the supports of the
      distributions A.
    high1: Floating-point `Tensor` of upper bounds on the supports of
      the distributions A.
    low2: Floating-point `Tensor` of lower bounds on the supports of the
      distributions B.
    high2: Floating-point `Tensor` of upper bounds on the supports of
      the distributions B.
    false_fail_rate: *Scalar* floating-point `Tensor` admissible total
      rate of false failures.
    false_pass_rate: *Scalar* floating-point `Tensor` admissible rate
      of false passes.
    name: A name for this operation (optional).

  Returns:
    n1: `Tensor` of numbers of samples to be drawn from the distributions A.
    n2: `Tensor` of numbers of samples to be drawn from the distributions B.

  For each batch member `i`, of `K` total, drawing `n1[i]` samples
  from scalar distribution A supported on `[low1[i], high1[i]]` and `n2[i]`
  samples from scalar distribution B supported on `[low2[i], high2[i]]`
  is enough to detect a difference in their true means of size
  `discr[i]` or more.  Specifically, we guarantee that (a) if their
  true means are equal, `assert_true_mean_equal_by_dkwm_two_sample`
  will fail with probability at most `false_fail_rate/K` (which
  amounts to `false_fail_rate` if applied to the whole batch at once),
  and (b) if their true means differ by at least `discr[i]`,
  `assert_true_mean_equal_by_dkwm_two_sample` will pass with
  probability at most `false_pass_rate`.

  The required number of samples scales as

  - `O((high1[i] - low1[i])**2)`, `O((high2[i] - low2[i])**2)`,
  - `O(-log(false_fail_rate/K))`,
  - `O(-log(false_pass_rate))`, and
  - `O(1 / discrepancy[i]**2)`.
  """
  args_list = (
      [low1, high1, low2, high2, false_fail_rate, false_pass_rate, discrepancy])
  with tf.compat.v1.name_scope(
      name, 'min_num_samples_for_dkwm_mean_two_sample_test', args_list):
    dtype = dtype_util.common_dtype(args_list, tf.float32)
    discrepancy = tf.convert_to_tensor(
        value=discrepancy, name='discrepancy', dtype=dtype)
    low1 = tf.convert_to_tensor(value=low1, name='low1', dtype=dtype)
    high1 = tf.convert_to_tensor(value=high1, name='high1', dtype=dtype)
    low2 = tf.convert_to_tensor(value=low2, name='low2', dtype=dtype)
    high2 = tf.convert_to_tensor(value=high2, name='high2', dtype=dtype)
    false_fail_rate = tf.convert_to_tensor(
        value=false_fail_rate, name='false_fail_rate', dtype=dtype)
    false_pass_rate = tf.convert_to_tensor(
        value=false_pass_rate, name='false_pass_rate', dtype=dtype)
    # Could choose to cleverly allocate discrepancy tolerances and
    # failure probabilities, but this is sound.
    n1 = min_num_samples_for_dkwm_mean_test(
        discrepancy / 2., low1, high1,
        false_fail_rate / 2., false_pass_rate / 2.)
    n2 = min_num_samples_for_dkwm_mean_test(
        discrepancy / 2., low2, high2,
        false_fail_rate / 2., false_pass_rate / 2.)
    return n1, n2


def _random_unit_hypersphere(sample_shape, event_shape, dtype, seed):
  target_shape = tf.concat([sample_shape, event_shape], axis=0)
  return tf.math.l2_normalize(
      tf.random.normal(target_shape, seed=seed, dtype=dtype),
      axis=-1 - tf.range(tf.size(input=event_shape)))


def assert_multivariate_true_cdf_equal_on_projections_two_sample(
    samples1, samples2, num_projections, event_ndims=1,
    false_fail_rate=1e-6, seed=17, name=None):
  """Asserts the given multivariate distributions are equal.

  The test is a 1-D DKWM-style test of equality in distribution along the given
  number of random projections.  This is of course imperfect, but can behave
  reasonably, especially if `num_projections` is significantly more than the
  dimensionality of the sample space.

  More precisely, the test
  (i) assumes the event shape is given by the trailing `event_ndims` dimensions
      in each `samples` Tensor;
  (ii) generates `num_projections` random projections from this space to scalar;
  (iii) fails if there is enough evidence (using the
      [Dvoretzky-Kiefer-Wolfowitz-Massart inequality]
      (https://en.wikipedia.org/wiki/CDF-based_nonparametric_confidence_interval))
      along any of these projections that `samples1` and `samples2` come from
      different true distributions.

  This test works as written even if the distribution in question has atoms
  (e.g., is discrete).

  Note that the top dimension of each `samples` is treated as iid, and the
  bottom `event_ndims` dimensions are projected to scalar.  The remaining
  dimensions, if any, are treated as batch dimensions, and `false_fail_rate` is
  a total false failure rate for all the assertions in the batch (and all
  projections).  As such, if the batch is nontrivial, the assertion will insist
  on stronger evidence to fail any one member.

  A note on experiment design: This test boils down to `num_projections`
  two-sample CDF equality tests.  As such, one can compute the number of samples
  to draw or the detectable discrepancy (along any projection) using
  `min_num_samples_for_dkwm_cdf_two_sample_test` and
  `min_discrepancy_of_true_cdfs_detectable_by_dkwm_two_sample` respectively,
  just being sure to divide the false failure rate by the number of projections
  requested (no need to adjust the false pass rate).

  Args:
    samples1: Tensor of shape [n] + B + E.  Samples from some (batch of)
      distribution(s) of interest.  Assumed IID across the 0 dimension.
    samples2: Tensor of shape [m] + B + E.  Samples from some (batch of)
      distribution(s) of interest.  Assumed IID across the 0 dimension.
    num_projections: Scalar integer Tensor.  Number of projections to use.  Each
      projection will be a random direction on the unit hypershpere of shape E.
    event_ndims: Number of trailing dimensions forming the event shape.
      `rank(E) == event_ndims`.
    false_fail_rate: *Scalar* floating-point `Tensor` admissible total
      rate of mistakes.
    seed: Optional PRNG seed to use for generating the random projections.
      Changing this from the default should generally not be necessary.
    name: A name for this operation (optional).

  Returns:
    check: Op that raises `InvalidArgumentError` if the two samples do not match
      along any of the generated projections.
  """
  # Shape of samples{1,2}: one iid sample dimension; arbitrary batch shape;
  # arbitrary event shape.  Batch and event shapes must agree.
  # The semantics of event shape is that the events are randomly projected to
  # scalar.
  # The test is done batched across the batch shape.
  # Notate the shape of samples1 as [n1] + batch_shape + event_shape.
  # Notate the shape of samples2 as [n2] + batch_shape + event_shape.
  args_list = (
      [samples1, samples2, num_projections, event_ndims, false_fail_rate])
  strm = seed_stream.SeedStream(salt='random projections', seed=seed)
  with tf.compat.v1.name_scope(
      name,
      'assert_multivariate_true_cdf_equal_on_projections_two_sample',
      args_list):
    dtype = dtype_util.common_dtype(
        [samples1, samples2, false_fail_rate], tf.float32)
    samples1 = tf.convert_to_tensor(
        value=samples1, name='samples1', dtype=dtype)
    samples2 = tf.convert_to_tensor(
        value=samples2, name='samples2', dtype=dtype)
    num_projections = tf.convert_to_tensor(
        value=num_projections, name='num_projections')
    false_fail_rate = tf.convert_to_tensor(
        value=false_fail_rate, name='false_fail_rate', dtype=dtype)
    tf.compat.v1.assert_scalar(false_fail_rate)  # Static shape
    compatible_samples = tf.compat.v1.assert_equal(
        tf.shape(input=samples1)[1:],
        tf.shape(input=samples2)[1:])
    with tf.control_dependencies([compatible_samples]):
      event_shape = tf.shape(input=samples1)[-event_ndims:]
      random_projections = _random_unit_hypersphere(
          [num_projections], event_shape, dtype=dtype, seed=strm())
      last_axes = list(range(-1, -(event_ndims+1), -1))
      # proj1 shape should be [n1] + batch_shape + [num_projections]
      proj1 = tf.tensordot(samples1, random_projections, [last_axes, last_axes])
      # proj2 shape should be [n2] + batch_shape + [num_projections]
      proj2 = tf.tensordot(samples2, random_projections, [last_axes, last_axes])
      return assert_true_cdf_equal_by_dkwm_two_sample(
          proj1, proj2, false_fail_rate=false_fail_rate)
