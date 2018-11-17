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
  check2 = tf.assert_less(
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
     difference the true means.

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

import itertools

import tensorflow as tf
from tensorflow_probability.python.internal import dtype_util

__all__ = [
    "true_mean_confidence_interval_by_dkwm",
    "assert_true_mean_equal_by_dkwm",
    "min_discrepancy_of_true_means_detectable_by_dkwm",
    "min_num_samples_for_dkwm_mean_test",
    "assert_true_mean_in_interval_by_dkwm",
    "assert_true_mean_equal_by_dkwm_two_sample",
    "min_discrepancy_of_true_means_detectable_by_dkwm_two_sample",
    "min_num_samples_for_dkwm_mean_two_sample_test",
]


def _batch_sort_vector(x, ascending=True, name=None):
  with tf.name_scope(name, "_batch_sort_vector", [x]):
    x = tf.convert_to_tensor(x, name="x")
    n = tf.shape(x)[-1]
    if ascending:
      y, _ = tf.nn.top_k(-x, k=n, sorted=True)
      y = -y
    else:
      y, _ = tf.nn.top_k(x, k=n, sorted=True)
    y.set_shape(x.shape)
    return y


def _do_maximum_mean(samples, envelope, high, name=None):
  """Common code between maximum_mean and minimum_mean."""
  with tf.name_scope(name, "do_maximum_mean", [samples, envelope, high]):
    dtype = dtype_util.common_dtype([samples, envelope, high], tf.float32)
    samples = tf.convert_to_tensor(samples, name="samples", dtype=dtype)
    envelope = tf.convert_to_tensor(envelope, name="envelope", dtype=dtype)
    high = tf.convert_to_tensor(high, name="high", dtype=dtype)
    n = tf.rank(samples)
    # Move the batch dimension of `samples` to the rightmost position,
    # where the _batch_sort_vector function wants it.
    perm = tf.concat([tf.range(1, n), [0]], axis=0)
    samples = tf.transpose(samples, perm)

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
    batch_size = tf.shape(samples)[-1]
    batch_size = tf.cast(batch_size, dtype=dtype)
    step = 1. / batch_size
    cum_steps = step * tf.range(1, batch_size + 1, dtype=dtype)
    max_mean_contrib = tf.clip_by_value(
        cum_steps - envelope[..., tf.newaxis],
        clip_value_min=0.,
        clip_value_max=step)
    return tf.reduce_sum(samples * max_mean_contrib, axis=-1) + envelope * high


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
  with tf.name_scope(name, "maximum_mean", [samples, envelope, high]):
    dtype = dtype_util.common_dtype([samples, envelope, high], tf.float32)
    samples = tf.convert_to_tensor(samples, name="samples", dtype=dtype)
    envelope = tf.convert_to_tensor(envelope, name="envelope", dtype=dtype)
    high = tf.convert_to_tensor(high, name="high", dtype=dtype)

    xmax = tf.reduce_max(samples, axis=[0])
    msg = "Given sample maximum value exceeds expectations"
    check_op = tf.assert_less_equal(xmax, high, message=msg)
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
  with tf.name_scope(name, "minimum_mean", [samples, envelope, low]):
    dtype = dtype_util.common_dtype([samples, envelope, low], tf.float32)
    samples = tf.convert_to_tensor(samples, name="samples", dtype=dtype)
    envelope = tf.convert_to_tensor(envelope, name="envelope", dtype=dtype)
    low = tf.convert_to_tensor(low, name="low", dtype=dtype)

    xmin = tf.reduce_min(samples, axis=[0])
    msg = "Given sample minimum value falls below expectations"
    check_op = tf.assert_greater_equal(xmin, low, message=msg)
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
  with tf.name_scope(name, "dkwm_cdf_envelope", [n, error_rate]):
    n = tf.cast(n, dtype=error_rate.dtype)
    return tf.sqrt(-tf.log(error_rate / 2.) / (2. * n))


def _check_shape_dominates(samples, parameters):
  """Check that broadcasting `samples` against `parameters` does not expand it.

  Why?  Because I want to be very sure that the samples tensor is not
  accidentally enlarged by broadcasting against tensors that are
  supposed to be describing the distribution(s) sampled from, lest the
  sample counts end up inflated.

  Args:
    samples: A `Tensor` whose shape is to be protected against broadcasting.
    parameters: A list of `Tensor`s who are parameters for the statistical test.

  Returns:
    samples: Return original `samples` with control dependencies attached
      to ensure no broadcasting.
  """
  def check(t):
    samples_batch_shape = tf.shape(samples)[1:]
    broadcasted_batch_shape = tf.broadcast_dynamic_shape(
        samples_batch_shape, tf.shape(t))
    # This rank check ensures that I don't get a wrong answer from the
    # _shapes_ broadcasting against each other.
    samples_batch_ndims = tf.size(samples_batch_shape)
    ge = tf.assert_greater_equal(samples_batch_ndims, tf.rank(t))
    eq = tf.assert_equal(samples_batch_shape, broadcasted_batch_shape)
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
  with tf.name_scope(name, "true_mean_confidence_interval_by_dkwm",
                     [samples, low, high, error_rate]):
    dtype = dtype_util.common_dtype(
        [samples, low, high, error_rate], tf.float32)
    samples = tf.convert_to_tensor(samples, name="samples", dtype=dtype)
    low = tf.convert_to_tensor(low, name="low", dtype=dtype)
    high = tf.convert_to_tensor(high, name="high", dtype=dtype)
    error_rate = tf.convert_to_tensor(
        error_rate, name="error_rate", dtype=dtype)
    samples = _check_shape_dominates(samples, [low, high])
    tf.assert_scalar(error_rate)  # Static shape
    error_rate = _itemwise_error_rate(error_rate, [low, high], samples)
    n = tf.shape(samples)[0]
    envelope = _dkwm_cdf_envelope(n, error_rate)
    min_mean = _minimum_mean(samples, envelope, low)
    max_mean = _maximum_mean(samples, envelope, high)
    return min_mean, max_mean


def _itemwise_error_rate(
    total_error_rate, param_tensors, sample_tensor=None, name=None):
  with tf.name_scope(name, "itemwise_error_rate",
                     [total_error_rate, param_tensors, sample_tensor]):
    result_shape = [1]
    for p_tensor in param_tensors:
      result_shape = tf.broadcast_dynamic_shape(
          tf.shape(p_tensor), result_shape)
    if sample_tensor is not None:
      result_shape = tf.broadcast_dynamic_shape(
          tf.shape(sample_tensor)[1:], result_shape)
    num_items = tf.reduce_prod(result_shape)
    return total_error_rate / tf.cast(num_items, dtype=total_error_rate.dtype)


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
  with tf.name_scope(name, "assert_true_mean_equal_by_dkwm",
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
  with tf.name_scope(name, "min_discrepancy_of_true_means_detectable_by_dkwm",
                     [n, low, high, false_fail_rate, false_pass_rate]):
    dtype = dtype_util.common_dtype(
        [n, low, high, false_fail_rate, false_pass_rate], tf.float32)
    n = tf.convert_to_tensor(n, name="n", dtype=dtype)
    low = tf.convert_to_tensor(low, name="low", dtype=dtype)
    high = tf.convert_to_tensor(high, name="high", dtype=dtype)
    false_fail_rate = tf.convert_to_tensor(
        false_fail_rate, name="false_fail_rate", dtype=dtype)
    false_pass_rate = tf.convert_to_tensor(
        false_pass_rate, name="false_pass_rate", dtype=dtype)
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
    false_fail_rate = _itemwise_error_rate(false_fail_rate, [n, low, high])
    analysis_envelope = _dkwm_cdf_envelope(n, false_fail_rate)
    return (high - low) * (sampling_envelope + analysis_envelope)


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
  with tf.name_scope(
      name, "min_num_samples_for_dkwm_mean_test",
      [low, high, false_fail_rate, false_pass_rate, discrepancy]):
    dtype = dtype_util.common_dtype(
        [low, high, false_fail_rate, false_pass_rate, discrepancy], tf.float32)
    discrepancy = tf.convert_to_tensor(
        discrepancy, name="discrepancy", dtype=dtype)
    low = tf.convert_to_tensor(low, name="low", dtype=dtype)
    high = tf.convert_to_tensor(high, name="high", dtype=dtype)
    false_fail_rate = tf.convert_to_tensor(
        false_fail_rate, name="false_fail_rate", dtype=dtype)
    false_pass_rate = tf.convert_to_tensor(
        false_pass_rate, name="false_pass_rate", dtype=dtype)
    # Could choose to cleverly allocate envelopes, but this is sound.
    envelope1 = discrepancy / (2. * (high - low))
    envelope2 = envelope1
    false_fail_rate = _itemwise_error_rate(
        false_fail_rate, [low, high, discrepancy])
    n1 = -tf.log(false_fail_rate / 2.) / (2. * envelope1**2)
    n2 = -tf.log(false_pass_rate / 2.) / (2. * envelope2**2)
    return tf.maximum(n1, n2)


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
  with tf.name_scope(
      name, "assert_true_mean_in_interval_by_dkwm", args_list):
    dtype = dtype_util.common_dtype(args_list, tf.float32)
    samples = tf.convert_to_tensor(samples, name="samples", dtype=dtype)
    low = tf.convert_to_tensor(low, name="low", dtype=dtype)
    high = tf.convert_to_tensor(high, name="high", dtype=dtype)
    expected_low = tf.convert_to_tensor(
        expected_low, name="expected_low", dtype=dtype)
    expected_high = tf.convert_to_tensor(
        expected_high, name="expected_high", dtype=dtype)
    false_fail_rate = tf.convert_to_tensor(
        false_fail_rate, name="false_fail_rate", dtype=dtype)
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
    check_confidence_interval_can_intersect = tf.assert_greater_equal(
        max_mean,
        expected_low,
        message="Confidence interval does not "
        "intersect: true mean smaller than expected")
    with tf.control_dependencies([check_confidence_interval_can_intersect]):
      return tf.assert_less_equal(
          min_mean,
          expected_high,
          message="Confidence interval does not "
          "intersect: true mean greater than expected")


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
  with tf.name_scope(
      name, "assert_true_mean_equal_by_dkwm_two_sample", args_list):
    dtype = dtype_util.common_dtype(args_list, tf.float32)
    samples1 = tf.convert_to_tensor(samples1, name="samples1", dtype=dtype)
    low1 = tf.convert_to_tensor(low1, name="low1", dtype=dtype)
    high1 = tf.convert_to_tensor(high1, name="high1", dtype=dtype)
    samples2 = tf.convert_to_tensor(samples2, name="samples2", dtype=dtype)
    low2 = tf.convert_to_tensor(low2, name="low2", dtype=dtype)
    high2 = tf.convert_to_tensor(high2, name="high2", dtype=dtype)
    false_fail_rate = tf.convert_to_tensor(
        false_fail_rate, name="false_fail_rate", dtype=dtype)
    samples1 = _check_shape_dominates(samples1, [low1, high1])
    samples2 = _check_shape_dominates(samples2, [low2, high2])
    compatible_samples = tf.assert_equal(
        tf.shape(samples1)[1:],
        tf.shape(samples2)[1:])
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
  with tf.name_scope(
      name,
      "min_discrepancy_of_true_means_detectable_by_dkwm_two_sample",
      args_list):
    dtype = dtype_util.common_dtype(args_list, tf.float32)
    n1 = tf.convert_to_tensor(n1, name="n1", dtype=dtype)
    low1 = tf.convert_to_tensor(low1, name="low1", dtype=dtype)
    high1 = tf.convert_to_tensor(high1, name="high1", dtype=dtype)
    n2 = tf.convert_to_tensor(n2, name="n2", dtype=dtype)
    low2 = tf.convert_to_tensor(low2, name="low2", dtype=dtype)
    high2 = tf.convert_to_tensor(high2, name="high2", dtype=dtype)
    false_fail_rate = tf.convert_to_tensor(
        false_fail_rate, name="false_fail_rate", dtype=dtype)
    false_pass_rate = tf.convert_to_tensor(
        false_pass_rate, name="false_pass_rate", dtype=dtype)
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
  with tf.name_scope(
      name,
      "min_num_samples_for_dkwm_mean_two_sample_test",
      args_list):
    dtype = dtype_util.common_dtype(args_list, tf.float32)
    discrepancy = tf.convert_to_tensor(
        discrepancy, name="discrepancy", dtype=dtype)
    low1 = tf.convert_to_tensor(low1, name="low1", dtype=dtype)
    high1 = tf.convert_to_tensor(high1, name="high1", dtype=dtype)
    low2 = tf.convert_to_tensor(low2, name="low2", dtype=dtype)
    high2 = tf.convert_to_tensor(high2, name="high2", dtype=dtype)
    false_fail_rate = tf.convert_to_tensor(
        false_fail_rate, name="false_fail_rate", dtype=dtype)
    false_pass_rate = tf.convert_to_tensor(
        false_pass_rate, name="false_pass_rate", dtype=dtype)
    # Could choose to cleverly allocate discrepancy tolerances and
    # failure probabilities, but this is sound.
    n1 = min_num_samples_for_dkwm_mean_test(
        discrepancy / 2., low1, high1,
        false_fail_rate / 2., false_pass_rate / 2.)
    n2 = min_num_samples_for_dkwm_mean_test(
        discrepancy / 2., low2, high2,
        false_fail_rate / 2., false_pass_rate / 2.)
    return n1, n2
