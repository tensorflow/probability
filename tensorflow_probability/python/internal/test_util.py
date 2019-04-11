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
"""Utilities for testing distributions and/or bijectors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports
from absl import flags
import hypothesis.strategies as hps
import numpy as np
import six

import tensorflow as tf

from tensorflow_probability.python.distributions import seed_stream
from tensorflow_probability.python.internal import dtype_util

__all__ = [
    'broadcasting_shapes',
    'derandomize_hypothesis',
    'test_seed',
    'test_seed_stream',
    'DiscreteScalarDistributionTestHelpers',
    'VectorDistributionTestHelpers',
]


def derandomize_hypothesis():
  # Use --test_env=TFP_DERANDOMIZE_HYPOTHESIS=0 to get random coverage.
  return os.environ.get('TFP_DERANDOMIZE_HYPOTHESIS', 1) in (0, '0')


FLAGS = flags.FLAGS

flags.DEFINE_bool('vary_seed', False,
                  ('Whether to vary the PRNG seed unpredictably.  '
                   'With --runs_per_test=N, produces N iid runs.'))

flags.DEFINE_string('fixed_seed', None,
                    ('PRNG seed to initialize every test with.  '
                     'Takes precedence over --vary-seed when both appear.'))


def _compute_rank_and_fullsize_reqd(draw, target_shape, current_shape, is_last):
  """Returns a param rank and a list of bools for full-size-required by axis.

  Args:
    draw: Hypothesis data sampler.
    target_shape: `tf.TensorShape`, the target broadcasted shape.
    current_shape: `tf.TensorShape`, the broadcasted shape of the shapes
      selected thus far. This is ignored for non-last shapes.
    is_last: bool indicator of whether this is the last shape (in which case, we
      must achieve the target shape).

  Returns:
    next_rank: Sampled rank for the next shape.
    force_fullsize_dim: `next_rank`-sized list of bool indicating whether the
      corresponding axis of the shape must be full-sized (True) or is allowed to
      be 1 (i.e., broadcast) (False).
  """
  target_rank = target_shape.ndims
  if is_last:
    # We must force full size dim on any mismatched axes, and proper rank.
    full_rank_current = tf.broadcast_static_shape(
        current_shape, tf.TensorShape([1] * target_rank))
    # Identify axes in which the target shape is not yet matched.
    axis_is_mismatched = [
        full_rank_current[i] != target_shape[i] for i in range(target_rank)
    ]
    min_rank = target_rank
    if current_shape.ndims == target_rank:
      # Current rank might be already correct, but we could have a case like
      # batch_shape=[4,3,2] and current_batch_shape=[4,1,2], in which case
      # we must have at least 2 axes on this param's batch shape.
      min_rank -= (axis_is_mismatched + [True]).index(True)
    next_rank = draw(
        hps.integers(min_value=min_rank, max_value=target_rank))
    # Get the last param_batch_rank (possibly 0!) items.
    force_fullsize_dim = axis_is_mismatched[target_rank - next_rank:]
  else:
    # There are remaining params to be drawn, so we will be able to force full
    # size axes on subsequent params.
    next_rank = draw(hps.integers(min_value=0, max_value=target_rank))
    force_fullsize_dim = [False] * next_rank
  return next_rank, force_fullsize_dim


@hps.composite
def broadcasting_shapes(draw, target_shape, n):
  """Draws a set of `n` shapes that broadcast to `target_shape`.

  For each shape we need to choose its rank, and whether or not each axis i is 1
  or target_shape[i]. This function chooses a set of `n` shapes that have
  possibly mismatched ranks, and possibly broadcasting axes, with the promise
  that the broadcast of the set of all shapes matches `target_shape`.

  Args:
    draw: Hypothesis sampler.
    target_shape: The target (fully-defined) batch shape.
    n: `int`, the number of shapes to draw.

  Returns:
    shapes: Sequence of `tf.TensorShape` such that the set of shapes broadcast
      to `target_shape`. The shapes are fully defined.
  """
  target_shape = tf.TensorShape(target_shape)
  target_rank = target_shape.ndims
  result = []
  current_shape = tf.TensorShape([])
  for is_last in [False] * (n-1) + [True]:
    next_rank, force_fullsize_dim = _compute_rank_and_fullsize_reqd(
        draw, target_shape, current_shape, is_last=is_last)

    # Get the last next_rank (possibly 0!) dimensions.
    next_shape = target_shape[target_rank - next_rank:].as_list()
    for i, force_fullsize in enumerate(force_fullsize_dim):
      if not force_fullsize and draw(hps.booleans()):
        # Choose to make this param broadcast against some other param.
        next_shape[i] = 1
    next_shape = tf.TensorShape(next_shape)
    current_shape = tf.broadcast_static_shape(current_shape, next_shape)
    result.append(next_shape)
  return result


def test_seed(hardcoded_seed=None, set_eager_seed=True):
  """Returns a command-line-controllable PRNG seed for unit tests.

  If your test will pass a seed to more than one operation, consider using
  `test_seed_stream` instead.

  When seeding unit-test PRNGs, we want:

  - The seed to be fixed to an arbitrary value most of the time, so the test
    doesn't flake even if its failure probability is noticeable.

  - To switch to different seeds per run when using --runs_per_test to measure
    the test's failure probability.

  - To set the seed to a specific value when reproducing a low-probability event
    (e.g., debugging a crash that only some seeds trigger).

  To those ends, this function returns 17, but respects the command line flags
  `--fixed_seed=<seed>` and `--vary-seed` (Boolean, default False).
  `--vary_seed` uses system entropy to produce unpredictable seeds.
  `--fixed_seed` takes precedence over `--vary_seed` when both are present.

  Note that TensorFlow graph mode operations tend to read seed state from two
  sources: a "graph-level seed" and an "op-level seed".  tf.test.TestCase will
  set the former to a fixed value per test, but in general it may be necessary
  to explicitly set both to ensure reproducibility.

  Args:
    hardcoded_seed: Optional Python value.  The seed to use instead of 17 if
      both the `--vary_seed` and `--fixed_seed` flags are unset.  This should
      usually be unnecessary, since a test should pass with any seed.
    set_eager_seed: Python bool.  If true (default), invoke `tf.set_random_seed`
      in Eager mode to get more reproducibility.  Should become unnecessary
      once b/68017812 is resolved.

  Returns:
    seed: 17, unless otherwise specified by arguments or command line flags.
  """
  if FLAGS.fixed_seed is not None:
    answer = int(FLAGS.fixed_seed)
  elif FLAGS.vary_seed:
    entropy = os.urandom(64)
    # Why does Python make it so hard to just grab a bunch of bytes from
    # /dev/urandom and get them interpreted as an integer?  Oh, well.
    if six.PY2:
      answer = int(entropy.encode('hex'), 16)
    else:
      answer = int.from_bytes(entropy, 'big')
    tf.compat.v1.logging.warning('Using seed {}'.format(answer))
  elif hardcoded_seed is not None:
    answer = hardcoded_seed
  else:
    answer = 17
  # TODO(b/68017812): Remove this clause once eager correctly supports seeding.
  if tf.executing_eagerly() and set_eager_seed:
    tf.compat.v1.set_random_seed(answer)
  return answer


def test_seed_stream(salt='Salt of the Earth', hardcoded_seed=None):
  """Returns a command-line-controllable SeedStream PRNG for unit tests.

  When seeding unit-test PRNGs, we want:

  - The seed to be fixed to an arbitrary value most of the time, so the test
    doesn't flake even if its failure probability is noticeable.

  - To switch to different seeds per run when using --runs_per_test to measure
    the test's failure probability.

  - To set the seed to a specific value when reproducing a low-probability event
    (e.g., debugging a crash that only some seeds trigger).

  To those ends, this function returns a `SeedStream` seeded with `test_seed`
  (which see).  The latter respects the command line flags `--fixed_seed=<seed>`
  and `--vary-seed` (Boolean, default False).  `--vary_seed` uses system entropy
  to produce unpredictable seeds.  `--fixed_seed` takes precedence over
  `--vary_seed` when both are present.

  Note that TensorFlow graph mode operations tend to read seed state from two
  sources: a "graph-level seed" and an "op-level seed".  tf.test.TestCase will
  set the former to a fixed value per test, but in general it may be necessary
  to explicitly set both to ensure reproducibility.

  Args:
    salt: Optional string wherewith to salt the returned SeedStream.  Setting
      this guarantees independent random numbers across tests.
    hardcoded_seed: Optional Python value.  The seed to use if both the
      `--vary_seed` and `--fixed_seed` flags are unset.  This should usually be
      unnecessary, since a test should pass with any seed.

  Returns:
    strm: A SeedStream instance seeded with 17, unless otherwise specified by
      arguments or command line flags.
  """
  return seed_stream.SeedStream(salt, test_seed(hardcoded_seed))


class DiscreteScalarDistributionTestHelpers(object):
  """DiscreteScalarDistributionTestHelpers."""

  def run_test_sample_consistent_log_prob(
      self, sess_run_fn, dist,
      num_samples=int(1e5), num_threshold=int(1e3), seed=42,
      batch_size=None,
      rtol=1e-2, atol=0.):
    """Tests that sample/log_prob are consistent with each other.

    "Consistency" means that `sample` and `log_prob` correspond to the same
    distribution.

    Note: this test only verifies a necessary condition for consistency--it does
    does not verify sufficiency hence does not prove `sample`, `log_prob` truly
    are consistent.

    Args:
      sess_run_fn: Python `callable` taking `list`-like of `Tensor`s and
        returning a list of results after running one "step" of TensorFlow
        computation, typically set to `sess.run`.
      dist: Distribution instance or object which implements `sample`,
        `log_prob`, `event_shape_tensor` and `batch_shape_tensor`.
      num_samples: Python `int` scalar indicating the number of Monte-Carlo
        samples to draw from `dist`.
      num_threshold: Python `int` scalar indicating the number of samples a
        bucket must contain before being compared to the probability.
        Default value: 1e3; must be at least 1.
        Warning, set too high will cause test to falsely pass but setting too
        low will cause the test to falsely fail.
      seed: Python `int` indicating the seed to use when sampling from `dist`.
        In general it is not recommended to use `None` during a test as this
        increases the likelihood of spurious test failure.
      batch_size: Hint for unpacking result of samples. Default: `None` means
        batch_size is inferred.
      rtol: Python `float`-type indicating the admissible relative error between
        analytical and sample statistics.
      atol: Python `float`-type indicating the admissible absolute error between
        analytical and sample statistics.

    Raises:
      ValueError: if `num_threshold < 1`.
    """
    if num_threshold < 1:
      raise ValueError('num_threshold({}) must be at least 1.'.format(
          num_threshold))
    # Histogram only supports vectors so we call it once per batch coordinate.
    y = dist.sample(num_samples, seed=seed)
    y = tf.reshape(y, shape=[num_samples, -1])
    if batch_size is None:
      batch_size = tf.reduce_prod(input_tensor=dist.batch_shape_tensor())
    batch_dims = tf.shape(input=dist.batch_shape_tensor())[0]
    edges_expanded_shape = 1 + tf.pad(tensor=[-2], paddings=[[0, batch_dims]])
    for b, x in enumerate(tf.unstack(y, num=batch_size, axis=1)):
      counts, edges = self.histogram(x)
      edges = tf.reshape(edges, edges_expanded_shape)
      probs = tf.exp(dist.log_prob(edges))
      probs = tf.reshape(probs, shape=[-1, batch_size])[:, b]

      [counts_, probs_] = sess_run_fn([counts, probs])
      valid = counts_ > num_threshold
      probs_ = probs_[valid]
      counts_ = counts_[valid]
      self.assertAllClose(probs_, counts_ / num_samples,
                          rtol=rtol, atol=atol)

  def run_test_sample_consistent_mean_variance(
      self, sess_run_fn, dist,
      num_samples=int(1e5), seed=24,
      rtol=1e-2, atol=0.):
    """Tests that sample/mean/variance are consistent with each other.

    "Consistency" means that `sample`, `mean`, `variance`, etc all correspond
    to the same distribution.

    Args:
      sess_run_fn: Python `callable` taking `list`-like of `Tensor`s and
        returning a list of results after running one "step" of TensorFlow
        computation, typically set to `sess.run`.
      dist: Distribution instance or object which implements `sample`,
        `log_prob`, `event_shape_tensor` and `batch_shape_tensor`.
      num_samples: Python `int` scalar indicating the number of Monte-Carlo
        samples to draw from `dist`.
      seed: Python `int` indicating the seed to use when sampling from `dist`.
        In general it is not recommended to use `None` during a test as this
        increases the likelihood of spurious test failure.
      rtol: Python `float`-type indicating the admissible relative error between
        analytical and sample statistics.
      atol: Python `float`-type indicating the admissible absolute error between
        analytical and sample statistics.
    """
    x = tf.cast(dist.sample(num_samples, seed=seed), dtype=tf.float32)
    sample_mean = tf.reduce_mean(input_tensor=x, axis=0)
    sample_variance = tf.reduce_mean(
        input_tensor=tf.square(x - sample_mean), axis=0)
    sample_stddev = tf.sqrt(sample_variance)

    [
        sample_mean_,
        sample_variance_,
        sample_stddev_,
        mean_,
        variance_,
        stddev_
    ] = sess_run_fn([
        sample_mean,
        sample_variance,
        sample_stddev,
        dist.mean(),
        dist.variance(),
        dist.stddev(),
    ])

    self.assertAllClose(mean_, sample_mean_, rtol=rtol, atol=atol)
    self.assertAllClose(variance_, sample_variance_, rtol=rtol, atol=atol)
    self.assertAllClose(stddev_, sample_stddev_, rtol=rtol, atol=atol)

  def histogram(self, x, value_range=None, nbins=None, name=None):
    """Return histogram of values.

    Given the tensor `values`, this operation returns a rank 1 histogram
    counting the number of entries in `values` that fell into every bin. The
    bins are equal width and determined by the arguments `value_range` and
    `nbins`.

    Args:
      x: 1D numeric `Tensor` of items to count.
      value_range:  Shape [2] `Tensor`. `new_values <= value_range[0]` will be
        mapped to `hist[0]`, `values >= value_range[1]` will be mapped to
        `hist[-1]`. Must be same dtype as `x`.
      nbins:  Scalar `int32 Tensor`.  Number of histogram bins.
      name: Python `str` name prefixed to Ops created by this class.

    Returns:
      counts: 1D `Tensor` of counts, i.e.,
        `counts[i] = sum{ edges[i-1] <= values[j] < edges[i] : j }`.
      edges: 1D `Tensor` characterizing intervals used for counting.
    """
    with tf.compat.v2.name_scope(name or 'histogram'):
      x = tf.convert_to_tensor(value=x, name='x')
      if value_range is None:
        value_range = [
            tf.reduce_min(input_tensor=x), 1 + tf.reduce_max(input_tensor=x)
        ]
      value_range = tf.convert_to_tensor(value=value_range, name='value_range')
      lo = value_range[0]
      hi = value_range[1]
      if nbins is None:
        nbins = tf.cast(hi - lo, dtype=tf.int32)
      delta = (hi - lo) / tf.cast(
          nbins, dtype=dtype_util.base_dtype(value_range.dtype))
      edges = tf.range(
          start=lo, limit=hi, delta=delta, dtype=dtype_util.base_dtype(x.dtype))
      counts = tf.histogram_fixed_width(x, value_range=value_range, nbins=nbins)
      return counts, edges


class VectorDistributionTestHelpers(object):
  """VectorDistributionTestHelpers helps test vector-event distributions."""

  def run_test_sample_consistent_log_prob(
      self,
      sess_run_fn,
      dist,
      num_samples=int(1e5),
      radius=1.,
      center=0.,
      seed=42,
      rtol=1e-2,
      atol=0.):
    """Tests that sample/log_prob are mutually consistent.

    "Consistency" means that `sample` and `log_prob` correspond to the same
    distribution.

    The idea of this test is to compute the Monte-Carlo estimate of the volume
    enclosed by a hypersphere, i.e., the volume of an `n`-ball. While we could
    choose an arbitrary function to integrate, the hypersphere's volume is nice
    because it is intuitive, has an easy analytical expression, and works for
    `dimensions > 1`.

    Technical Details:

    Observe that:

    ```none
    int_{R**d} dx [x in Ball(radius=r, center=c)]
    = E_{p(X)}[ [X in Ball(r, c)] / p(X) ]
    = lim_{m->infty} m**-1 sum_j^m [x[j] in Ball(r, c)] / p(x[j]),
        where x[j] ~iid p(X)
    ```

    Thus, for fixed `m`, the above is approximately true when `sample` and
    `log_prob` are mutually consistent.

    Furthermore, the above calculation has the analytical result:
    `pi**(d/2) r**d / Gamma(1 + d/2)`.

    Note: this test only verifies a necessary condition for consistency--it does
    does not verify sufficiency hence does not prove `sample`, `log_prob` truly
    are consistent. For this reason we recommend testing several different
    hyperspheres (assuming the hypersphere is supported by the distribution).
    Furthermore, we gain additional trust in this test when also tested `sample`
    against the first, second moments
    (`run_test_sample_consistent_mean_covariance`); it is probably unlikely that
    a "best-effort" implementation of `log_prob` would incorrectly pass both
    tests and for different hyperspheres.

    For a discussion on the analytical result (second-line) see:
      https://en.wikipedia.org/wiki/Volume_of_an_n-ball.

    For a discussion of importance sampling (fourth-line) see:
      https://en.wikipedia.org/wiki/Importance_sampling.

    Args:
      sess_run_fn: Python `callable` taking `list`-like of `Tensor`s and
        returning a list of results after running one "step" of TensorFlow
        computation, typically set to `sess.run`.
      dist: Distribution instance or object which implements `sample`,
        `log_prob`, `event_shape_tensor` and `batch_shape_tensor`. The
        distribution must have non-zero probability of sampling every point
        enclosed by the hypersphere.
      num_samples: Python `int` scalar indicating the number of Monte-Carlo
        samples to draw from `dist`.
      radius: Python `float`-type indicating the radius of the `n`-ball which
        we're computing the volume.
      center: Python floating-type vector (or scalar) indicating the center of
        the `n`-ball which we're computing the volume. When scalar, the value is
        broadcast to all event dims.
      seed: Python `int` indicating the seed to use when sampling from `dist`.
        In general it is not recommended to use `None` during a test as this
        increases the likelihood of spurious test failure.
      rtol: Python `float`-type indicating the admissible relative error between
        actual- and approximate-volumes.
      atol: Python `float`-type indicating the admissible absolute error between
        actual- and approximate-volumes. In general this should be zero since
        a typical radius implies a non-zero volume.
    """

    def actual_hypersphere_volume(dims, radius):
      # https://en.wikipedia.org/wiki/Volume_of_an_n-ball
      # Using tf.lgamma because we'd have to otherwise use SciPy which is not
      # a required dependency of core.
      radius = np.asarray(radius)
      dims = tf.cast(dims, dtype=radius.dtype)
      return tf.exp((dims / 2.) * np.log(np.pi) -
                    tf.math.lgamma(1. + dims / 2.) + dims * tf.math.log(radius))

    def monte_carlo_hypersphere_volume(dist, num_samples, radius, center):
      # https://en.wikipedia.org/wiki/Importance_sampling
      x = dist.sample(num_samples, seed=seed)
      x = tf.identity(x)  # Invalidate bijector cacheing.
      inverse_log_prob = tf.exp(-dist.log_prob(x))
      importance_weights = tf.where(
          tf.norm(tensor=x - center, axis=-1) <= radius, inverse_log_prob,
          tf.zeros_like(inverse_log_prob))
      return tf.reduce_mean(input_tensor=importance_weights, axis=0)

    # Build graph.
    with tf.compat.v2.name_scope('run_test_sample_consistent_log_prob'):
      batch_shape = dist.batch_shape_tensor()
      actual_volume = actual_hypersphere_volume(
          dims=dist.event_shape_tensor()[0],
          radius=radius)
      sample_volume = monte_carlo_hypersphere_volume(
          dist,
          num_samples=num_samples,
          radius=radius,
          center=center)
      init_op = tf.compat.v1.global_variables_initializer()

    # Execute graph.
    sess_run_fn(init_op)
    [batch_shape_, actual_volume_, sample_volume_] = sess_run_fn([
        batch_shape, actual_volume, sample_volume])

    # Check results.
    self.assertAllClose(np.tile(actual_volume_, reps=batch_shape_),
                        sample_volume_,
                        rtol=rtol, atol=atol)

  def run_test_sample_consistent_mean_covariance(
      self,
      sess_run_fn,
      dist,
      num_samples=int(1e5),
      seed=24,
      rtol=1e-2,
      atol=0.1,
      cov_rtol=None,
      cov_atol=None):
    """Tests that sample/mean/covariance are consistent with each other.

    "Consistency" means that `sample`, `mean`, `covariance`, etc all correspond
    to the same distribution.

    Args:
      sess_run_fn: Python `callable` taking `list`-like of `Tensor`s and
        returning a list of results after running one "step" of TensorFlow
        computation, typically set to `sess.run`.
      dist: Distribution instance or object which implements `sample`,
        `log_prob`, `event_shape_tensor` and `batch_shape_tensor`.
      num_samples: Python `int` scalar indicating the number of Monte-Carlo
        samples to draw from `dist`.
      seed: Python `int` indicating the seed to use when sampling from `dist`.
        In general it is not recommended to use `None` during a test as this
        increases the likelihood of spurious test failure.
      rtol: Python `float`-type indicating the admissible relative error between
        analytical and sample statistics.
      atol: Python `float`-type indicating the admissible absolute error between
        analytical and sample statistics.
      cov_rtol: Python `float`-type indicating the admissible relative error
        between analytical and sample covariance. Default: rtol.
      cov_atol: Python `float`-type indicating the admissible absolute error
        between analytical and sample covariance. Default: atol.
    """

    x = dist.sample(num_samples, seed=seed)
    sample_mean = tf.reduce_mean(input_tensor=x, axis=0)
    sample_covariance = tf.reduce_mean(
        input_tensor=_vec_outer_square(x - sample_mean), axis=0)
    sample_variance = tf.linalg.diag_part(sample_covariance)
    sample_stddev = tf.sqrt(sample_variance)

    [
        sample_mean_,
        sample_covariance_,
        sample_variance_,
        sample_stddev_,
        mean_,
        covariance_,
        variance_,
        stddev_
    ] = sess_run_fn([
        sample_mean,
        sample_covariance,
        sample_variance,
        sample_stddev,
        dist.mean(),
        dist.covariance(),
        dist.variance(),
        dist.stddev(),
    ])

    self.assertAllClose(mean_, sample_mean_, rtol=rtol, atol=atol)
    self.assertAllClose(covariance_, sample_covariance_,
                        rtol=cov_rtol or rtol,
                        atol=cov_atol or atol)
    self.assertAllClose(variance_, sample_variance_, rtol=rtol, atol=atol)
    self.assertAllClose(stddev_, sample_stddev_, rtol=rtol, atol=atol)


def _vec_outer_square(x, name=None):
  """Computes the outer-product of a vector, i.e., x.T x."""
  with tf.compat.v2.name_scope(name or 'vec_osquare'):
    return x[..., :, tf.newaxis] * x[..., tf.newaxis, :]
