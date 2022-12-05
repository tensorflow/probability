# Copyright 2020 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the _License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Weighted resampling methods, e.g., for use in SMC methods."""

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import exponential
from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.internal import distribution_util as dist_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.math.generic import log_cumsum_exp
from tensorflow_probability.python.math.gradient import value_and_gradient
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

__all__ = [
    'resample',
    'resample_independent',
    'resample_deterministic_minimum_error',
    'resample_stratified',
    'resample_systematic',
]


def resample(particles, log_weights, resample_fn, target_log_weights=None,
             seed=None):
  """Resamples the current particles according to provided weights.

  Args:
    particles: Nested structure of `Tensor`s each of shape
      `[num_particles, b1, ..., bN, ...]`, where
      `b1, ..., bN` are optional batch dimensions.
    log_weights: float `Tensor` of shape `[num_particles, b1, ..., bN]`, where
      `b1, ..., bN` are optional batch dimensions.
    resample_fn: choose the function used for resampling.
      Use 'resample_independent' for independent resamples.
      Use 'resample_stratified' for stratified resampling.
      Use 'resample_systematic' for systematic resampling.
    target_log_weights: optional float `Tensor` of the same shape and dtype as
      `log_weights`, specifying the target measure on `particles` if this is
      different from that implied by normalizing `log_weights`. The
      returned `log_weights_after_resampling` will represent this measure. If
      `None`, the target measure is implicitly taken to be the normalized
      log weights (`log_weights - tf.reduce_logsumexp(log_weights, axis=0)`).
      Default value: `None`.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.

  Returns:
    resampled_particles: Nested structure of `Tensor`s, matching `particles`.
    resample_indices: int `Tensor` of shape `[num_particles, b1, ..., bN]`.
    log_weights_after_resampling: float `Tensor` of same shape and dtype as
      `log_weights`, such that weighted sums of the resampled particles are
      equal (in expectation over the resampling step) to weighted sums of
      the original particles:
      `E [ exp(log_weights_after_resampling) * some_fn(resampled_particles) ]
      == exp(target_log_weights) * some_fn(particles)`.
      If no `target_log_weights` was specified, the log weights after
      resampling are uniformly equal to `-log(num_particles)`.
  """
  with tf.name_scope('resample'):
    num_particles = ps.size0(log_weights)
    log_num_particles = tf.math.log(tf.cast(num_particles, log_weights.dtype))

    # Normalize the weights and sample the ancestral indices.
    log_probs = tf.math.log_softmax(log_weights, axis=0)
    resampled_indices = resample_fn(log_probs, num_particles, (), seed=seed)

    gather_ancestors = lambda x: (  # pylint: disable=g-long-lambda
        mcmc_util.index_remapping_gather(x, resampled_indices, axis=0))
    resampled_particles = tf.nest.map_structure(gather_ancestors, particles)
    if target_log_weights is None:
      log_weights_after_resampling = tf.fill(ps.shape(log_weights),
                                             -log_num_particles)
    else:
      importance_weights = target_log_weights - log_probs - log_num_particles
      log_weights_after_resampling = tf.nest.map_structure(
          gather_ancestors, importance_weights)
  return resampled_particles, resampled_indices, log_weights_after_resampling


# TODO(b/153689734): rewrite so as not to use `move_dimension`.
def _resample_using_log_points(log_probs, sample_shape, log_points, name=None):
  """Resample from `log_probs` using supplied points in interval `[0, 1]`."""

  # We divide up the unit interval [0, 1] according to the provided
  # probability distributions using `cumulative_logsumexp`.
  # At the end of each division we place a 'marker'.
  # We use points on the unit interval supplied by caller.
  # We sort the combination of points and markers. The number
  # of points between the markers defining a division gives the number
  # of samples we require in that division.
  # For example, suppose `probs` is `[0.2, 0.3, 0.5]`.
  # We divide up `[0, 1]` using 3 markers:
  #
  #     |     |          |
  # 0.  0.2   0.5        1.0  <- markers
  #
  # Suppose we are given four points: [0.1, 0.25, 0.9, 0.75]
  # After sorting the combination we get:
  #
  # 0.1  0.25     0.75 0.9    <- points
  #  *  | *   |    *    *|
  # 0.   0.2 0.5         1.0  <- markers
  #
  # We have one sample in the first category, one in the second and
  # two in the last.
  #
  # All of these computations are carried out in batched form.

  with tf.name_scope(name or 'resample_using_log_points') as name:
    points_shape = ps.shape(log_points)
    batch_shape, [num_markers] = ps.split(ps.shape(log_probs),
                                          num_or_size_splits=[-1, 1])

    # `working_shape` specifies the total number of events
    # we will be generating.
    working_shape = ps.concat([sample_shape, batch_shape], axis=0)
    # `markers_shape` is the shape of the markers we temporarily insert.
    markers_shape = ps.concat([working_shape, [num_markers]], axis=0)

    markers = ps.concat(
        [tf.ones(markers_shape, dtype=tf.int32),
         tf.zeros(points_shape, dtype=tf.int32)],
        axis=-1)
    log_marker_positions = tf.broadcast_to(
        log_cumsum_exp(log_probs, axis=-1),
        markers_shape)
    log_markers_and_points = ps.concat(
        [log_marker_positions, log_points], axis=-1)
    # Stable sort is used to ensure that no points get sorted between
    # markers that have zero distance between them. This ensures that
    # there will never be a sample drawn whose probability is intended
    # to be zero even when a point falls on the edge of the
    # corresponding zero-width bucket.
    indices = tf.argsort(log_markers_and_points, axis=-1, stable=True)
    sorted_markers = tf.gather_nd(
        markers,
        indices[..., tf.newaxis],
        batch_dims=(
            ps.rank_from_shape(sample_shape) +
            ps.rank_from_shape(batch_shape)))
    markers_and_samples = ps.cast(
        tf.cumsum(sorted_markers, axis=-1), dtype=tf.int32)
    markers_and_samples = tf.math.minimum(markers_and_samples,
                                          num_markers - np.int32(1))

    # Collect up samples, omitting markers.
    samples_mask = tf.equal(sorted_markers, 0)

    # The following block of code is equivalent to
    # `samples = markers_and_samples[samples_mask]` however boolean mask
    # indices are not supported by XLA.
    # Instead we use `argsort` to pick out the top `num_samples`
    # elements of `markers_and_samples` when sorted using `samples_mask`
    # as key.
    num_samples = points_shape[-1]
    sample_locations = tf.argsort(
        ps.cast(samples_mask, dtype=tf.int32),
        direction='DESCENDING',
        stable=True)
    samples = tf.gather_nd(
        markers_and_samples,
        sample_locations[..., :num_samples, tf.newaxis],
        batch_dims=(
            ps.rank_from_shape(sample_shape) +
            ps.rank_from_shape(batch_shape)))

    return tf.reshape(samples, points_shape)


# TODO(b/153689734): rewrite so as not to use `move_dimension`.
def resample_independent(log_probs, event_size, sample_shape,
                         seed=None, name=None):
  """Categorical resampler for sequential Monte Carlo.

  The return value from this function is similar to sampling with

  ```python
  expanded_sample_shape = tf.concat([[event_size], sample_shape]), axis=-1)
  tfd.Categorical(logits=log_probs).sample(expanded_sample_shape)`
  ```

  but with values sorted along the first axis. It can be considered to be
  sampling events made up of a length-`event_size` vector of draws from
  the `Categorical` distribution. For large input values this function should
  give better performance than using `Categorical`.
  The sortedness is an unintended side effect of the algorithm that is
  harmless in the context of simple SMC algorithms.

  This implementation is based on the algorithms in [Maskell et al. (2006)][1].
  It is also known as multinomial resampling as described in
  [Doucet et al. (2011)][2].

  Args:
    log_probs: A tensor-valued batch of discrete log probability distributions.
    event_size: the dimension of the vector considered a single draw.
    sample_shape: the `sample_shape` determining the number of draws.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
      Default value: None (i.e. no seed).
    name: Python `str` name for ops created by this method.
      Default value: `None` (i.e., `'resample_independent'`).

  Returns:
    resampled_indices: a tensor of samples.

  #### References

  [1]: S. Maskell, B. Alun-Jones and M. Macleod. A Single Instruction Multiple
       Data Particle Filter.
       In 2006 IEEE Nonlinear Statistical Signal Processing Workshop.
       http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf
  [2]: A. Doucet & A. M. Johansen. Tutorial on Particle Filtering and
       Smoothing: Fifteen Years Later
       In 2011 The Oxford Handbook of Nonlinear Filtering
       https://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf

  """
  with tf.name_scope(name or 'resample_independent') as name:
    log_probs = tf.convert_to_tensor(log_probs, dtype_hint=tf.float32)
    log_probs = dist_util.move_dimension(log_probs, source_idx=0, dest_idx=-1)
    points_shape = ps.concat([sample_shape,
                              ps.shape(log_probs)[:-1],
                              [event_size]], axis=0)
    log_points = -exponential.Exponential(
        rate=tf.constant(1.0, dtype=log_probs.dtype)).sample(
            points_shape, seed=seed)

    resampled = _resample_using_log_points(log_probs, sample_shape, log_points)
    return dist_util.move_dimension(resampled, source_idx=-1, dest_idx=0)


# TODO(b/153689734): rewrite so as not to use `move_dimension`.
def resample_systematic(log_probs, event_size, sample_shape,
                        seed=None, name=None):
  """A systematic resampler for sequential Monte Carlo.

  The value returned from this function is similar to sampling with
  ```python
  expanded_sample_shape = tf.concat([[event_size], sample_shape]), axis=-1)
  logits = dist_util.move_dimension(log_probs, source_idx=0, dest_idx=-1)
  tfd.Categorical(logits=logits).sample(expanded_sample_shape)
  ```
  but with values sorted along the first axis. It can be considered to be
  sampling events made up of a length-`event_size` vector of draws from
  the `Categorical` distribution. However, although the elements of
  this event have the appropriate marginal distribution, they are not
  independent of each other. Instead they are drawn using a stratified
  sampling method that in some sense reduces variance and is suitable for
  use with Sequential Monte Carlo algorithms as described in
  [Doucet et al. (2011)][2].
  The sortedness is an unintended side effect of the algorithm that is
  harmless in the context of simple SMC algorithms.

  This implementation is based on the algorithms in [Maskell et al. (2006)][1]
  where it is called minimum variance resampling.

  Args:
    log_probs: A tensor-valued batch of discrete log probability distributions.
      It is expected that those log probabilities are normalized along the
      first dimension (such that ``sum(exp(log_probs), axis=0) == 1``).
      The remaining dimensions are batch dimensions.
    event_size: the dimension of the vector considered a single draw.
    sample_shape: the `sample_shape` determining the number of draws.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
      Default value: None (i.e. no seed).
    name: Python `str` name for ops created by this method.
      Default value: `None` (i.e., `'resample_systematic'`).

  Returns:
    resampled_indices: a tensor of samples.

  #### References
  [1]: S. Maskell, B. Alun-Jones and M. Macleod. A Single Instruction Multiple
       Data Particle Filter.
       In 2006 IEEE Nonlinear Statistical Signal Processing Workshop.
       http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf
  [2]: A. Doucet & A. M. Johansen. Tutorial on Particle Filtering and
       Smoothing: Fifteen Years Later
       In 2011 The Oxford Handbook of Nonlinear Filtering
       https://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf

  """
  with tf.name_scope(name or 'resample_systematic') as name:
    log_probs = tf.convert_to_tensor(log_probs, dtype_hint=tf.float32)
    log_probs = dist_util.move_dimension(log_probs, source_idx=0, dest_idx=-1)
    working_shape = ps.concat([sample_shape,
                               ps.shape(log_probs)[:-1]], axis=0)
    points_shape = ps.concat([working_shape, [event_size]], axis=0)
    # Draw a single offset for each event.
    interval_width = ps.cast(1. / event_size, dtype=log_probs.dtype)
    offsets = uniform.Uniform(
        low=ps.cast(0., dtype=log_probs.dtype),
        high=interval_width).sample(
            working_shape, seed=seed)[..., tf.newaxis]
    even_spacing = ps.linspace(
        start=ps.cast(0., dtype=log_probs.dtype),
        stop=1 - interval_width,
        num=event_size) + offsets
    log_points = tf.broadcast_to(tf.math.log(even_spacing), points_shape)

    resampled = _resample_using_log_points(log_probs, sample_shape, log_points)
    return dist_util.move_dimension(resampled, source_idx=-1, dest_idx=0)


# TODO(b/153689734): rewrite so as not to use `move_dimension`.
def resample_stratified(log_probs, event_size, sample_shape,
                        seed=None, name=None):
  """Stratified resampler for sequential Monte Carlo.

  The value returned from this algorithm is similar to sampling with
  ```python
  expanded_sample_shape = tf.concat([[event_size], sample_shape]), axis=-1)
  tfd.Categorical(logits=log_probs).sample(expanded_sample_shape)`
  ```
  but with values sorted along the first axis. It can be considered to be
  sampling events made up of a length-`event_size` vector of draws from
  the `Categorical` distribution. However, although the elements of
  this event have the appropriate marginal distribution, they are not
  independent of each other. Instead they are drawn using a low variance
  stratified sampling method suitable for use with Sequential Monte
  Carlo algorithms.
  The sortedness is an unintended side effect of the algorithm that is
  harmless in the context of simple SMC algorithms.

  This function is based on Algorithm #1 in the paper
  [Maskell et al. (2006)][1].

  Args:
    log_probs: A tensor-valued batch of discrete log probability distributions.
    event_size: the dimension of the vector considered a single draw.
    sample_shape: the `sample_shape` determining the number of draws.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
      Default value: None (i.e. no seed).
    name: Python `str` name for ops created by this method.
      Default value: `None` (i.e., `'resample_independent'`).

  Returns:
    resampled_indices: a tensor of samples.

  #### References

  [1]: S. Maskell, B. Alun-Jones and M. Macleod. A Single Instruction Multiple
       Data Particle Filter.
       In 2006 IEEE Nonlinear Statistical Signal Processing Workshop.
       http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf

  """
  with tf.name_scope(name or 'resample_stratified') as name:
    log_probs = tf.convert_to_tensor(log_probs, dtype_hint=tf.float32)
    log_probs = dist_util.move_dimension(log_probs, source_idx=0, dest_idx=-1)
    points_shape = ps.concat([sample_shape,
                              ps.shape(log_probs)[:-1],
                              [event_size]], axis=0)
    # Draw an offset for every element of an event.
    interval_width = ps.cast(1. / event_size, dtype=log_probs.dtype)
    offsets = uniform.Uniform(low=ps.cast(0., dtype=log_probs.dtype),
                              high=interval_width).sample(
                                  points_shape, seed=seed)
    # The unit interval is divided into equal partitions and each point
    # is a random offset into a partition.
    even_spacing = tf.linspace(
        start=ps.cast(0., dtype=log_probs.dtype),
        stop=1 - interval_width,
        num=event_size) + offsets
    log_points = tf.math.log(even_spacing)

    resampled = _resample_using_log_points(log_probs, sample_shape, log_points)
    return dist_util.move_dimension(resampled, source_idx=-1, dest_idx=0)


# TODO(b/153199903): replace this function with `tf.scatter_nd` when
# it supports `batch_dims`.
def _scatter_nd_batch(indices, updates, shape, batch_dims=0):
  """A partial implementation of `scatter_nd` supporting `batch_dims`."""

  # `tf.scatter_nd` does not support a `batch_dims` argument.
  # Instead we use the gradient of `tf.gather_nd`.
  # From a purely mathematical perspective this works because
  # (if `tf.scatter_nd` supported `batch_dims`)
  # `gather_nd` and `scatter_nd` (with matching `indices`) are
  # adjoint linear operators and
  # the gradient w.r.t `x` of `dot(y, A(x))` is `adjoint(A)(y)`.
  #
  # Another perspective: back propagating through a "neural" network
  # containing a gather operation carries derivatives backwards through the
  # network, accumulating the derivatives in the locations that
  # were gathered from, ie. they are scattered.
  # If the network multiplies each gathered element by
  # some quantity, then the backwardly propagating derivatives are scaled
  # by this quantity before being scattered.
  # Combining this with the fact that`GradientTape.gradient`
  # starts back-propagation with derivatives equal to `1`, this allows us
  # to use the multipliers to determine the quantities scattered.
  #
  # However, derivatives are only supported for floating point types
  # so we 'tunnel' our types through the `float64` type.
  # So the implmentation is "partial" in the sense that it supports
  # data that can be losslessly converted to `tf.float64` and back.
  dtype = updates.dtype
  internal_dtype = tf.float64
  multipliers = ps.cast(updates, internal_dtype)

  def weighted_gathered(zeros):
    return multipliers * tf.gather_nd(zeros, indices, batch_dims=batch_dims)

  zeros = tf.zeros(shape, dtype=internal_dtype)
  _, grad = value_and_gradient(weighted_gathered, zeros)
  return ps.cast(grad, dtype=dtype)


def _finite_differences(sums):
  """The inverse of `tf.cumsum` with `axis=-1`."""
  return ps.concat(
      [sums[..., :1], sums[..., 1:] - sums[..., :-1]], axis=-1)


def _samples_from_counts(values, counts, total_number):
  """Construct sequences of values from tabulated numbers of counts."""

  extended_result_shape = ps.concat(
      [ps.shape(counts)[:-1],
       [total_number + 1]], axis=0)
  padded_counts = ps.concat(
      [ps.zeros_like(counts[..., :1]),
       counts[..., :-1]], axis=-1)
  edge_positions = ps.cumsum(padded_counts, axis=-1)

  # We need to scatter `values` into an array according to
  # the given `counts`.
  # Because the final result typically consists of sequences of samples
  # that are constant in blocks, we can scatter just the finite
  # differences of the values (which become the 'edges' of the blocks)
  # and then cumulatively sum them back up
  # at the end. (Reminiscent of run length encoding.)
  # Eg. suppose we have values = `[0, 2, 1]`
  # and counts = `[2, 3, 4]`
  # Then the output we require is `[0, 0, 2, 2, 2, 1, 1, 1, 1]`.
  # The finite differences of the input are:
  #   `[0, 2, -1]`.
  # The finite differences of the output are:
  #   `[0, 0, 2, 0, 0, -1, 0, 0, 0]`.
  # The latter is the former scattered into a larger array.
  #
  # So the algorithm is essentially
  # compute finite differences -> scatter -> undo finite differences
  edge_heights = _finite_differences(values)
  edges = _scatter_nd_batch(
      edge_positions[..., tf.newaxis],
      edge_heights,
      extended_result_shape,
      batch_dims=ps.rank_from_shape(ps.shape(counts)) - 1)

  result = tf.cumsum(edges, axis=-1)[..., :-1]
  return result


# TODO(b/153689734): rewrite so as not to use `move_dimension`.
def resample_deterministic_minimum_error(
    log_probs, event_size, sample_shape,
    seed=None, name='resample_deterministic_minimum_error'):
  """Deterministic minimum error resampler for sequential Monte Carlo.

    The return value of this function is similar to sampling with

    ```python
    expanded_sample_shape = tf.concat([sample_shape, [event_size]]), axis=-1)
    tfd.Categorical(logits=log_probs).sample(expanded_sample_shape)`
    ```

    but with values chosen deterministically so that the empirical distribution
    is as close as possible to the specified distribution.
    (Note that the empirical distribution can only exactly equal the requested
    distribution if multiplying every probability by `event_size` gives
    an integer. So in general this is a biased "sampler".)
    It is intended to provide a good representative sample, suitable for use
    with some Sequential Monte Carlo algorithms.

  This function is based on Algorithm #3 in [Maskell et al. (2006)][1].

  Args:
    log_probs: a tensor-valued batch of discrete log probability distributions.
    event_size: the dimension of the vector considered a single draw.
    sample_shape: the `sample_shape` determining the number of draws. Because
      this resampler is deterministic it simply replicates the draw you
      would get for `sample_shape=[1]`.
    seed: This argument is unused but is present so that this function shares
      its interface with the other resampling functions.
      Default value: None
    name: Python `str` name for ops created by this method.
      Default value: `None` (i.e., `'resample_deterministic_minimum_error'`).

  Returns:
    resampled_indices: a tensor of samples.

  #### References
  [1]: S. Maskell, B. Alun-Jones and M. Macleod. A Single Instruction Multiple
       Data Particle Filter.
       In 2006 IEEE Nonlinear Statistical Signal Processing Workshop.
       http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf

  """
  del seed

  with tf.name_scope(name or 'resample_deterministic_minimum_error'):
    sample_shape = tf.convert_to_tensor(sample_shape, dtype_hint=tf.int32)
    log_probs = dist_util.move_dimension(
        log_probs, source_idx=0, dest_idx=-1)
    probs = tf.math.exp(log_probs)
    prob_shape = ps.shape(probs)
    pdf_size = prob_shape[-1]
    # If we could draw fractional numbers of samples we would
    # choose `ideal_numbers` for the number of each element.
    ideal_numbers = event_size * probs
    # We approximate the ideal numbers by truncating to integers
    # and then repair the counts starting with the one with the
    # largest fractional error and working our way down.
    first_approximation = tf.floor(ideal_numbers)
    missing_fractions = ideal_numbers - first_approximation
    first_approximation = ps.cast(
        first_approximation, dtype=tf.int32)
    fraction_order = tf.argsort(missing_fractions, axis=-1)
    # We sort the integer parts and fractional parts together.
    batch_dims = ps.rank_from_shape(prob_shape) - 1
    first_approximation = tf.gather_nd(
        first_approximation,
        fraction_order[..., tf.newaxis],
        batch_dims=batch_dims)
    missing_fractions = tf.gather_nd(
        missing_fractions,
        fraction_order[..., tf.newaxis],
        batch_dims=batch_dims)
    sample_defect = event_size - tf.reduce_sum(
        first_approximation, axis=-1, keepdims=True)
    unpermuted = tf.broadcast_to(
        tf.range(pdf_size),
        prob_shape)
    increments = tf.cast(
        unpermuted >= pdf_size - sample_defect,
        dtype=first_approximation.dtype)
    counts = first_approximation + increments
    samples = _samples_from_counts(fraction_order, counts, event_size)
    result_shape = tf.concat([sample_shape,
                              prob_shape[:-1],
                              [event_size]], axis=0)
    # Replicate sample up to batch size.
    # TODO(dpiponi): rather than replicating, spread the "error" over
    # multiple samples with a minimum-discrepancy sequence.
    resampled = tf.broadcast_to(samples, result_shape)
    return dist_util.move_dimension(resampled, source_idx=-1, dest_idx=0)
