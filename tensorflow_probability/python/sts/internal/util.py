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
"""Structural Time Series utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf

from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.distributions.mvn_linear_operator import MultivariateNormalLinearOperator
from tensorflow_probability.python.internal import distribution_util as dist_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.sts.internal import missing_values_util

tfl = tf.linalg


def broadcast_batch_shape(distributions):
  """Get broadcast batch shape from distributions, statically if possible."""

  # Static case
  batch_shape = distributions[0].batch_shape
  for distribution in distributions:
    batch_shape = tf.broadcast_static_shape(batch_shape,
                                            distribution.batch_shape)
  if batch_shape.is_fully_defined():
    return batch_shape.as_list()

  # Fallback on dynamic.
  batch_shape = distributions[0].batch_shape_tensor()
  for distribution in distributions:
    batch_shape = tf.broadcast_dynamic_shape(batch_shape,
                                             distribution.batch_shape_tensor())

  return tf.convert_to_tensor(value=batch_shape)


def pad_batch_dimension_for_multiple_chains(
    observed_time_series, model, chain_batch_shape):
  """"Expand the observed time series with extra batch dimension(s)."""
  # Running with multiple chains introduces an extra batch dimension. In
  # general we also need to pad the observed time series with a matching batch
  # dimension.
  #
  # For example, suppose our model has batch shape [3, 4] and
  # the observed time series has shape `concat([[5], [3, 4], [100])`,
  # corresponding to `sample_shape`, `batch_shape`, and `num_timesteps`
  # respectively. The model will produce distributions with batch shape
  # `concat([chain_batch_shape, [3, 4]])`, so we pad `observed_time_series` to
  # have matching shape `[5, 1, 3, 4, 100]`, where the added `1` dimension
  # between the sample and batch shapes will broadcast to `chain_batch_shape`.

  [  # Extract mask and guarantee `event_ndims=2`.
      observed_time_series,
      is_missing
  ] = canonicalize_observed_time_series_with_mask(observed_time_series)

  event_ndims = 2  # event_shape = [num_timesteps, observation_size=1]

  model_batch_ndims = (
      model.batch_shape.ndims if model.batch_shape.ndims is not None else
      tf.shape(input=model.batch_shape_tensor())[0])

  # Compute ndims from chain_batch_shape.
  chain_batch_shape = tf.convert_to_tensor(
      value=chain_batch_shape, name='chain_batch_shape', dtype=tf.int32)
  if not chain_batch_shape.shape.is_fully_defined():
    raise ValueError('Batch shape must have static rank. (given: {})'.format(
        chain_batch_shape))
  if chain_batch_shape.shape.ndims == 0:  # expand int `k` to `[k]`.
    chain_batch_shape = chain_batch_shape[tf.newaxis]
  chain_batch_ndims = tf.compat.dimension_value(chain_batch_shape.shape[0])

  def do_padding(observed_time_series_tensor):
    current_sample_shape = tf.shape(
        input=observed_time_series_tensor)[:-(model_batch_ndims + event_ndims)]
    current_batch_and_event_shape = tf.shape(
        input=observed_time_series_tensor)[-(model_batch_ndims + event_ndims):]
    return tf.reshape(
        tensor=observed_time_series_tensor,
        shape=tf.concat([
            current_sample_shape,
            tf.ones([chain_batch_ndims], dtype=tf.int32),
            current_batch_and_event_shape], axis=0))

  # Padding is only needed if the observed time series has sample shape.
  observed_time_series = prefer_static.cond(
      (dist_util.prefer_static_rank(observed_time_series) >
       model_batch_ndims + event_ndims),
      lambda: do_padding(observed_time_series),
      lambda: observed_time_series)

  if is_missing is not None:
    is_missing = prefer_static.cond(
        (dist_util.prefer_static_rank(is_missing) >
         model_batch_ndims + event_ndims),
        lambda: do_padding(is_missing),
        lambda: is_missing)
    return missing_values_util.MaskedTimeSeries(observed_time_series,
                                                is_missing=is_missing)

  return observed_time_series


def factored_joint_mvn(distributions):
  """Combine MultivariateNormals into a factored joint distribution.

   Given a list of multivariate normal distributions
   `dist[i] = Normal(loc[i], scale[i])`, construct the joint
   distribution given by concatenating independent samples from these
   distributions. This is multivariate normal with mean vector given by the
   concatenation of the component mean vectors, and block-diagonal covariance
   matrix in which the blocks are the component covariances.

   Note that for computational efficiency, multivariate normals are represented
   by a 'scale' (factored covariance) linear operator rather than the full
   covariance matrix.

  Args:
    distributions: Python `iterable` of MultivariateNormal distribution
      instances (e.g., `tfd.MultivariateNormalDiag`,
      `tfd.MultivariateNormalTriL`, etc.). These must be broadcastable to a
      consistent batch shape, but may have different event shapes
      (i.e., defined over spaces of different dimension).

  Returns:
    joint_distribution: An instance of `tfd.MultivariateNormalLinearOperator`
      representing the joint distribution constructed by concatenating
      an independent sample from each input distributions.
  """

  graph_parents = [tensor for distribution in distributions
                   for tensor in distribution._graph_parents]  # pylint: disable=protected-access
  with tf.compat.v1.name_scope('factored_joint_mvn', values=graph_parents):

    # We explicitly broadcast the `locs` so that we can concatenate them.
    # We don't have direct numerical access to the `scales`, which are arbitrary
    # linear operators, but `LinearOperatorBlockDiag` appears to do the right
    # thing without further intervention.
    dtype = tf.debugging.assert_same_float_dtype(distributions)
    broadcast_ones = tf.ones(broadcast_batch_shape(distributions),
                             dtype=dtype)[..., tf.newaxis]
    return MultivariateNormalLinearOperator(
        loc=tf.concat([mvn.mean() * broadcast_ones for mvn in distributions],
                      axis=-1),
        scale=tfl.LinearOperatorBlockDiag([mvn.scale for mvn in distributions],
                                          is_square=True))


def sum_mvns(distributions):
  """Attempt to sum MultivariateNormal distributions.

  The sum of (multivariate) normal random variables is itself (multivariate)
  normal, with mean given by the sum of means and (co)variance given by the
  sum of (co)variances. This method exploits this fact to compute the
  sum of a list of `tfd.MultivariateNormalDiag` objects.

  It may in the future be extended to support summation of other forms of
  (Multivariate)Normal distributions.

  Args:
    distributions: Python `iterable` of `tfd.MultivariateNormalDiag`
      distribution instances. These must all have the same event
      shape, and broadcast to a consistent batch shape.

  Returns:
    sum_distribution: A `tfd.MultivariateNormalDiag` instance with mean
      equal to the sum of input means and covariance equal to the sum of
      input covariances.
  """

  graph_parents = [tensor for distribution in distributions
                   for tensor in distribution._graph_parents]  # pylint: disable=protected-access
  with tf.compat.v1.name_scope('sum_mvns', values=graph_parents):
    if all([isinstance(mvn, tfd.MultivariateNormalDiag)
            for mvn in distributions]):
      return tfd.MultivariateNormalDiag(
          loc=sum([mvn.mean() for mvn in distributions]),
          scale_diag=tf.sqrt(sum([
              mvn.scale.diag**2 for mvn in distributions])))
    else:
      raise NotImplementedError(
          'Sums of distributions other than MultivariateNormalDiag are not '
          'currently implemented. (given: {})'.format(distributions))


def empirical_statistics(observed_time_series):
  """Compute statistics of a provided time series, as heuristic initialization.

  Args:
    observed_time_series: `Tensor` representing a time series, or batch of time
       series, of shape either `batch_shape + [num_timesteps, 1]` or
       `batch_shape + [num_timesteps]` (allowed if `num_timesteps > 1`).

  Returns:
    observed_mean: `Tensor` of shape `batch_shape`, giving the empirical
      mean of each time series in the batch.
    observed_stddev: `Tensor` of shape `batch_shape`, giving the empirical
      standard deviation of each time series in the batch.
    observed_initial_centered: `Tensor of shape `batch_shape`, giving the
      initial value of each time series in the batch after centering
      (subtracting the mean).
  """

  with tf.compat.v1.name_scope(
      'empirical_statistics', values=[observed_time_series]):

    [
        observed_time_series,
        mask
    ] = canonicalize_observed_time_series_with_mask(observed_time_series)

    squeezed_series = observed_time_series[..., 0]
    if mask is None:
      observed_mean, observed_variance = tf.nn.moments(
          x=squeezed_series, axes=-1)
      observed_initial = squeezed_series[..., 0]
    else:
      broadcast_mask = tf.broadcast_to(tf.cast(mask, tf.bool),
                                       tf.shape(input=squeezed_series))
      observed_mean, observed_variance = (
          missing_values_util.moments_of_masked_time_series(
              squeezed_series, broadcast_mask=broadcast_mask))
      try:
        observed_initial = (
            missing_values_util.initial_value_of_masked_time_series(
                squeezed_series, broadcast_mask=broadcast_mask))
      except NotImplementedError:
        tf.compat.v1.logging.warn(
            'Cannot compute initial values for a masked time series'
            'with dynamic shape; using the mean instead. This will'
            'affect heuristic priors and may change the results of'
            'inference.')
        observed_initial = observed_mean

    observed_stddev = tf.sqrt(observed_variance)
    observed_initial_centered = observed_initial - observed_mean
    return observed_mean, observed_stddev, observed_initial_centered


def _maybe_expand_trailing_dim(observed_time_series_tensor):
  """Ensures `observed_time_series_tensor` has a trailing dimension of size 1.

  The `tfd.LinearGaussianStateSpaceModel` Distribution has event shape of
  `[num_timesteps, observation_size]`, but canonical BSTS models
  are univariate, so their observation_size is always `1`. The extra trailing
  dimension gets annoying, so this method allows arguments with or without the
  extra dimension. There is no ambiguity except in the trivial special case
  where  `num_timesteps = 1`; this can be avoided by specifying any unit-length
  series in the explicit `[num_timesteps, 1]` style.

  Most users should not call this method directly, and instead call
  `canonicalize_observed_time_series_with_mask`, which handles converting
  to `Tensor` and specifying an optional missingness mask.

  Args:
    observed_time_series_tensor: `Tensor` of shape
      `batch_shape + [num_timesteps, 1]` or `batch_shape + [num_timesteps]`,
      where `num_timesteps > 1`.

  Returns:
    expanded_time_series: `Tensor` of shape `batch_shape + [num_timesteps, 1]`.
  """

  with tf.compat.v1.name_scope(
      'maybe_expand_trailing_dim', values=[observed_time_series_tensor]):
    if (observed_time_series_tensor.shape.ndims is not None and
        tf.compat.dimension_value(
            observed_time_series_tensor.shape[-1]) is not None):
      expanded_time_series = (
          observed_time_series_tensor
          if observed_time_series_tensor.shape[-1] == 1
          else observed_time_series_tensor[..., tf.newaxis])
    else:
      expanded_time_series = tf.cond(
          pred=tf.equal(tf.shape(input=observed_time_series_tensor)[-1], 1),
          true_fn=lambda: observed_time_series_tensor,
          false_fn=lambda: observed_time_series_tensor[..., tf.newaxis])
    return expanded_time_series


def canonicalize_observed_time_series_with_mask(
    maybe_masked_observed_time_series):
  """Extract a Tensor with canonical shape and optional mask.

  Args:
    maybe_masked_observed_time_series: a `Tensor`-like object with shape
      `[..., num_timesteps]` or `[..., num_timesteps, 1]`, or a
      `tfp.sts.MaskedTimeSeries` containing such an object.
  Returns:
    masked_time_series: a `tfp.sts.MaskedTimeSeries` namedtuple, in which
      the `observed_time_series` is converted to `Tensor` with canonical shape
      `[..., num_timesteps, 1]`, and `is_missing` is either `None` or a boolean
      `Tensor`.
  """

  with tf.compat.v1.name_scope('canonicalize_observed_time_series_with_mask'):
    if hasattr(maybe_masked_observed_time_series, 'is_missing'):
      observed_time_series = (
          maybe_masked_observed_time_series.time_series)
      is_missing = maybe_masked_observed_time_series.is_missing
    else:
      observed_time_series = maybe_masked_observed_time_series
      is_missing = None

    observed_time_series = tf.convert_to_tensor(value=observed_time_series,
                                                name='observed_time_series')
    observed_time_series = _maybe_expand_trailing_dim(observed_time_series)

    if is_missing is not None:
      is_missing = tf.convert_to_tensor(
          value=is_missing, name='is_missing', dtype_hint=tf.bool)

    return missing_values_util.MaskedTimeSeries(observed_time_series,
                                                is_missing=is_missing)


def mix_over_posterior_draws(means, variances):
  """Construct a predictive normal distribution that mixes over posterior draws.

  Args:
    means: float `Tensor` of shape
      `[num_posterior_draws, ..., num_timesteps]`.
    variances: float `Tensor` of shape
      `[num_posterior_draws, ..., num_timesteps]`.

  Returns:
    mixture_dist: `tfd.MixtureSameFamily(tfd.Independent(tfd.Normal))` instance
      representing a uniform mixture over the posterior samples, with
      `batch_shape = ...` and `event_shape = [num_timesteps]`.

  """
  # The inputs `means`, `variances` have shape
  #   `concat([
  #      [num_posterior_draws],
  #      sample_shape,
  #      batch_shape,
  #      [num_timesteps]])`
  # Because MixtureSameFamily mixes over the rightmost batch dimension,
  # we need to move the `num_posterior_draws` dimension to be rightmost
  # in the batch shape. This requires use of `Independent` (to preserve
  # `num_timesteps` as part of the event shape) and `move_dimension`.
  # TODO(b/120245392): enhance `MixtureSameFamily` to reduce along an
  # arbitrary axis, and eliminate `move_dimension` calls here.

  with tf.compat.v1.name_scope(
      'mix_over_posterior_draws', values=[means, variances]):
    num_posterior_draws = dist_util.prefer_static_value(
        tf.shape(input=means))[0]

    component_observations = tfd.Independent(
        distribution=tfd.Normal(
            loc=dist_util.move_dimension(means, 0, -2),
            scale=tf.sqrt(dist_util.move_dimension(variances, 0, -2))),
        reinterpreted_batch_ndims=1)

    return tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(
            logits=tf.zeros([num_posterior_draws],
                            dtype=component_observations.dtype)),
        components_distribution=component_observations)
