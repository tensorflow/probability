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

  return tf.convert_to_tensor(batch_shape)


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
  with tf.name_scope('factored_joint_mvn', values=graph_parents):

    # We explicitly broadcast the `locs` so that we can concatenate them.
    # We don't have direct numerical access to the `scales`, which are arbitrary
    # linear operators, but `LinearOperatorBlockDiag` appears to do the right
    # thing without further intervention.
    dtype = tf.assert_same_float_dtype(distributions)
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
  with tf.name_scope('sum_mvns', values=graph_parents):
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
    observed_stddev: `Tensor` of shape `batch_shape`, giving the empirical
      standard deviation of each time series in the batch.
    observed_initial: `Tensor of shape `batch_shape`, giving the initial value
      of each time series in the batch.
  """

  with tf.name_scope('empirical_statistics', values=[observed_time_series]):
    observed_time_series = tf.convert_to_tensor(
        observed_time_series, name='observed_time_series')
    observed_time_series = maybe_expand_trailing_dim(observed_time_series)
    _, observed_variance = tf.nn.moments(
        tf.squeeze(observed_time_series, -1), axes=-1)
    observed_stddev = tf.sqrt(observed_variance)
    observed_initial = observed_time_series[..., 0, 0]
    return observed_stddev, observed_initial


def maybe_expand_trailing_dim(observed_time_series):
  """Ensures `observed_time_series` has a trailing dimension of size 1.

  This utility method tries to make time-series shape handling more ergonomic.
  The `tfd.LinearGaussianStateSpaceModel` Distribution has event shape of
  `[num_timesteps, observation_size]`, but canonical BSTS models
  are univariate, so their observation_size is always `1`. The extra trailing
  dimension gets annoying, so this method allows arguments with or without the
  extra dimension. There is no ambiguity except in the trivial special case
  where  `num_timesteps = 1`; this can be avoided by specifying any unit-length
  series in the explicit `[num_timesteps, 1]` style.

  Args:
    observed_time_series: `Tensor` of shape `batch_shape + [num_timesteps, 1]`
      or `batch_shape + [num_timesteps]`, where `num_timesteps > 1`.

  Returns:
    expanded_time_series: `Tensor` of shape `batch_shape + [num_timesteps, 1]`.
  """

  with tf.name_scope(
      'maybe_expand_trailing_dim', values=[observed_time_series]):
    if (observed_time_series.shape.ndims is not None and
        observed_time_series.shape[-1].value is not None):
      expanded_time_series = (
          observed_time_series if observed_time_series.shape[-1] == 1 else
          observed_time_series[..., tf.newaxis])
    else:
      expanded_time_series = tf.cond(
          tf.equal(tf.shape(observed_time_series)[-1], 1),
          lambda: observed_time_series,
          lambda: observed_time_series[..., tf.newaxis])
    return expanded_time_series
