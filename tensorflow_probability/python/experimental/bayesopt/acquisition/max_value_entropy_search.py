# Copyright 2023 The TensorFlow Probability Authors.
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
"""Max-Value Entropy Search Acquisition Function."""

import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import gumbel
from tensorflow_probability.python.distributions import normal as normal_lib
from tensorflow_probability.python.experimental.bayesopt.acquisition import acquisition_function
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.math import root_search
from tensorflow_probability.python.math import special
from tensorflow_probability.python.mcmc import sample_halton_sequence_lib


class GaussianProcessMaxValueEntropySearch(
    acquisition_function.AcquisitionFunction):
  """Max-value entropy search acquisition function.

  Computes the sequential max-value entropy search acquisition function.

  Requires that `predictive_distribution` has a `.mean`, `stddev` method.

  #### Examples

  Build and evaluate a Gausian Process Maximum Value Entropy Search acquisition
  function.

  ```python
  import numpy as np
  import tensorflow_probability as tfp

  tfd = tfp.distributions
  tfpk = tfp.math.psd_kernels
  tfp_acq = tfp.experimental.bayesopt.acquisition

  # Sample 10 20-dimensional index points and associated observations.
  index_points = np.random.uniform(size=[10, 20])
  observations = np.random.uniform(size=[10])

  # Build a Gaussian Process regression model.
  dist = tfd.GaussianProcessRegressionModel(
      kernel=tfpk.MaternFiveHalves(),
      observation_index_points=index_points,
      observations=observations)

  # Define a GP max value entropy search acquisition function.
  gp_mes = tfp_acq.GaussianProcessMaxValueEntropySearch(
      predictive_distribution=dist,
      observations=observations,
      num_max_value_samples=200)

  # Evaluate the acquisition function at a set of predictive index points.
  pred_index_points = np.random.uniform(size=[6, 20])
  acq_fn_vals = gp_mes(pred_index_points)
  ```

  #### References
  [1] Z. Wang, S. Jegelka. Max-value Entropy Search for Efficient Bayesian
      Optimization. https://arxiv.org/abs/1703.01968

  """

  def __init__(
      self,
      predictive_distribution,
      observations,
      seed=None,
      num_max_value_samples=100):
    """Constructs a max-value entropy search acquisition function.

    Args:
      predictive_distribution: `tfd.Distribution`-like, the distribution over
        observations at a set of index points.
      observations: `Float` `Tensor` of observations. Shape has the form
        `[b1, ..., bB, e]`, where `e` is the number of index points (such that
        the event shape of `predictive_distribution` is `[e]`) and
        `[b1, ..., bB]` is broadcastable with the batch shape of
        `predictive_distribution`.
      seed: PRNG seed; see tfp.random.sanitize_seed for details.
      num_max_value_samples: The number of samples to use for the max-value
        distribution.
    """
    self._num_max_value_samples = num_max_value_samples
    self._precomputed_samples = self._precompute_max_value_samples(
        predictive_distribution, observations, seed, num_max_value_samples)
    super(GaussianProcessMaxValueEntropySearch, self).__init__(
        predictive_distribution=predictive_distribution,
        observations=observations,
        seed=seed)

  @property
  def num_max_value_samples(self):
    return self._num_max_value_samples

  def _precompute_max_value_samples(
      self,
      predictive_distribution, observations, seed, num_max_value_samples):
    # First, approximately sample from p(y* | D), where y* is the true optima,
    # and D is the observed data.
    max_value_dist = fit_max_value_distribution(
        predictive_distribution,
        observations,
        num_grid_points=100, seed=seed)
    return max_value_dist.sample(num_max_value_samples, seed=seed)

  def __call__(self, **kwargs):
    """Computes the max-value entropy search acquisition function.

    Args:
      **kwargs: Keyword args passed on to the `mean` and `stddev` methods of
        `predictive_distribution`.

    Returns:
      Acquisition function values at index points implied by
      `predictive_distribution` (or overridden in `**kwargs`).
    """
    mean = self.predictive_distribution.mean(**kwargs)
    stddev = self.predictive_distribution.stddev(**kwargs)

    # Finally compute the approximation to the conditional mutual information
    # between our new location and the best value I({x, y} ; y* | D)

    norm = normal_lib.Normal(
        dtype_util.as_numpy_dtype(self.predictive_distribution.dtype)(0.), 1.)

    z = (self._precomputed_samples[..., tf.newaxis] - mean) / stddev
    return tf.reduce_mean(
        0.5 * z * inverse_mills_ratio(z) - norm.log_cdf(z), axis=0)


def fit_max_value_distribution(
    predictive_distribution,
    observations,
    num_grid_points=100,
    seed=None):
  """Computes a Gumbel approximation to the max-value distribution p(y* | D)."""
  # This is based on the Gumbel approximation in [1].

  # First approximate the CDF of the maximum F(y* < z | D) as prod_k F_k(z),
  # where F_k is the marginal (Normal) CDF at various points.

  # Adjoin a grid of points so the approximation is more accurate.
  grid_points = sample_halton_sequence_lib.sample_halton_sequence(
      dim=predictive_distribution.index_points.shape[-1],
      num_results=num_grid_points,
      dtype=predictive_distribution.index_points.dtype, seed=seed)

  dtype = dtype_util.as_numpy_dtype(predictive_distribution.index_points.dtype)
  norm = normal_lib.Normal(dtype(0.), 1.)

  broadcast_grid_shape = tf.concat(
      [ps.shape(predictive_distribution.index_points)[:-2],
       ps.shape(grid_points)], axis=0)
  grid_points = tf.broadcast_to(grid_points, broadcast_grid_shape)
  full_points = tf.concat(
      [grid_points, predictive_distribution.index_points], axis=-2)

  # Compute the stddev and mean at each point.
  stddev = predictive_distribution.stddev(index_points=full_points)
  mean = predictive_distribution.mean(index_points=full_points)

  # The values below will be used to initialize the root search in somewhere
  # in the center of the region.

  # We'll make the very greedy approximation that the max of the function is
  # more than the observations we have seen so far.
  min_value = tf.math.reduce_max(observations, axis=-1)

  # We'll assume that the maximum is no more than 5 standard deviations away
  # from any point.
  max_value = tf.math.reduce_max(mean + 5 * stddev, axis=-1)

  initial_position = tf.stack([0.5 * (min_value + max_value)] * 3)

  percentiles = np.array([0.25, 0.5, 0.75]).astype(dtype)
  percentiles_expanded = tf.reshape(
      percentiles,
      tf.concat(
          [[3], tf.ones_like(tf.shape(initial_position)[1:])], axis=0))

  # Finally compute the approximation to the log CDF. We want to find the
  # quantile values, so we will use a root search algorithm to do this.
  def log_cdf(x):
    return tf.math.log(percentiles_expanded) - tf.reduce_sum(
        norm.log_cdf((x[..., tf.newaxis] - mean) / stddev), axis=-1)

  quantiles = root_search.find_root_secant(
      log_cdf,
      initial_position=initial_position).estimated_root

  y1, y2, y3 = tf.unstack(quantiles, num=3)
  p1, p2, p3 = percentiles

  # Estimate the univariate Gumbel CDF through quantile matching.
  scale = (y3 - y1) / (np.log(-np.log(p1)) - np.log(-np.log(p3)))
  loc = y2 + scale * np.log(-np.log(p2))
  return gumbel.Gumbel(loc=loc, scale=scale)


def inverse_mills_ratio(x):
  """Compute the ratio of the Normal PDF and the Normal CDF."""
  dtype = dtype_util.as_numpy_dtype(x.dtype)
  return tf.math.reciprocal(
      np.sqrt(np.pi / 2.).astype(dtype) *
      special.erfcx(-x / np.sqrt(2.).astype(dtype)))
