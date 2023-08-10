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
"""Probability of Improvement Acquisition Function."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.experimental.bayesopt.acquisition import acquisition_function
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import samplers


class ParallelProbabilityOfImprovement(
    acquisition_function.AcquisitionFunction):
  """Parallel probability of improvement acquisition function.

  Computes the q-PI from a multivariate observation model. This is also known as
  batch probability of improvement.

  Requires that `predictive_distribution` has a `sample` method.

  #### Examples

  Build and evaluate a Parallel Probability of Improvement acquisition function.

  ```python
  import numpy as np
  import tensorflow_probability as tfp

  tfd = tfp.distributions
  tfpk = tfp.math.psd_kernels
  tfp_acq = tfp.experimental.bayesopt.acquisition

  # Sample 10 20-dimensional index points and associated observations.
  index_points = np.random.uniform(size=[10, 20])
  observations = np.random.uniform(size=[10])

  # Build a Student T Process regression model conditioned on observed data.
  dist = tfd.StudentTProcessRegressionModel(
      kernel=tfpk.ExponentiatedQuadratic(),
      df=5.,
      observation_index_points=index_points,
      observations=observations)

  # Define a Parallel Probability of Improvement acquisition function.
  stp_pei = tfp_acq.ParallelProbabilityOfImprovement(
      predictive_distribution=dist,
      observations=observations,
      num_samples=10_000)

  # Evaluate the acquisition function at a new set of index points.
  pred_index_points = np.random.uniform(size=[6, 20])
  acq_fn_vals = stp_pei(pred_index_points)  # Has shape [6].
  ```

  """

  def __init__(
      self,
      predictive_distribution,
      observations,
      seed=None,
      num_samples=100,
      transform_fn=None):
    """Constructs a Parallel Probability of Improvement acquisition function.

    Args:
      predictive_distribution: `tfd.Distribution`-like, the distribution over
        observations at a set of index points. Must have a `sample` method.
      observations: `Float` `Tensor` of observations. Shape has the form
        `[b1, ..., bB, e]`, where `e` is the number of index points (such that
        the event shape of `predictive_distribution` is `[e]`) and
        `[b1, ..., bB]` is broadcastable with the batch shape of
        `predictive_distribution`.
      seed: PRNG seed; see tfp.random.sanitize_seed for details.
      num_samples: The number of samples to use for the Parallel Probability of
        Improvement approximation.
      transform_fn: Optional Python `Callable` that transforms objective values.
        This is used for optimizing a composite grey box function `g(f(x))`
        where `f` is our black box function and `g` is `transform_fn`.
    """
    self._num_samples = num_samples
    self._transform_fn = transform_fn
    super(ParallelProbabilityOfImprovement, self).__init__(
        predictive_distribution=predictive_distribution,
        observations=observations,
        seed=seed)

  @property
  def num_samples(self):
    return self._num_samples

  @property
  def transform_fn(self):
    return self._transform_fn

  @property
  def is_parallel(self):
    return True

  def __call__(self, **kwargs):
    """Computes the Parallel Probability of Improvement.

    Args:
      **kwargs: Keyword args passed on to the `sample` method of
        `predictive_distribution`.

    Returns:
      Parallel Probability of improvement at index points implied by
      `predictive_distribution` (or overridden in `**kwargs`).
    """
    # Fix the seed so we get a deterministic objective per iteration.
    seed = samplers.sanitize_seed(
        [100, 2] if self.seed is None else self.seed, salt='qei')

    samples = self.predictive_distribution.sample(
        self.num_samples, seed=seed, **kwargs)

    transform_fn = lambda x: x
    if self._transform_fn is not None:
      transform_fn = self._transform_fn

    best_observed = tf.reduce_max(transform_fn(self.observations), axis=-1)
    qpi = (transform_fn(samples) - best_observed) > 0.
    return tf.reduce_mean(
        tf.cast(tf.reduce_any(qpi, axis=-1), dtype=samples.dtype), axis=0)


class GaussianProcessProbabilityOfImprovement(
    acquisition_function.AcquisitionFunction):
  """Gaussian Process probability of improvement acquisition function.

  Computes the analytic sequential probability of improvement for a Gaussian
  process model relative to observed data.

  Requires that `predictive_distribution` has `mean` and `stddev` methods.

  #### Examples

  Build and evaluate a GP Probability of Improvement acquisition function.

  ```python
  import numpy as np
  import tensorflow_probability as tfp

  tfd = tfp.distributions
  tfpk = tfp.math.psd_kernels
  tfp_acq = tfp.experimental.bayesopt.acquisition

  # Sample 10 4-dimensional index points and associated observations.
  index_points = np.random.uniform(size=[10, 4])
  observations = np.random.uniform(size=[10])

  # Build a GP regression model.
  dist = tfd.GaussianProcessRegressionModel(
      kernel=tfpk.ExponentiatedQuadratic(),
      observation_index_points=index_points,
      observations=observations)

  gp_poi = tfp_acq.GaussianProcessProbabilityOfImprovement(
      predictive_distribution=dist,
      observations=observations)

  # Evaluate the acquisition function at a set of predictive index points.
  pred_index_points = np.random.uniform(size=[6, 4])
  acq_fn_vals = gp_poi(pred_index_points)  # Has shape [6].
  ```

  """

  def __init__(self, predictive_distribution, observations, seed=None):
    """Constructs a Probability of Improvement acquisition function.

    Args:
      predictive_distribution: `tfd.Distribution`-like, the distribution over
        observations at a set of index points. Must have `mean`, `stddev`
        methods.
      observations: `Float` `Tensor` of observations. Shape has the form
        `[b1, ..., bB, e]`, where `e` is the number of index points (such that
        the event shape of `predictive_distribution` is `[e]`) and
        `[b1, ..., bB]` is broadcastable with the batch shape of
        `predictive_distribution`.
      seed: PRNG seed; see tfp.random.sanitize_seed for details.
    """
    super(GaussianProcessProbabilityOfImprovement, self).__init__(
        predictive_distribution=predictive_distribution,
        observations=observations,
        seed=seed)

  def __call__(self, **kwargs):
    """Computes analytic GP probability of improvement.

    Args:
      **kwargs: Keyword args passed on to the `mean` and `stddev` methods of
        `predictive_distribution`.

    Returns:
      Probability of improvement at index points implied by
      `predictive_distribution` (or overridden in `**kwargs`).
    """
    stddev = self.predictive_distribution.stddev(**kwargs)
    mean = self.predictive_distribution.mean(**kwargs)
    best_observed = tf.reduce_max(self.observations, axis=-1)
    return normal_probability_of_improvement(best_observed, mean, stddev)


def normal_probability_of_improvement(best_observed, mean, stddev):
  """Normal distribution probability of improvement.

  Args:
    best_observed: Array of best (largest) observed values. Must broadcast with
      `mean` and `stddev`.
    mean: Array of predicted means. Must broadcast with `best_observed` and
      `stddev`.
    stddev: Array of predicted standard deviations. Must broadcast with
      `best_observed` and `mean`.

  Returns:
    poi: Array of expected improvement values.
  """
  dtype = dtype_util.common_dtype([best_observed, mean, stddev])
  best_observed = tf.convert_to_tensor(best_observed, dtype=dtype)
  mean = tf.convert_to_tensor(mean, dtype=dtype)
  stddev = tf.convert_to_tensor(stddev, dtype=dtype)
  norm = normal.Normal(mean, stddev)
  return norm.survival_function(best_observed)
