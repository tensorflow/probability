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
"""Upper Confidence Bound."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.bayesopt.acquisition import acquisition_function
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import samplers


class ParallelUpperConfidenceBound(acquisition_function.AcquisitionFunction):
  """Parallel upper confidence bound acquisition function.

  Computes the q-UCB based on observed data using a stochastic process surrogate
  model. The computation is of the form `mean + exploration * stddev`.

  Requires that `predictive_distribution` has a `sample` method.

  #### Examples

  Build and evaluate a Parallel Upper Confidence Bound acquisition function.

  ```python
  import numpy as np
  import tensorflow_probability as tfp

  tfd = tfp.distributions
  tfpk = tfp.math.psd_kernels
  tfp_acq = tfp.experimental.bayesopt.acquisition

  # Sample 10 20-dimensional index points and associated observations.
  index_points = np.random.uniform(size=[10, 20])
  observations = np.random.uniform(size=[10])

  # Build a GP regression model conditioned on observed data.
  dist = tfd.GaussianProcessRegressionModel(
      kernel=tfpk.ExponentiatedQuadratic(),
      observation_index_points=index_points,
      observations=observations)

  gp_pucb = tfp_acq.ParallelUpperConfidenceBound(
      predictive_distribution=dist,
      observations=observations,
      exploration=0.05,
      num_samples=int(2e4))

  # Evaluate the acquisition function at a set of predictive index points.
  pred_index_points = np.random.uniform(size=[6, 20])
  acq_fn_vals = gp_pucb(pred_index_points)  # Has shape [6].
  ```

  """

  def __init__(
      self,
      predictive_distribution,
      observations,
      seed=None,
      exploration=0.01,
      num_samples=100,
      transform_fn=None):
    """Parallel Upper Confidence Bound acquisition function.

    Args:
      predictive_distribution: `tfd.Distribution`-like, the distribution over
        observations at a set of index points. Must have a `sample` method.
      observations: `Float` `Tensor` of observations. Shape has the form
        `[b1, ..., bB, e]`, where `e` is the number of index points (such that
        the event shape of `predictive_distribution` is `[e]`) and
        `[b1, ..., bB]` is broadcastable with the batch shape of
        `predictive_distribution`.
      seed: PRNG seed; see tfp.random.sanitize_seed for details.
      exploration: Exploitation-exploration trade-off parameter.
      num_samples: The number of samples to use for the Paralle Expected
        Improvement approximation.
      transform_fn: Optional Python `Callable` that transforms objective values.
        This is used for optimizing a composite grey box function `g(f(x))`
        where `f` is our black box function and `g` is `transform_fn`.
    """
    self._exploration = exploration
    self._num_samples = num_samples
    self._transform_fn = transform_fn
    super(ParallelUpperConfidenceBound, self).__init__(
        predictive_distribution=predictive_distribution,
        observations=observations,
        seed=seed)

  @property
  def exploration(self):
    return self._exploration

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
    """Computes the Parallel Upper Confidence Bound.

    Args:
      **kwargs: Keyword args passed on to the `sample` method of
        `predictive_distribution`.

    Returns:
      Parallel upper confidence bounds at index points implied by
      `predictive_distribution` (or overridden in `**kwargs`).

    #### References
    [1] J. Wilson, R. Moriconi, F. Hutter, M. Deisenroth
    The reparameterization trick for acquisition functions
    https://bayesopt.github.io/papers/2017/32.pdf
    """
    # Fix the seed so we get a deterministic objective per iteration.
    seed = samplers.sanitize_seed(
        [100, 2] if self.seed is None else self.seed, salt='qucb')

    samples = self.predictive_distribution.sample(
        self.num_samples, seed=seed, **kwargs)

    # This parameterization differs from [1] in that we don't assume that
    # samples come from a Normal distribution with a rescaled covariance. This
    # effectively reparameterizes the exploration parameter by a factor of
    # sqrt(pi / 2).
    if self._transform_fn is not None:
      samples = self._transform_fn(samples)
      mean = tf.math.reduce_mean(samples, axis=0)
    else:
      mean = self.predictive_distribution.mean(**kwargs)

    qucb = mean + self.exploration * tf.math.abs(samples - mean)
    return tf.reduce_mean(tf.reduce_max(qucb, axis=-1), axis=0)


class GaussianProcessUpperConfidenceBound(
    acquisition_function.AcquisitionFunction):
  """Analytical Gaussian Process upper confidence bound acquisition function.

  Computes the analytic sequential upper confidence bound for a Gaussian
  process model.

  Requires that `predictive_distribution` has a `.mean`, `stddev` method.

  #### Examples

  Build and evaluate a GP Upper Confidence Bound acquisition function.

  ```python
  import numpy as np
  import tensorflow_probability as tfp

  tfd = tfp.distributions
  tfpk = tfp.math.psd_kernels
  tfp_acq = tfp.experimental.bayesopt.acquisition

  # Sample 12 5-dimensional index points and associated observations.
  index_points = np.random.uniform(size=[12, 5])
  observations = np.random.uniform(size=[12])

  # Build a GP regression model conditioned on observed data.
  dist = tfd.GaussianProcessRegressionModel(
      kernel=tfpk.ExponentiatedQuadratic(),
      observation_index_points=index_points,
      observations=observations)

  # Build a GP upper confidence bound acquisition function.
  gp_ucb = tfp_acq.GausianProcessUpperConfidenceBound(
      predictive_distribution=dist,
      observations=observations,
      exploration=0.05,
      num_samples=int(2e4))

  # Evaluate the acquisition function at a set of 6 predictive index points.
  pred_index_points = np.random.uniform(size=[6, 5])
  acq_fn_vals = gp_ucb(pred_index_points)  # Has shape [6].
  ```

  """

  def __init__(
      self,
      predictive_distribution,
      observations,
      seed=None,
      exploration=0.01):
    """Constructs a GP Upper Confidence Bound acquisition function.

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
      exploration: Exploitation-exploration trade-off parameter.
    """

    self._exploration = exploration
    super(GaussianProcessUpperConfidenceBound, self).__init__(
        predictive_distribution=predictive_distribution,
        observations=observations,
        seed=seed)

  @property
  def exploration(self):
    return self._exploration

  def __call__(self, **kwargs):
    """Computes analytic GP upper confidence bound.

    Args:
      **kwargs: Keyword args passed on to the `mean` and `stddev` methods of
        `predictive_distribution`.

    Returns:
      Upper confidence bound at index points implied by
      `predictive_distribution` (or overridden in `**kwargs`).
    """
    stddev = self.predictive_distribution.stddev(**kwargs)
    mean = self.predictive_distribution.mean(**kwargs)
    return normal_upper_confidence_bound(
        mean, stddev, exploration=self.exploration)


def normal_upper_confidence_bound(mean, stddev, exploration=0.01):
  """Normal distribution upper confidence bound.

  Args:
    mean: Array of predicted means. Must broadcast with `stddev`.
    stddev: Array of predicted standard deviations. Must broadcast with `mean`.
    exploration: Float parameter controlling the exploration/exploitation
      tradeoff.

  Returns:
    ucb: Array of upper confidence bound values.
  """
  dtype = dtype_util.common_dtype([mean, stddev])
  mean = tf.convert_to_tensor(mean, dtype=dtype)
  stddev = tf.convert_to_tensor(stddev, dtype=dtype)
  return mean + exploration * stddev
