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
"""Expected Improvement."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import student_t
from tensorflow_probability.python.experimental.bayesopt.acquisition import acquisition_function
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import samplers


class ParallelExpectedImprovement(acquisition_function.AcquisitionFunction):
  """Parallel expected improvement acquisition function.

  Computes the q-EI from a multivariate observation model. This is also known as
  batch expected improvement.

  Requires that `predictive_distribution` has a `sample` method.

  #### Examples

  Build and evaluate a Parallel Expected Improvement acquisition function.

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

  # Define a Parallel Expected Improvement acquisition function.
  stp_pei = tfp_acq.ParallelExpectedImprovement(
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
      exploration=0.01,
      num_samples=100,
      transform_fn=None):
    """Constructs a Parallel Expected Improvement acquisition function.

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
      num_samples: The number of samples to use for the Parallel Expected
        Improvement approximation.
      transform_fn: Optional Python `Callable` that transforms objective values.
        This is used for optimizing a composite grey box function `g(f(x))`
        where `f` is our black box function and `g` is `transform_fn`.
    """
    self._exploration = exploration
    self._num_samples = num_samples
    self._transform_fn = transform_fn
    super(ParallelExpectedImprovement, self).__init__(
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
    """Computes the Parallel Expected Improvement.

    Args:
      **kwargs: Keyword args passed on to the `sample` method of
        `predictive_distribution`.

    Returns:
      Parallel Expected improvements at index points implied by
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

    qei = tf.nn.relu(
        transform_fn(samples) - best_observed - self.exploration)
    return tf.reduce_mean(tf.reduce_max(qei, axis=-1), axis=0)


class StudentTProcessExpectedImprovement(
    acquisition_function.AcquisitionFunction):
  """Student-T Process expected improvement acquisition function.

  Computes the analytic sequential expected improvement for a Student-T process
  model.

  Requires that `predictive_distribution` has a `mean`, `stddev` method.

  #### Examples

  Build and evaluate a Student T Process Expected Improvement acquisition
  function.

  ```python
  import numpy as np
  import tensorflow_probability as tfp

  tfd = tfp.distributions
  tfpk = tfp.math.psd_kernels
  tfp_acq = tfp.experimental.bayesopt.acquisition

  # Sample 10 5-dimensional index points and associated observations.
  index_points = np.random.uniform(size=[10, 5])
  observations = np.random.uniform(size=[10])

  # Build a Student T Process regression model over the function values at
  # `predictive_index_points` conditioned on observations.
  predictive_index_points = np.random.uniform(size=[8, 5])
  dist = tfd.StudentTProcessRegressionModel(
      kernel=tfpk.MaternFiveHalves(),
      df=5.,
      observation_index_points=index_points,
      observations=observations,
      predictive_index_points=predictive_index_points)

  # Define a Student T Process Expected Improvement acquisition function.
  stp_ei = tfp_acq.StudentTProcessExpectedImprovement(
      predictive_distribution=dist,
      observations=observations,
      exploration=0.02)

  # Evaluate the acquisition function at `predictive_index_points`.
  acq_fn_vals = stp_ei()  # Has shape [8].

  # Evaluate the acquisition function at a new set of predictive index points.
  new_pred_index_points = np.random.uniform(size=[6, 5])
  acq_fn_vals = stp_ei(pred_index_points)  # Has shape [6].
  ```

  """

  def __init__(
      self,
      predictive_distribution,
      observations,
      seed=None,
      exploration=0.01):
    """Compute Expected Improvement w.r.t a Student-T Process analytically.

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
    super(StudentTProcessExpectedImprovement, self).__init__(
        predictive_distribution=predictive_distribution,
        observations=observations,
        seed=seed)

  @property
  def exploration(self):
    return self._exploration

  def __call__(self, **kwargs):
    """Computes the Student-T process expected improvement.

    Args:
      **kwargs: Keyword args passed on to the `mean` and `stddev` methods of
        `predictive_distribution`.

    Returns:
      Expected improvements at index points implied by `predictive_distribution`
      (or overridden in `**kwargs`).
    """
    mean = self.predictive_distribution.mean(**kwargs)
    stddev = self.predictive_distribution.stddev(**kwargs)
    df = self.predictive_distribution.df
    best_observed = tf.reduce_max(self.observations, axis=-1)
    return student_t_expected_improvement(
        best_observed, df, mean, stddev, exploration=self.exploration)


class GaussianProcessExpectedImprovement(
    acquisition_function.AcquisitionFunction):
  """Gaussian Process expected improvement acquisition function.

  Computes the analytic sequential expected improvement for a Gaussian process
  model.

  Requires that `predictive_distribution` has a `mean`, `stddev` method.

  #### Examples

  Build and evaluate a Gausian Process Expected Improvement acquisition
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

  # Build a Gaussian Process regression model over the function values at
  # `predictive_index_points` conditioned on observations.
  predictive_index_points = np.random.uniform(size=[8, 20])
  dist = tfd.GaussianProcessRegressionModel(
      kernel=tfpk.MaternFiveHalves(),
      observation_index_points=index_points,
      observations=observations,
      predictive_index_points=predictive_index_points)

  # Define a GP Expected Improvement acquisition function.
  gp_ei = tfp_acq.GaussianProcessExpectedImprovement(
      predictive_distribution=dist,
      observations=observations)

  # Evaluate the acquisition function at `predictive_index_points`.
  acq_fn_vals = gp_ei()

  # Evaluate the acquisition function at a new set of predictive index points.
  pred_index_points = np.random.uniform(size=[6, 20])
  acq_fn_vals = gp_ei(pred_index_points)
  ```

  """

  def __init__(
      self,
      predictive_distribution,
      observations,
      seed=None,
      exploration=0.01):
    """Compute Expected Improvement w.r.t a Gaussian Process analytically.

    Args:
      predictive_distribution: `tfd.Distribution`-like, the distribution over
        observations at a set of index points. Must have `mean`, `stddev`
        methods.
      observations: `Float` `Tensor` of observations.
      seed: PRNG seed; see tfp.random.sanitize_seed for details.
      exploration: Exploitation-exploration trade-off parameter.
    """

    self._exploration = exploration
    super(GaussianProcessExpectedImprovement, self).__init__(
        predictive_distribution=predictive_distribution,
        observations=observations,
        seed=seed)

  @property
  def exploration(self):
    return self._exploration

  def __call__(self, **kwargs):
    """Computes analytic GP expected improvement.

    Args:
      **kwargs: Keyword args passed on to the `mean` and `stddev` methods of
        `predictive_distribution`.

    Returns:
      Expected improvements at index points implied by `predictive_distribution`
      (or overridden in `**kwargs`).
    """
    mean = self.predictive_distribution.mean(**kwargs)
    stddev = self.predictive_distribution.stddev(**kwargs)
    best_observed = tf.reduce_max(self.observations, axis=-1)
    return normal_expected_improvement(
        best_observed, mean, stddev, exploration=self.exploration)


def normal_expected_improvement(best_observed, mean, stddev, exploration=0.01):
  """Normal distribution expected improvement.

  Args:
    best_observed: Array of best (largest) observed values. Must broadcast with
      `mean` and `stddev`.
    mean: Array of predicted means. Must broadcast with `best_observed` and
      `stddev`.
    stddev: Array of predicted standard deviations. Must broadcast with
      `best_observed` and `mean`.
    exploration: Float parameter controlling the exploration/exploitation
      tradeoff.

  Returns:
    ei: Array of expected improvement values.
  """
  dtype = dtype_util.common_dtype([best_observed, mean, stddev])
  best_observed = tf.convert_to_tensor(best_observed, dtype=dtype)
  mean = tf.convert_to_tensor(mean, dtype=dtype)
  stddev = tf.convert_to_tensor(stddev, dtype=dtype)
  norm = normal.Normal(tf.zeros([], dtype=dtype), 1.)
  imp = mean - best_observed - exploration
  z = imp / stddev
  return imp * norm.cdf(z) + stddev * norm.prob(z)


def student_t_expected_improvement(
    best_observed, df, mean, stddev, exploration=0.01):
  """Student-T distribution expected improvement.

  Args:
    best_observed: Array of best (largest) observed values. Must broadcast with
      `mean` and `stddev`.
    df: Student T degrees of freedom.
    mean: Array of predicted means. Must broadcast with `best_observed` and
      `stddev`.
    stddev: Array of predicted standard deviations. Must broadcast with
      `best_observed` and `mean`.
    exploration: Float parameter controlling the exploration/exploitation
      tradeoff.

  Returns:
    ei: Array of expected improvement values.
  """
  dtype = dtype_util.common_dtype([best_observed, df, mean, stddev])
  best_observed = tf.convert_to_tensor(best_observed, dtype=dtype)
  df = tf.convert_to_tensor(df, dtype=dtype)
  mean = tf.convert_to_tensor(mean, dtype=dtype)
  stddev = tf.convert_to_tensor(stddev, dtype=dtype)
  st = student_t.StudentT(df, tf.zeros([], dtype=dtype), 1.)
  imp = mean - best_observed - exploration
  z = imp / stddev
  return stddev * (z * st.cdf(z) +
                   df / (df - 1.) * (1 + tf.math.square(z) / df) * st.prob(z))
