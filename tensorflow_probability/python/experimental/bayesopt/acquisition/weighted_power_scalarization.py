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
"""Weighted Power/Chebyshev Scalarization."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.experimental.bayesopt.acquisition import acquisition_function
from tensorflow_probability.python.experimental.distributions import multitask_gaussian_process as mtgp
from tensorflow_probability.python.experimental.distributions import multitask_gaussian_process_regression_model as mtgprm
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util


class WeightedPowerScalarization(acquisition_function.AcquisitionFunction):
  """Weighted power scalarization acquisition function.

  Given a multi-task distribution over `T` tasks, the weighted power
  scalarization acquisition function computes

  `(sum_t w_t |a_t(x)|^p)^(1/p)`

  where:
    * `a_t` are the list of `acquisition_function_classes`.
    * `w_t` are `weights`.
    * `p` is `power`.

  By default `p` is `None` which corresponds to a value of `inf`. This is
  Chebyshev scalarization: `max w_t |a_t(x)|`, and for non `inf` `p` corresponds
  to weighted power scalarization.

  #### Examples

  Build and evaluate a Weighted Power Scalarization acquisition function.

  ```python
  import numpy as np
  import tensorflow_probability as tfp

  tfpk = tfp.math.psd_kernels
  tfpke = tfp.experimental.psd_kernels
  tfde = tfp.experimental.distributions
  tfp_acq = tfp.experimental.bayesopt.acquisition

  kernel = tfpk.ExponentiatedQuadratic()
  mt_kernel = tfpke.Independent(base_kernel=kernel, num_tasks=4)

  # Sample 10 20-dimensional index points and associated 4-dimensional
  # observations.
  index_points = np.random.uniform(size=[10, 20])
  observations = np.random.uniform(size=[10, 4])

  # Build a multitask GP.
  dist = tfde.MultiTaskGaussianProcessRegressionModel(
      kernel=mt_kernel,
      observation_index_points=index_points,
      observations,
      observation_noise_variance=1e-4)

  # Choose weights and acquisition functions for each task.
  weights = np.array([0.8, 1., 1.1, 0.5])
  acquisition_function_classes = [
      tfp_acq.GaussianProcessExpectedImprovement,
      tfp_acq.GaussianProcessUpperConfidenceBound,
      tfp_acq.GaussianProcessExpectedImprovement,
      tfp_acq.GaussianProcessUpperConfidenceBound]

  # Build the acquisition function.
  cheb_scalar_fn = tfp_acq.WeightedPowerScalarization(
      predictive_distribution=dist,
      acquisition_function_classes=acquisition_function_classes,
      observations=observations,
      weights=weights)

  # Evaluate the acquisition function on 6 predictive index points. Note that
  # `index_points` must be passed as a keyword arg.
  pred_index_points = np.random.uniform(size=[6, 20])
  acq_fn_vals = cheb_scalar_fn(index_points=pred_index_points)
  ```

  """

  def __init__(
      self,
      predictive_distribution,
      observations,
      seed=None,
      acquisition_function_classes=None,
      acquisition_kwargs_list=None,
      power=None,
      weights=None):
    r"""Construct a weighted power scalarization acquisition function.

    Args:
      predictive_distribution: `tfd.Distribution`-like, the distribution over
        observations at a set of index points (expected to be a
        `tfd.MultiTaskGaussianProcess` or
        `tfd.MultiTaskGaussianProcessRegressionModel`).
      observations: `Float` `Tensor` of observed function values. Shape has the
        form `[b1, ..., bB, N, T]`, where `N` is the number of index points and
        `T` is the number of tasks (such that the event shape of
        `predictive_distribution` is `[N, T]`) and `[b1, ..., bB]` is
        broadcastable with the batch shape of `predictive_distribution`.
      seed: PRNG seed; see tfp.random.sanitize_seed for details.
      acquisition_function_classes: `Callable` acquisition function, one per
        task.
      acquisition_kwargs_list: Kwargs to pass in to acquisition function.
      power: Numpy `float`. When this is set to `None`, this corresponds to
        Chebyshev scalarization.
      weights: `Tensor` of shape `[T]` where, `T` is the number of tasks.
    """

    if not isinstance(
        predictive_distribution, (
            mtgp.MultiTaskGaussianProcess,
            mtgprm.MultiTaskGaussianProcessRegressionModel)):
      raise ValueError(
          'Expected `predictive_distribution` to be a '
          '`MultiTaskGaussianProcess` model.')
    if acquisition_function_classes is None:
      raise ValueError(
          'Expected a list of Acquisition Function classes')
    self._acquisition_function_classes = acquisition_function_classes
    self._acquisition_kwargs_list = acquisition_kwargs_list
    if acquisition_kwargs_list and (
        len(acquisition_function_classes) != len(acquisition_kwargs_list)):
      raise ValueError('Expected `acquisition_kwargs_list` to be the same '
                       'size as `acquisition_function_classes`')
    dtype = dtype_util.common_dtype(
        [weights, power, predictive_distribution])
    self._weights = tensor_util.convert_nonref_to_tensor(
        weights, dtype=dtype)
    self._power = tensor_util.convert_nonref_to_tensor(
        power, dtype=dtype)
    super(WeightedPowerScalarization, self).__init__(
        predictive_distribution=predictive_distribution,
        observations=observations,
        seed=seed)

  @property
  def acquisition_function_classes(self):
    return self._acquisition_function_classes

  @property
  def acquisition_kwargs_list(self):
    return self._acquisition_kwargs_list

  @property
  def weights(self):
    return self._weights

  @property
  def power(self):
    return self._power

  @property
  def is_parallel(self):
    return False

  def __call__(self, **kwargs):
    """Computes the weighted power scalarization.

    Args:
      **kwargs: Keyword args passed on to the `mean` and `stddev` methods of
        `predictive_distribution`.

    Returns:
      Weighted power scalarization at index points implied by
      `predictive_distribution` (or overridden in `**kwargs`).
    """
    # Fix the seed so we get a deterministic objective per iteration.
    seed = samplers.sanitize_seed(
        [100, 2] if self.seed is None else self.seed, salt='qei')

    loc = self.predictive_distribution.mean(**kwargs)
    scale = self.predictive_distribution.stddev(**kwargs)
    observations = tf.convert_to_tensor(self.observations)

    # Because we expect a multitask GP, squeeze index points if there is only
    # one of them.
    if loc.shape[-2] == 1:
      loc = tf.squeeze(loc, axis=-2)
    if scale.shape[-2] == 1:
      scale = tf.squeeze(scale, axis=-2)

    acquisition_values = []
    seeds = samplers.split_seed(
        seed, self.predictive_distribution.kernel.num_tasks)
    for i in range(self.predictive_distribution.kernel.num_tasks):
      # Get the distribution for the i-th task.
      dist = normal.Normal(loc[..., i], scale[..., i])
      acquisition_kwargs = None
      if self.acquisition_kwargs_list:
        acquisition_kwargs = self.acquisition_kwargs_list[i]
      if acquisition_kwargs is None:
        acquisition_kwargs = {}

      acquisition_values.append(
          self.acquisition_function_classes[i](
              dist, observations[..., i], seeds[i], **acquisition_kwargs)())
    acquisition_values = tf.math.abs(tf.stack(acquisition_values, axis=-1))

    weights = 1. if self.weights is None else self.weights

    if self.power is None:
      # Chebyshev scalarization.
      return tf.reduce_max(weights * acquisition_values, axis=-1)

    power = tf.convert_to_tensor(self.power)

    return tf.math.pow(
        tf.reduce_sum(weights * acquisition_values ** power, axis=-1),
        tf.math.reciprocal(power))
