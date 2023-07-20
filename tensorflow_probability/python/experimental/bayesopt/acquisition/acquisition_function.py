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
"""Acquisition Function Base Class."""

import abc

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensor_util


class AcquisitionFunction(object, metaclass=abc.ABCMeta):
  """Base class for acquisition functions.

  Acquisition Functions are (relatively) inexpensive functions that guide
  Bayesian Optimization search. Typically, their values at points will
  correspond to how desirable it is to evaluate the function at that point.
  This desirability can come in the form of improving information about the
  black box function (exploration) or trying to find an extrema given past
  evaluations (exploitation).

  TFP acquisition functions are callable objects that may be instantiated with
  subclass-specific parameters. This design enables expensive one-time
  computation to be run in `__init__` before the acquisition function is called
  repeatedly, for example during optimization. The `AcquisitionFunction` base
  class is instantiated with a predictive distribution (typically an instance of
  `tfd.GaussianProcessRegressionModel`, `tfd.StudentTProcessRegressionModel`, or
  `tfp.experimental.distributions.MultiTaskGaussianProcessRegressionModel`),
  previously-observed function values, and an optional random seed. The
  `__call__` method evaluates the acquisition function.
  """

  def __init__(
      self,
      predictive_distribution,
      observations,
      seed=None):
    """Initializes the acquisition function.

    Args:
      predictive_distribution: `tfd.Distribution`-like, the distribution over
        observations at a set of index points.
      observations: `Float` `Tensor` of observations.
      seed: PRNG seed; see tfp.random.sanitize_seed for details.
    """
    dtype = dtype_util.common_dtype([
        observations, predictive_distribution])
    self._predictive_distribution = predictive_distribution
    self._observations = tensor_util.convert_nonref_to_tensor(
        observations, dtype=dtype, name='observations')
    self._seed = seed

  @property
  def predictive_distribution(self):
    """The distribution over observations at a set of index points."""
    return self._predictive_distribution

  @property
  def observations(self):
    """Float `Tensor` of observations."""
    return self._observations

  @property
  def seed(self):
    """PRNG seed."""
    return self._seed

  @property
  def is_parallel(self):
    """Python `bool` indicating whether the acquisition function is parallel.

    Parallel (batched) acquisition functions evaluate batches of points rather
    than single points.
    """
    return False

  def __call__(self, **kwargs):
    raise NotImplementedError('Subclasses must implement `__call__`.')


class MCMCReducer(AcquisitionFunction):
  """Acquisition function for reducing over batch dimensions.

  `MCMCReducer` evaluates a base acquisition function and takes the mean of the
  function values over the dimensions indicated by `reduce_dims`. `MCMCReducer`
  is useful for marginalizing over an MCMC sample of GP kernel hyperparameters,
  for example.

  #### Examples

  Build and evaluate an acquisition function that computes Gaussian Process
  Expected Improvement and then marginalizes over the leftmost batch dimension.

  ```python
  import numpy as np
  import tensorflow_probability as tfp

  tfd = tfp.distributions
  tfpk = tfp.math.psd_kernels
  tfp_acq = tfp.experimental.bayesopt.acquisition

  # Sample 10 20-dimensional index points and associated observations.
  index_points = np.random.uniform(size=[10, 20])
  observations = np.random.uniform(size=[10])

  # The kernel and GP have batch shape [32], representing a sample of
  # hyperparameters that we want to marginalize over.
  kernel_amplitudes = np.random.uniform(size=[32])

  # Build a (batched) Gaussian Process regression model.
  dist = tfd.GaussianProcessRegressionModel(
      kernel=tfpk.MaternFiveHalves(amplitude=kernel_amplitudes),
      observation_index_points=index_points,
      observations=observations)

  # Define an `MCMCReducer` with GP Expected Improvement.
  mcmc_ei = tfp_acq.MCMCReducer(
      predictive_distribution=dist,
      observations=observations,
      acquisition_class=GaussianProcessExpectedImprovement,
      reduce_dims=0)

  # Evaluate the acquisition function at a new set of index points,
  # marginalizing over the hyperparameter batch.
  pred_index_points = np.random.uniform(size=[6, 20])
  acq_fn_vals = mcmc_ei(pred_index_points)  # Has shape [6].
  ```

  """

  def __init__(
      self,
      predictive_distribution,
      observations,
      seed=None,
      acquisition_class=None,
      reduce_dims=None,
      **acquisition_kwargs):
    """Initializes the acquisition function.

    Args:
      predictive_distribution: `tfd.Distribution`-like, the distribution over
        observations at a set of index points.
      observations: `Float` `Tensor` of observations.
      seed: PRNG seed; see tfp.random.sanitize_seed for details.
      acquisition_class: `AcquisitionFunction`-like callable.
      reduce_dims: Axis of the acquisition function value array over which to
        marginalize.
      **acquisition_kwargs: Kwargs passed to `acquisition_class`.
    """
    self._acquisition = acquisition_class(
        predictive_distribution,
        observations,
        seed=seed,
        **acquisition_kwargs)
    self._reduce_dims = reduce_dims
    super(MCMCReducer, self).__init__(
        predictive_distribution=predictive_distribution,
        observations=observations,
        seed=seed)

  @property
  def acquisition(self):
    return self._acquisition

  @property
  def reduce_dims(self):
    return self._reduce_dims

  @property
  def is_parallel(self):
    return self.acquisition.is_parallel

  def __call__(self, **kwargs):
    return tf.reduce_mean(
        self.acquisition(**kwargs), axis=ps.range(self.reduce_dims))
