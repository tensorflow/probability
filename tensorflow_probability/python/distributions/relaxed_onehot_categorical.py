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
"""Relaxed OneHotCategorical distribution classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf
from tensorflow_probability.python.bijectors import exp as exp_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization


class ExpRelaxedOneHotCategorical(distribution.Distribution):
  """ExpRelaxedOneHotCategorical distribution with temperature and logits.

  An ExpRelaxedOneHotCategorical distribution is a log-transformed
  RelaxedOneHotCategorical distribution. The RelaxedOneHotCategorical is a
  distribution over random probability vectors, vectors of positive real
  values that sum to one, which continuously approximates a OneHotCategorical.
  The degree of approximation is controlled by a temperature: as the temperature
  goes to 0 the RelaxedOneHotCategorical becomes discrete with a distribution
  described by the logits, as the temperature goes to infinity the
  RelaxedOneHotCategorical becomes the constant distribution that is identically
  the constant vector of (1/event_size, ..., 1/event_size).

  Because computing log-probabilities of the RelaxedOneHotCategorical can
  suffer from underflow issues, this class is one solution for loss
  functions that depend on log-probabilities, such as the KL Divergence found
  in the variational autoencoder loss. The KL divergence between two
  distributions is invariant under invertible transformations, so evaluating
  KL divergences of ExpRelaxedOneHotCategorical samples, which are always
  followed by a `tf.exp` op, is equivalent to evaluating KL divergences of
  RelaxedOneHotCategorical samples. See the appendix of Maddison et al., 2016
  for more mathematical details, where this distribution is called the
  ExpConcrete.

  #### Examples

  Creates a continuous distribution, whose exp approximates a 3-class one-hot
  categorical distribution. The 2nd class is the most likely to be the
  largest component in samples drawn from this distribution. If those samples
  are followed by a `tf.exp` op, then they are distributed as a relaxed onehot
  categorical.

  ```python
  temperature = 0.5
  p = [0.1, 0.5, 0.4]
  dist = ExpRelaxedOneHotCategorical(temperature, probs=p)
  samples = dist.sample()
  exp_samples = tf.exp(samples)
  # exp_samples has the same distribution as samples from
  # RelaxedOneHotCategorical(temperature, probs=p)
  ```

  Creates a continuous distribution, whose exp approximates a 3-class one-hot
  categorical distribution. The 2nd class is the most likely to be the
  largest component in samples drawn from this distribution.

  ```python
  temperature = 0.5
  logits = [-2, 2, 0]
  dist = ExpRelaxedOneHotCategorical(temperature, logits=logits)
  samples = dist.sample()
  exp_samples = tf.exp(samples)
  # exp_samples has the same distribution as samples from
  # RelaxedOneHotCategorical(temperature, probs=p)
  ```

  Creates a continuous distribution, whose exp approximates a 3-class one-hot
  categorical distribution. Because the temperature is very low, samples from
  this distribution are almost discrete, with one component almost 0 and the
  others very negative. The 2nd class is the most likely to be the largest
  component in samples drawn from this distribution.

  ```python
  temperature = 1e-5
  logits = [-2, 2, 0]
  dist = ExpRelaxedOneHotCategorical(temperature, logits=logits)
  samples = dist.sample()
  exp_samples = tf.exp(samples)
  # exp_samples has the same distribution as samples from
  # RelaxedOneHotCategorical(temperature, probs=p)
  ```

  Creates a continuous distribution, whose exp approximates a 3-class one-hot
  categorical distribution. Because the temperature is very high, samples from
  this distribution are usually close to the (-log(3), -log(3), -log(3)) vector.
  The 2nd class is still the most likely to be the largest component
  in samples drawn from this distribution.

  ```python
  temperature = 10
  logits = [-2, 2, 0]
  dist = ExpRelaxedOneHotCategorical(temperature, logits=logits)
  samples = dist.sample()
  exp_samples = tf.exp(samples)
  # exp_samples has the same distribution as samples from
  # RelaxedOneHotCategorical(temperature, probs=p)
  ```

  Chris J. Maddison, Andriy Mnih, and Yee Whye Teh. The Concrete Distribution:
  A Continuous Relaxation of Discrete Random Variables. 2016.
  """

  def __init__(
      self,
      temperature,
      logits=None,
      probs=None,
      validate_args=False,
      allow_nan_stats=True,
      name="ExpRelaxedOneHotCategorical"):
    """Initialize ExpRelaxedOneHotCategorical using class log-probabilities.

    Args:
      temperature: An 0-D `Tensor`, representing the temperature
        of a set of ExpRelaxedCategorical distributions. The temperature should
        be positive.
      logits: An N-D `Tensor`, `N >= 1`, representing the log probabilities
        of a set of ExpRelaxedCategorical distributions. The first
        `N - 1` dimensions index into a batch of independent distributions and
        the last dimension represents a vector of logits for each class. Only
        one of `logits` or `probs` should be passed in.
      probs: An N-D `Tensor`, `N >= 1`, representing the probabilities
        of a set of ExpRelaxedCategorical distributions. The first
        `N - 1` dimensions index into a batch of independent distributions and
        the last dimension represents a vector of probabilities for each
        class. Only one of `logits` or `probs` should be passed in.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with tf.compat.v1.name_scope(
        name, values=[logits, probs, temperature]) as name:

      dtype = dtype_util.common_dtype([logits, probs, temperature], tf.float32)
      self._logits, self._probs = distribution_util.get_logits_and_probs(
          name=name,
          logits=logits,
          probs=probs,
          validate_args=validate_args,
          multidimensional=True,
          dtype=dtype)

      with tf.control_dependencies(
          [tf.compat.v1.assert_positive(temperature)] if validate_args else []):
        self._temperature = tf.convert_to_tensor(
            value=temperature, name="temperature", dtype=dtype)
        self._temperature_2d = tf.reshape(
            self._temperature, [-1, 1], name="temperature_2d")

      logits_shape_static = self._logits.shape.with_rank_at_least(1)
      if logits_shape_static.ndims is not None:
        self._batch_rank = tf.convert_to_tensor(
            value=logits_shape_static.ndims - 1,
            dtype=tf.int32,
            name="batch_rank")
      else:
        with tf.compat.v1.name_scope(name="batch_rank"):
          self._batch_rank = tf.rank(self._logits) - 1

      with tf.compat.v1.name_scope(name="event_size"):
        self._event_size = tf.shape(input=self._logits)[-1]

    super(ExpRelaxedOneHotCategorical, self).__init__(
        dtype=dtype,
        reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._logits, self._probs, self._temperature],
        name=name)

  @classmethod
  def _params_event_ndims(cls):
    return dict(temperature=0, logits=1, probs=1)

  @property
  def event_size(self):
    """Scalar `int32` tensor: the number of classes."""
    return self._event_size

  @property
  def temperature(self):
    """Batchwise temperature tensor of a RelaxedCategorical."""
    return self._temperature

  @property
  def logits(self):
    """Vector of coordinatewise logits."""
    return self._logits

  @property
  def probs(self):
    """Vector of probabilities summing to one."""
    return self._probs

  def _batch_shape_tensor(self):
    return tf.broadcast_dynamic_shape(
        tf.shape(input=self.temperature),
        tf.shape(input=self.logits)[:-1])

  def _batch_shape(self):
    return tf.broadcast_static_shape(self.temperature.shape,
                                     self.logits.shape[:-1])

  def _event_shape_tensor(self):
    return tf.shape(input=self.logits)[-1:]

  def _event_shape(self):
    return self.logits.shape.with_rank_at_least(1)[-1:]

  def _sample_n(self, n, seed=None):
    # Uniform variates must be sampled from the open-interval `(0, 1)` rather
    # than `[0, 1)`. To do so, we use `np.finfo(self.dtype.as_numpy_dtype).tiny`
    # because it is the smallest, positive, "normal" number. A "normal" number
    # is such that the mantissa has an implicit leading 1. Normal, positive
    # numbers x, y have the reasonable property that, `x + y >= max(x, y)`. In
    # this case, a subnormal number (i.e., np.nextafter) can cause us to sample
    # 0.
    uniform_shape = tf.concat(
        [[n], self.batch_shape_tensor(), self.event_shape_tensor()], 0)
    uniform = tf.random.uniform(
        shape=uniform_shape,
        minval=np.finfo(self.dtype.as_numpy_dtype).tiny,
        maxval=1.,
        dtype=self.dtype,
        seed=seed)
    gumbel = -tf.math.log(-tf.math.log(uniform))
    noisy_logits = (gumbel + self.logits) / self.temperature[..., tf.newaxis]
    return tf.nn.log_softmax(noisy_logits)

  def _log_prob(self, x):
    x = self._assert_valid_sample(x)
    # broadcast logits or x if need be.
    logits = self.logits
    if (not x.shape.is_fully_defined() or
        not logits.shape.is_fully_defined() or
        x.shape != logits.shape):
      logits = tf.ones_like(x, dtype=logits.dtype) * logits
      x = tf.ones_like(logits, dtype=x.dtype) * x
    # compute the normalization constant
    k = tf.cast(self.event_size, x.dtype)
    log_norm_const = (
        tf.math.lgamma(k) + (k - 1.) * tf.math.log(self.temperature))
    # compute the unnormalized density
    log_softmax = tf.nn.log_softmax(
        self.logits - x * self.temperature[..., tf.newaxis])
    log_unnorm_prob = tf.reduce_sum(
        input_tensor=log_softmax, axis=[-1], keepdims=False)
    # combine unnormalized density with normalization constant
    return log_norm_const + log_unnorm_prob

  def _assert_valid_sample(self, x):
    if not self.validate_args:
      return x
    return distribution_util.with_dependencies([
        tf.compat.v1.assert_non_positive(x),
        tf.compat.v1.assert_near(
            tf.zeros([], dtype=self.dtype),
            tf.reduce_logsumexp(input_tensor=x, axis=[-1])),
    ], x)


class RelaxedOneHotCategorical(
    transformed_distribution.TransformedDistribution):
  """RelaxedOneHotCategorical distribution with temperature and logits.

  The RelaxedOneHotCategorical is a distribution over random probability
  vectors, vectors of positive real values that sum to one, which continuously
  approximates a OneHotCategorical. The degree of approximation is controlled by
  a temperature: as the temperature goes to 0 the RelaxedOneHotCategorical
  becomes discrete with a distribution described by the `logits` or `probs`
  parameters, as the temperature goes to infinity the RelaxedOneHotCategorical
  becomes the constant distribution that is identically the constant vector of
  (1/event_size, ..., 1/event_size).

  The RelaxedOneHotCategorical distribution was concurrently introduced as the
  Gumbel-Softmax (Jang et al., 2016) and Concrete (Maddison et al., 2016)
  distributions for use as a reparameterized continuous approximation to the
  `Categorical` one-hot distribution. If you use this distribution, please cite
  both papers.

  #### Examples

  Creates a continuous distribution, which approximates a 3-class one-hot
  categorical distribution. The 2nd class is the most likely to be the
  largest component in samples drawn from this distribution.

  ```python
  temperature = 0.5
  p = [0.1, 0.5, 0.4]
  dist = RelaxedOneHotCategorical(temperature, probs=p)
  ```

  Creates a continuous distribution, which approximates a 3-class one-hot
  categorical distribution. The 2nd class is the most likely to be the
  largest component in samples drawn from this distribution.

  ```python
  temperature = 0.5
  logits = [-2, 2, 0]
  dist = RelaxedOneHotCategorical(temperature, logits=logits)
  ```

  Creates a continuous distribution, which approximates a 3-class one-hot
  categorical distribution. Because the temperature is very low, samples from
  this distribution are almost discrete, with one component almost 1 and the
  others nearly 0. The 2nd class is the most likely to be the largest component
  in samples drawn from this distribution.

  ```python
  temperature = 1e-5
  logits = [-2, 2, 0]
  dist = RelaxedOneHotCategorical(temperature, logits=logits)
  ```

  Creates a continuous distribution, which approximates a 3-class one-hot
  categorical distribution. Because the temperature is very high, samples from
  this distribution are usually close to the (1/3, 1/3, 1/3) vector. The 2nd
  class is still the most likely to be the largest component
  in samples drawn from this distribution.

  ```python
  temperature = 10
  logits = [-2, 2, 0]
  dist = RelaxedOneHotCategorical(temperature, logits=logits)
  ```

  Eric Jang, Shixiang Gu, and Ben Poole. Categorical Reparameterization with
  Gumbel-Softmax. 2016.

  Chris J. Maddison, Andriy Mnih, and Yee Whye Teh. The Concrete Distribution:
  A Continuous Relaxation of Discrete Random Variables. 2016.
  """

  def __init__(
      self,
      temperature,
      logits=None,
      probs=None,
      validate_args=False,
      allow_nan_stats=True,
      name="RelaxedOneHotCategorical"):
    """Initialize RelaxedOneHotCategorical using class log-probabilities.

    Args:
      temperature: An 0-D `Tensor`, representing the temperature
        of a set of RelaxedOneHotCategorical distributions. The temperature
        should be positive.
      logits: An N-D `Tensor`, `N >= 1`, representing the log probabilities
        of a set of RelaxedOneHotCategorical distributions. The first
        `N - 1` dimensions index into a batch of independent distributions and
        the last dimension represents a vector of logits for each class. Only
        one of `logits` or `probs` should be passed in.
      probs: An N-D `Tensor`, `N >= 1`, representing the probabilities
        of a set of RelaxedOneHotCategorical distributions. The first `N - 1`
        dimensions index into a batch of independent distributions and the last
        dimension represents a vector of probabilities for each class. Only one
        of `logits` or `probs` should be passed in.
      validate_args: Unused in this distribution.
      allow_nan_stats: Python `bool`, default `True`. If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member. If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: A name for this distribution (optional).
    """
    dist = ExpRelaxedOneHotCategorical(temperature,
                                       logits=logits,
                                       probs=probs,
                                       validate_args=validate_args,
                                       allow_nan_stats=allow_nan_stats)
    super(RelaxedOneHotCategorical, self).__init__(dist,
                                                   exp_bijector.Exp(),
                                                   name=name)
