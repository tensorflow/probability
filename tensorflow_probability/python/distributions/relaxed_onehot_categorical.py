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
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import chain as chain_bijector
from tensorflow_probability.python.bijectors import exp as exp_bijector
from tensorflow_probability.python.bijectors import softmax_centered as softmax_centered_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import


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
      name='ExpRelaxedOneHotCategorical'):
    """Initialize ExpRelaxedOneHotCategorical using class log-probabilities.

    Args:
      temperature: A `Tensor`, representing the temperature of one or more
        distributions. The temperature values must be positive, and the shape
        must broadcast against `(logits or probs)[..., 0]`.
      logits: An N-D `Tensor`, `N >= 1`, representing the log probabilities
        of one or many distributions. The first `N - 1` dimensions index into a
        batch of independent distributions and the last dimension represents a
        vector of logits for each class. Only one of `logits` or `probs` should
        be passed in.
      probs: An N-D `Tensor`, `N >= 1`, representing the probabilities
        of one or many distributions. The first `N - 1` dimensions index into a
        batch of independent distributions and the last dimension represents a
        vector of probabilities for each class. Only one of `logits` or `probs`
        should be passed in.
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
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([logits, probs, temperature], tf.float32)
      self._temperature = tensor_util.convert_nonref_to_tensor(
          temperature, dtype_hint=dtype, name='temperature')
      self._logits = tensor_util.convert_nonref_to_tensor(
          logits, dtype_hint=dtype, name='logits')
      self._probs = tensor_util.convert_nonref_to_tensor(
          probs, dtype_hint=dtype, name='probs')
      if (self._probs is None) == (self._logits is None):
        raise ValueError('Must pass `probs` or `logits`, but not both.')

      super(ExpRelaxedOneHotCategorical, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        temperature=parameter_properties.ParameterProperties(
            shape_fn=lambda sample_shape: sample_shape[:-1],
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        logits=parameter_properties.ParameterProperties(event_ndims=1),
        probs=parameter_properties.ParameterProperties(
            event_ndims=1,
            default_constraining_bijector_fn=softmax_centered_bijector
            .SoftmaxCentered,
            is_preferred=False))
    # pylint: enable=g-long-lambda

  @property
  @deprecation.deprecated(
      '2019-10-01', 'The `event_size` property is deprecated.  Use '
      '`tf.shape(self.probs if self.logits is None else self.logits)[-1]` '
      'instead.')
  def event_size(self):
    """Scalar `int32` tensor: the number of classes."""
    return self._event_size()

  def _event_size(self, logits=None):
    param = logits
    if param is None:
      param = self._logits if self._logits is not None else self._probs
    if param.shape is not None:
      event_size = tf.compat.dimension_value(param.shape[-1])
      if event_size is not None:
        return event_size
    return tf.shape(param)[-1]

  @property
  def temperature(self):
    """Batchwise temperature tensor of a RelaxedCategorical."""
    return self._temperature

  @property
  def logits(self):
    """Input argument `logits`."""
    return self._logits

  @property
  def probs(self):
    """Input argument `probs`."""
    return self._probs

  def _batch_shape_tensor(self, temperature=None, logits=None):
    param = logits
    if param is None:
      param = self._logits if self._logits is not None else self._probs
    if temperature is None:
      temperature = self.temperature
    return ps.broadcast_shape(
        ps.shape(temperature), ps.shape(param)[:-1])

  def _batch_shape(self):
    param = self._logits if self._logits is not None else self._probs
    return tf.broadcast_static_shape(self.temperature.shape, param.shape[:-1])

  def _event_shape_tensor(self, logits=None):
    param = logits
    if param is None:
      param = self._logits if self._logits is not None else self._probs
    return ps.shape(param)[-1:]

  def _event_shape(self):
    param = self._logits if self._logits is not None else self._probs
    return tensorshape_util.with_rank(param.shape[-1:], rank=1)

  def _sample_n(self, n, seed=None):
    temperature = tf.convert_to_tensor(self.temperature)
    logits = self._logits_parameter_no_checks()

    # Uniform variates must be sampled from the open-interval `(0, 1)` rather
    # than `[0, 1)`. To do so, we use
    # `np.finfo(dtype_util.as_numpy_dtype(self.dtype)).tiny` because it is the
    # smallest, positive, 'normal' number. A 'normal' number is such that the
    # mantissa has an implicit leading 1. Normal, positive numbers x, y have the
    # reasonable property that, `x + y >= max(x, y)`. In this case, a subnormal
    # number (i.e., np.nextafter) can cause us to sample 0.
    uniform_shape = ps.concat(
        [[n],
         self._batch_shape_tensor(temperature=temperature, logits=logits),
         self._event_shape_tensor(logits=logits)], 0)
    uniform = samplers.uniform(
        shape=uniform_shape,
        minval=np.finfo(dtype_util.as_numpy_dtype(self.dtype)).tiny,
        maxval=1.,
        dtype=self.dtype,
        seed=seed)
    gumbel = -tf.math.log(-tf.math.log(uniform))
    noisy_logits = (gumbel + logits) / temperature[..., tf.newaxis]
    return tf.math.log_softmax(noisy_logits)

  def _log_prob(self, x):
    temperature = tf.convert_to_tensor(self.temperature)
    logits = self._logits_parameter_no_checks()

    # broadcast logits or x if need be.
    if (not tensorshape_util.is_fully_defined(x.shape) or
        not tensorshape_util.is_fully_defined(logits.shape) or
        x.shape != logits.shape):
      logits = tf.ones_like(x, dtype=logits.dtype) * logits
      x = tf.ones_like(logits, dtype=x.dtype) * x
    # compute the normalization constant
    k = tf.cast(self._event_size(logits), x.dtype)
    log_norm_const = (
        tf.math.lgamma(k) + (k - 1.) * tf.math.log(temperature))
    # compute the unnormalized density
    log_softmax = tf.math.log_softmax(logits - x * temperature[..., tf.newaxis])
    log_unnorm_prob = tf.reduce_sum(log_softmax, axis=[-1], keepdims=False)
    # combine unnormalized density with normalization constant
    return log_norm_const + log_unnorm_prob

  def logits_parameter(self, name=None):
    """Logits vec computed from non-`None` input arg (`probs` or `logits`)."""
    with self._name_and_control_scope(name or 'logits_parameter'):
      return self._logits_parameter_no_checks()

  def _logits_parameter_no_checks(self):
    if self._logits is None:
      return tf.math.log(self._probs)
    return tf.identity(self._logits)

  def probs_parameter(self, name=None):
    """Probs vec computed from non-`None` input arg (`probs` or `logits`)."""
    with self._name_and_control_scope(name or 'probs_parameter'):
      return self._probs_parameter_no_checks()

  def _probs_parameter_no_checks(self):
    if self._logits is None:
      return tf.identity(self._probs)
    return tf.math.softmax(self._logits)

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(assert_util.assert_non_positive(
        x,
        message=('Samples must be less than or equal to `0` for '
                 '`ExpRelaxedOneHotCategorical` or `1` for '
                 '`RelaxedOneHotCategorical`.')))
    assertions.append(assert_util.assert_near(
        tf.zeros([], dtype=self.dtype), tf.reduce_logsumexp(x, axis=[-1]),
        message=('Final dimension of samples must sum to `0` for ''.'
                 '`ExpRelaxedOneHotCategorical` or `1` '
                 'for `RelaxedOneHotCategorical`.')))
    return assertions

  def _parameter_control_dependencies(self, is_init):
    assertions = []

    logits = self._logits
    probs = self._probs
    param, name = (probs, 'probs') if logits is None else (logits, 'logits')

    # In init, we can always build shape and dtype checks because
    # we assume shape doesn't change for Variable backed args.
    if is_init:
      if not dtype_util.is_floating(param.dtype):
        raise TypeError('Argument `{}` must having floating type.'.format(name))

      msg = 'Argument `{}` must have rank at least 1.'.format(name)
      shape_static = tensorshape_util.dims(param.shape)
      if shape_static is not None:
        if len(shape_static) < 1:
          raise ValueError(msg)
      elif self.validate_args:
        param = tf.convert_to_tensor(param)
        assertions.append(
            assert_util.assert_rank_at_least(param, 1, message=msg))

      msg1 = 'Argument `{}` must have final dimension >= 1.'.format(name)
      msg2 = 'Argument `{}` must have final dimension <= {}.'.format(
          name, dtype_util.max(tf.int32))
      event_size = shape_static[-1] if shape_static is not None else None
      if event_size is not None:
        if event_size < 1:
          raise ValueError(msg1)
        if event_size > dtype_util.max(tf.int32):
          raise ValueError(msg2)
      elif self.validate_args:
        param = tf.convert_to_tensor(param)
        assertions.append(assert_util.assert_greater_equal(
            tf.shape(param)[-1:], 1, message=msg1))
        # NOTE: For now, we leave out a runtime assertion that
        # `tf.shape(param)[-1] <= tf.int32.max`.  An earlier `tf.shape` call
        # will fail before we get to this point.

    if not self.validate_args:
      assert not assertions  # Should never happen.
      return []

    if is_init != tensor_util.is_ref(self.temperature):
      assertions.append(assert_util.assert_positive(self.temperature))

    if probs is not None:
      probs = param  # reuse tensor conversion from above
      if is_init != tensor_util.is_ref(probs):
        probs = tf.convert_to_tensor(probs)
        one = tf.ones([], dtype=probs.dtype)
        assertions.extend([
            assert_util.assert_non_negative(probs),
            assert_util.assert_less_equal(probs, one),
            assert_util.assert_near(
                tf.reduce_sum(probs, axis=-1), one,
                message='Argument `probs` must sum to 1.'),
        ])

    return assertions

  def _default_event_space_bijector(self):
    # TODO(b/145620027) Finalize choice of bijector.
    return chain_bijector.Chain([
        exp_bijector.Log(validate_args=self.validate_args),
        softmax_centered_bijector.SoftmaxCentered(
            validate_args=self.validate_args),
    ], validate_args=self.validate_args)


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
      name='RelaxedOneHotCategorical'):
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
    parameters = dict(locals())
    dist = ExpRelaxedOneHotCategorical(temperature,
                                       logits=logits,
                                       probs=probs,
                                       validate_args=validate_args,
                                       allow_nan_stats=allow_nan_stats)

    super(RelaxedOneHotCategorical, self).__init__(dist,
                                                   exp_bijector.Exp(),
                                                   validate_args=validate_args,
                                                   parameters=parameters,
                                                   name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        temperature=parameter_properties.ParameterProperties(
            shape_fn=lambda sample_shape: sample_shape[:-1],
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        logits=parameter_properties.ParameterProperties(event_ndims=1),
        probs=parameter_properties.ParameterProperties(
            event_ndims=1,
            default_constraining_bijector_fn=softmax_centered_bijector
            .SoftmaxCentered,
            is_preferred=False))
    # pylint: enable=g-long-lambda

  @property
  def temperature(self):
    """Batchwise temperature tensor of a RelaxedCategorical."""
    return self.distribution.temperature

  @property
  @deprecation.deprecated(
      '2019-10-01', 'The `event_size` property is deprecated.  Use '
      '`tf.shape(self.probs if self.logits is None else self.logits)[-1]` '
      'instead.')
  def event_size(self):
    """Scalar `int32` tensor: the number of classes."""
    return self.distribution.event_size

  @property
  def probs(self):
    """Input argument `probs`."""
    return self.distribution.probs

  @property
  def logits(self):
    """Input argument `logits`."""
    return self.distribution.logits

  experimental_is_sharded = False

  def logits_parameter(self, name=None):
    """Logits vec computed from non-`None` input arg (`probs` or `logits`)."""
    return self.distribution.logits_parameter(name)

  def probs_parameter(self, name=None):
    """Probs vec computed from non-`None` input arg (`probs` or `logits`)."""
    return self.distribution.probs_parameter(name)

  def _default_event_space_bijector(self):
    return softmax_centered_bijector.SoftmaxCentered(
        validate_args=self.validate_args)
