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
"""Generalized Linear Model specifications."""

import contextlib
import numpy as np

import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import name_util
from tensorflow_probability.python.util.deferred_tensor import DeferredTensor


__all__ = [
    'Bernoulli',
    'BernoulliNormalCDF',
    'Binomial',
    'CustomExponentialFamily',
    'ExponentialFamily',
    'GammaExp',
    'GammaSoftplus',
    'LogNormal',
    'LogNormalSoftplus',
    'NegativeBinomial',
    'NegativeBinomialSoftplus',
    'Normal',
    'NormalReciprocal',
    'Poisson',
    'PoissonSoftplus',
]


class ExponentialFamily(tf.Module):
  """Specifies a mean-value parameterized exponential family.

  Subclasses implement [exponential-family distribution](
  https://en.wikipedia.org/wiki/Exponential_family) properties (e.g.,
  `log_prob`, `variance`) as a function of a real-value which is transformed via
  some [link function](
  https://en.wikipedia.org/wiki/Generalized_linear_model#Link_function)
  to be interpreted as the distribution's mean. The distribution is
  parameterized by this mean, i.e., "mean-value parameterized."

  Subclasses are typically used to specify a Generalized Linear Model (GLM). A
  [GLM]( https://en.wikipedia.org/wiki/Generalized_linear_model) is a
  generalization of linear regression which enables efficient fitting of
  log-likelihood losses beyond just assuming `Normal` noise. See `tfp.glm.fit`
  for more details.

  Subclasses must implement `_as_distribution` which does not need to be either
  "tape-safe" or "variable-safe." (`tfp.glm` families are however guaranteed to
  be both tape and variable safe.)

  Subclasses may optionally implement `_call` and `_log_prob` which otherwise
  default to:

  ```python
  def _call(self, predicted_linear_response):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(predicted_linear_response)
      likelihood = self.as_distribution(predicted_linear_response)
      mean = likelihood.mean()
    variance = likelihood.variance()
    grad_mean = tape.gradient(mean, predicted_linear_response)
    return mean, variance, grad_mean

  def _log_prob(self, response, predicted_linear_response):
    likelihood = self.as_distribution(predicted_linear_response)
    return likelihood.log_prob(response)
  ```

  In context of `tfp.glm.fit` and `tfp.glm.fit_sparse`, these functions are used
  to find the best fitting weights for given model matrix ("X") and responses
  ("Y").
  """

  def __init__(self, name=None):
    """Creates the ExponentialFamily.

    Args:
      name: Python `str` used as TF namescope for ops created by member
        functions. Default value: `None` (i.e., the subclass name).
    """
    if not name:
      name = type(self).__name__
      name = name_util.camel_to_lower_snake(name)
    name = name_util.get_name_scope_name(name)
    name = name_util.strip_invalid_chars(name)
    super(ExponentialFamily, self).__init__(name=name)

  def _call(self, predicted_linear_response):
    """Default implementation of the __call__ computation."""
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(predicted_linear_response)
      likelihood = self.as_distribution(predicted_linear_response)
      mean = likelihood.mean()
    variance = likelihood.variance()
    grad_mean = tape.gradient(mean, predicted_linear_response)
    return mean, variance, grad_mean

  def __call__(self, predicted_linear_response, name=None):
    """Computes `mean(r), var(mean), d/dr mean(r)` for linear response, `r`.

    Here `mean` and `var` are the mean and variance of the sufficient statistic,
    which may not be the same as the mean and variance of the random variable
    itself.  If the distribution's density has the form

    ```none
    p_Y(y) = h(y) Exp[dot(theta, T(y)) - A]
    ```

    where `theta` and `A` are constants and `h` and `T` are known functions,
    then `mean` and `var` are the mean and variance of `T(Y)`.  In practice,
    often `T(Y) := Y` and in that case the distinction doesn't matter.

    Args:
      predicted_linear_response: `float`-like `Tensor` corresponding to
        `tf.linalg.matmul(model_matrix, weights)`.
      name: Python `str` used as TF namescope for ops created by member
        functions. Default value: `None` (i.e., 'call').

    Returns:
      mean: `Tensor` with shape and dtype of `predicted_linear_response`
        representing the distribution prescribed mean, given the prescribed
        linear-response to mean mapping.
      variance: `Tensor` with shape and dtype of `predicted_linear_response`
        representing the distribution prescribed variance, given the prescribed
        linear-response to mean mapping.
      grad_mean: `Tensor` with shape and dtype of `predicted_linear_response`
        representing the gradient of the mean with respect to the
        linear-response and given the prescribed linear-response to mean
        mapping.
    """
    with self._base_name_scope(name, 'call'):
      predicted_linear_response = tf.convert_to_tensor(
          predicted_linear_response,
          dtype_hint=tf.float32,
          name='predicted_linear_response')
      return self._call(predicted_linear_response)

  def _log_prob(self, response, predicted_linear_response):
    """Implementation for `log_prob` public function."""
    likelihood = self.as_distribution(predicted_linear_response)
    return likelihood.log_prob(response)

  def log_prob(self, response, predicted_linear_response, name=None):
    """Computes `D(param=mean(r)).log_prob(response)` for linear response, `r`.

    Args:
      response: `float`-like `Tensor` representing observed ("actual")
        responses.
      predicted_linear_response: `float`-like `Tensor` corresponding to
        `tf.linalg.matmul(model_matrix, weights)`.
      name: Python `str` used as TF namescope for ops created by member
        functions. Default value: `None` (i.e., 'log_prob').

    Returns:
      log_prob: `Tensor` with shape and dtype of `predicted_linear_response`
        representing the distribution prescribed log-probability of the observed
        `response`s.
    """
    with self._base_name_scope(name, 'log_prob'):
      dtype = dtype_util.common_dtype([response, predicted_linear_response],
                                      dtype_hint=tf.float32)
      response = tf.convert_to_tensor(
          response, dtype=dtype, name='response')
      predicted_linear_response = tf.convert_to_tensor(
          predicted_linear_response, dtype=dtype,
          name='predicted_linear_response')
      return self._log_prob(response, predicted_linear_response)

  def _as_distribution(self, predicted_linear_response):
    """Implementation for `as_distribution` public function."""
    raise NotImplementedError('`_as_distribution` is not implemented.')

  def as_distribution(self, predicted_linear_response, name=None):
    """Builds a mean parameterized TFP Distribution from linear response.

    Example:

    ```python
    model = tfp.glm.Bernoulli()
    r = tfp.glm.compute_predicted_linear_response(x, w)
    yhat = model.as_distribution(r)
    ```

    Args:
      predicted_linear_response: `response`-shaped `Tensor` representing linear
        predictions based on new `model_coefficients`, i.e.,
        `tfp.glm.compute_predicted_linear_response(
           model_matrix, model_coefficients, offset)`.
      name: Python `str` used as TF namescope for ops created by member
        functions. Default value: `None` (i.e., 'log_prob').

    Returns:
      model: `tfp.distributions.Distribution`-like object with mean
        parameterized by `predicted_linear_response`.
    """
    with self._base_name_scope(name, 'as_distribution'):
      predicted_linear_response = tf.convert_to_tensor(
          predicted_linear_response, dtype_hint=tf.float32,
          name='predicted_linear_response')
      return self._as_distribution(predicted_linear_response)

  def __str__(self):
    return 'tfp.glm.family.{type_name}(\'{self_name}\')'.format(
        type_name=type(self).__name__,
        self_name=self.name)

  def __repr__(self):
    return '<tfp.glm.family.{type_name} \'{self_name}\'>'.format(
        type_name=type(self).__name__,
        self_name=self.name)

  @contextlib.contextmanager
  def _base_name_scope(self, name=None, default_name=None):
    """Helper function to standardize op scope."""
    with tf.name_scope(self.name):
      with tf.name_scope(name or default_name) as scope:
        yield scope


class CustomExponentialFamily(ExponentialFamily):
  """Constucts GLM from arbitrary distribution and inverse link function."""

  def __init__(self,
               distribution_fn,
               linear_model_to_mean_fn,
               name=None):
    """Creates the `CustomExponentialFamily`.

    Args:
      distribution_fn: Python `callable` which returns a
        `tf.distribution.Distribution`-like instance from a single input
        representing the distribution's required `mean`, i.e.,
        `mean = linear_model_to_mean_fn(matmul(model_matrix, weights))`.
      linear_model_to_mean_fn: Python `callable` which returns the
        distribution's required mean as computed from the predicted linear
        response, `tf.linalg.matmul(model_matrix, weights)`.
      name: Python `str` used as TF namescope for ops created by member
        functions. Default value: `None` (i.e., the subclass name).
    """
    super(CustomExponentialFamily, self).__init__(name)
    self._distribution_fn = distribution_fn
    self._inverse_link_fn = (
        linear_model_to_mean_fn.forward
        if isinstance(linear_model_to_mean_fn, bijectors.Bijector)
        else linear_model_to_mean_fn)

  @property
  def distribution_fn(self):
    return self._distribution_fn

  @property
  def linear_model_to_mean_fn(self):
    return self._inverse_link_fn

  def _as_distribution(self, r):
    mean = DeferredTensor(r, self._inverse_link_fn)
    return self._distribution_fn(mean)


class Bernoulli(ExponentialFamily):
  """`Bernoulli(probs=mean)` where `mean = sigmoid(X @ weights)`."""

  def _call(self, r):
    mean = tf.math.sigmoid(r)
    variance = grad_mean = mean * tf.math.sigmoid(-r)
    return mean, variance, grad_mean

  def _as_distribution(self, r):
    return tfd.Bernoulli(logits=r)


class BernoulliNormalCDF(ExponentialFamily):
  """`Bernoulli(probs=mean)` where `mean = Normal(0, 1).cdf(X @ weights)`."""

  def _call(self, r):
    dtype = dtype_util.as_numpy_dtype(r.dtype)
    d = tfd.Normal(loc=tf.zeros([], dtype), scale=tf.ones([], dtype))
    mean = d.cdf(r)
    # var = cdf(r) * cdf(-r) but cdf(-r) = 1 - cdf(r) = survival_function(r).
    variance = mean * d.survival_function(r)
    grad_mean = d.prob(r)
    return mean, variance, grad_mean

  def _as_distribution(self, r):
    return tfd.Bernoulli(logits=DeferredTensor(r, self._as_logits))

  def _as_logits(self, r):
    dtype = dtype_util.as_numpy_dtype(r.dtype)
    d = tfd.Normal(loc=tf.zeros([], dtype), scale=tf.ones([], dtype))
    # logit(ncdf(r)) = log(ncdf(r)) - log(1-ncdf(r)) = logncdf(r) - lognsf(r).
    return d.log_cdf(r) - d.log_survival_function(r)


class Binomial(ExponentialFamily):
  """`Binomial(total_count, probs=mean)`.

  Where `mean = total_count * sigmoid(matmul(X, weights))`.
  """

  def __init__(self, total_count=1., name=None):
    self._total_count = total_count
    super(Binomial, self).__init__(name=name)

  def _call(self, r):
    mean = self._total_count * tf.nn.sigmoid(r)
    variance = grad_mean = mean * tf.nn.sigmoid(-r)
    return mean, variance, grad_mean

  def _as_distribution(self, r):
    total_count = DeferredTensor(
        self._total_count, lambda x: tf.cast(x, r.dtype), dtype=r.dtype)
    return tfd.Binomial(total_count=total_count, logits=r)


class GammaExp(ExponentialFamily):
  """`Gamma(concentration=1, rate=1 / mean)` where `mean = exp(X @ w))`."""

  def __init__(self, concentration=1., name=None):
    self._concentration = concentration
    super(GammaExp, self).__init__(name=name)

  def _call(self, r):
    c = tf.cast(self._concentration, r.dtype)
    er = tf.math.exp(r)
    mean = grad_mean = er * c
    variance = er * mean
    return mean, variance, grad_mean

  def _as_distribution(self, r):
    concentration = DeferredTensor(
        self._concentration, lambda x: tf.cast(x, r.dtype), dtype=r.dtype)
    return tfd.Gamma(
        concentration=concentration,
        rate=DeferredTensor(r, lambda x: tf.math.exp(-x)))


class GammaSoftplus(ExponentialFamily):
  """`Gamma(concentration=1, rate=1 / mean)` where `mean = softplus(X @ w))`."""

  def __init__(self, concentration=1., name=None):
    self._concentration = concentration
    super(GammaSoftplus, self).__init__(name=name)

  def _call(self, r):
    c = tf.cast(self._concentration, r.dtype)
    spr = tf.math.softplus(r)
    mean = spr * c
    variance = spr * mean
    grad_mean = tf.math.sigmoid(r) * c
    return mean, variance, grad_mean

  def _as_distribution(self, r):
    concentration = DeferredTensor(
        self._concentration, lambda x: tf.cast(x, r.dtype), dtype=r.dtype)
    return tfd.Gamma(
        concentration=concentration,
        rate=DeferredTensor(r, lambda x: 1. / tf.math.softplus(x)))


class Poisson(ExponentialFamily):
  """`Poisson(rate=mean)` where `mean = exp(X @ weights)`."""

  def _call(self, r):
    mean = variance = grad_mean = tf.math.exp(r)
    return mean, variance, grad_mean

  def _as_distribution(self, r):
    return tfd.Poisson(log_rate=r)


class PoissonSoftplus(ExponentialFamily):
  """`Poisson(rate=mean)` where `mean = softplus(X @ weights)`."""

  def _call(self, r):
    mean = variance = tf.math.softplus(r)
    grad_mean = tf.math.sigmoid(r)
    return mean, variance, grad_mean

  def _as_distribution(self, r):
    return tfd.Poisson(rate=DeferredTensor(r, tf.math.softplus))


class LogNormal(ExponentialFamily):
  """`LogNormal(loc=log(mean) - log(2) / 2, scale=sqrt(log(2)))` where
  `mean = exp(X @ weights)`.
  """

  def __init__(self, scale=np.sqrt(np.log(2.)), name=None):
    self._scale = scale
    super(LogNormal, self).__init__(name=name)

  def _call(self, r):
    mean = grad_mean = tf.math.exp(r)
    variance = mean**2.
    if tf.get_static_value(self._scale) != np.sqrt(np.log(2.)):
      s = tf.cast(self._scale, r.dtype)
      variance *= tf.math.expm1(s**2.)
    return mean, variance, grad_mean

  def _as_distribution(self, r):
    scale = DeferredTensor(
        self._scale, lambda x: tf.cast(x, r.dtype), dtype=r.dtype)
    return tfd.LogNormal(
        loc=DeferredTensor(r, lambda x: x - 0.5 * scale**2.),
        scale=scale)


class LogNormalSoftplus(ExponentialFamily):
  """`LogNormal(loc=log(mean) - log(2) / 2, scale=sqrt(log(2)))`
  `mean = softplus(X @ weights)`.
  """

  def __init__(self, scale=np.sqrt(np.log(2.)), name=None):
    self._scale = scale
    super(LogNormalSoftplus, self).__init__(name=name)

  def _call(self, r):
    s = tf.cast(self._scale, r.dtype)
    mean = tf.math.softplus(r)
    grad_mean = tf.math.sigmoid(r)
    variance = mean**2.
    if tf.get_static_value(self._scale) != np.sqrt(np.log(2.)):
      s = tf.cast(self._scale, r.dtype)
      variance *= tf.math.expm1(s**2.)
    return mean, variance, grad_mean

  def _as_distribution(self, r):
    scale = DeferredTensor(
        self._scale, lambda x: tf.cast(x, r.dtype), dtype=r.dtype)
    return tfd.LogNormal(
        loc=DeferredTensor(
            r, lambda x: tf.math.log(tf.math.softplus(x)) - 0.5 * scale**2.),
        scale=scale)


class Normal(ExponentialFamily):
  """`Normal(loc=mean, scale=1)` where `mean = X @ weights`."""

  def __init__(self, scale=1., name=None):
    self._scale = scale
    super(Normal, self).__init__(name=name)

  def _call(self, r):
    mean = tf.identity(r)
    grad_mean = tf.ones_like(r)
    s = tf.cast(self._scale, r.dtype)
    variance = tf.fill(tf.shape(r), s**2.)
    return mean, variance, grad_mean

  def _as_distribution(self, r):
    scale = DeferredTensor(
        self._scale, lambda x: tf.cast(x, r.dtype), dtype=r.dtype)
    return tfd.Normal(loc=r, scale=scale)


class NormalReciprocal(ExponentialFamily):
  """`Normal(loc=mean, scale=1)` where `mean = 1 / (X @ weights)`."""

  def __init__(self, scale=1., name=None):
    self._scale = scale
    super(NormalReciprocal, self).__init__(name=name)

  def _call(self, r):
    mean = tf.math.reciprocal(r)
    grad_mean = -r**-2
    s = tf.cast(self._scale, r.dtype)
    variance = tf.fill(tf.shape(r), s**2.)
    return mean, variance, grad_mean

  def _as_distribution(self, r):
    scale = DeferredTensor(
        self._scale, lambda x: tf.cast(x, r.dtype), dtype=r.dtype)
    return tfd.Normal(loc=DeferredTensor(r, tf.math.reciprocal), scale=scale)


class NegativeBinomial(ExponentialFamily):
  """`NegativeBinomial(total_count, probs=mean / (mean + total_count))`.

  Where `mean = exp(X @ weights)`.
  """

  def __init__(self, total_count=1., name=None):
    self._total_count = total_count
    super(NegativeBinomial, self).__init__(name=name)

  def _call(self, r):
    mean = grad_mean = tf.math.exp(r)
    variance = mean + mean**2 / tf.cast(self._total_count, r.dtype)
    return mean, variance, grad_mean

  def _as_distribution(self, r):
    total_count = DeferredTensor(
        self._total_count, lambda x: tf.cast(x, r.dtype), dtype=r.dtype)
    return tfd.NegativeBinomial(
        total_count=total_count,
        logits=DeferredTensor(r, lambda x: x - tf.math.log(total_count)))


class NegativeBinomialSoftplus(ExponentialFamily):
  """`NegativeBinomial(total_count, probs=mean / (mean + total_count))`.

  Where `mean = softplus(X @ weights)`.
  """

  def __init__(self, total_count=1., name=None):
    self._total_count = total_count
    super(NegativeBinomialSoftplus, self).__init__(name=name)

  def _call(self, r):
    mean = tf.math.softplus(r)
    grad_mean = tf.math.sigmoid(r)
    variance = mean + mean**2 / tf.cast(self._total_count, r.dtype)
    return mean, variance, grad_mean

  def _as_distribution(self, r):
    total_count = DeferredTensor(
        self._total_count, lambda x: tf.cast(x, r.dtype), dtype=r.dtype)
    return tfd.NegativeBinomial(
        total_count=total_count,
        logits=DeferredTensor(r, lambda x: tf.math.log(  # pylint: disable=g-long-lambda
            tf.math.softplus(x)) - tf.math.log(total_count)))
