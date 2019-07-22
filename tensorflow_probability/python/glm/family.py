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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import numpy as np

import tensorflow as tf
from tensorflow_probability.python import bijectors
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.math.gradient import value_and_gradient


__all__ = [
    'Bernoulli',
    'BernoulliNormalCDF',
    'CustomExponentialFamily',
    'GammaExp',
    'GammaSoftplus',
    'ExponentialFamily',
    'LogNormal',
    'LogNormalSoftplus',
    'Normal',
    'NormalReciprocal',
    'Poisson',
    'PoissonSoftplus',
]


class ExponentialFamily(object):
  """Specifies a mean-value parameterized exponential family.

  Subclasses implement [exponential-family distribution](
  https://en.wikipedia.org/wiki/Exponential_family) properties (e.g.,
  `log_prob`, `variance`) as a function of a real-value which is transformed via
  some [link function](
  https://en.wikipedia.org/wiki/Generalized_linear_model#Link_function)
  to be interpreted as the distribution's mean. The distribution is
  parameterized by this mean, i.e., "mean-value parametrized."

  Subclasses are typically used to specify a Generalized Linear Model (GLM). A
  [GLM]( https://en.wikipedia.org/wiki/Generalized_linear_model) is a
  generalization of linear regression which enables efficient fitting of
  log-likelihood losses beyond just assuming `Normal` noise. See `tfp.glm.fit`
  for more details.

  Subclasses must implement `_call`, `_log_prob`, and `_is_canonical`. In
  context of `tfp.glm.fit`, these functions are used to find the best fitting
  weights for given model matrix ("X") and responses ("Y").
  """

  def __init__(self, name=None):
    """Creates the ExponentialFamily.

    Args:
      name: Python `str` used as TF namescope for ops created by member
        functions. Default value: `None` (i.e., the subclass name).
    """
    if not name or name[-1] != '/':  # `name` is not a name scope.
      with tf.compat.v1.name_scope(name or type(self).__name__) as name:
        pass
    self._name = name

  @property
  def is_canonical(self):
    """Returns `True` when `variance(r) == grad_mean(r)` for all `r`."""
    return self._is_canonical

  @property
  def name(self):
    """Returns TF namescope prefixed to ops created by member functions."""
    return self._name

  def _call(self, predicted_linear_response):
    """Implements the __call__ computation."""
    raise NotImplementedError('`_call` is not implemented.')

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
        `tf.matmul(model_matrix, weights)`.
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
    with self._name_scope(name, 'call', [predicted_linear_response]):
      predicted_linear_response = tf.convert_to_tensor(
          value=predicted_linear_response, name='predicted_linear_response')
      return self._call(predicted_linear_response)

  def _log_prob(self, response, predicted_linear_response):
    """Implements the log-probability computation."""
    raise NotImplementedError('`_log_prob` is not implemented.')

  def log_prob(self, response, predicted_linear_response, name=None):
    """Computes `D(param=mean(r)).log_prob(response)` for linear response, `r`.

    Args:
      response: `float`-like `Tensor` representing observed ("actual")
        responses.
      predicted_linear_response: `float`-like `Tensor` corresponding to
        `tf.matmul(model_matrix, weights)`.
      name: Python `str` used as TF namescope for ops created by member
        functions. Default value: `None` (i.e., 'log_prob').

    Returns:
      log_prob: `Tensor` with shape and dtype of `predicted_linear_response`
        representing the distribution prescribed log-probability of the observed
        `response`s.
    """

    with self._name_scope(
        name, 'log_prob', [response, predicted_linear_response]):
      dtype = dtype_util.common_dtype([response, predicted_linear_response])
      response = tf.convert_to_tensor(
          value=response, dtype=dtype, name='response')
      predicted_linear_response = tf.convert_to_tensor(
          value=predicted_linear_response, name='predicted_linear_response')
      return self._log_prob(response, predicted_linear_response)

  def __str__(self):
    return 'tfp.glm.family.{type_name}(\'{self_name}\')'.format(
        type_name=type(self).__name__,
        self_name=self.name)

  def __repr__(self):
    return '<tfp.glm.family.{type_name} \'{self_name}\'>'.format(
        type_name=type(self).__name__,
        self_name=self.name)

  @contextlib.contextmanager
  def _name_scope(self, name=None, default_name=None, values=None):
    """Helper function to standardize op scope."""
    with tf.compat.v1.name_scope(self.name):
      with tf.compat.v1.name_scope(
          name, default_name, values=values or []) as scope:
        yield scope


class CustomExponentialFamily(
    ExponentialFamily):
  """Constucts GLM from arbitrary distribution and inverse link function."""

  def __init__(self,
               distribution_fn,
               linear_model_to_mean_fn,
               is_canonical=False,
               name=None):
    """Creates the `CustomExponentialFamily`.

    Args:
      distribution_fn: Python `callable` which returns a
        `tf.distribution.Distribution`-like instance from a single input
        representing the distribution's required `mean`, i.e.,
        `mean = linear_model_to_mean_fn(matmul(model_matrix, weights))`.
      linear_model_to_mean_fn: Python `callable` which returns the
        distribution's required mean as computed from the predicted linear
        response, `matmul(model_matrix, weights)`.
      is_canonical: Python `bool` indicating that taken together,
        `distribution_fn` and `linear_model_to_mean_fn` imply that the
        distribution's `variance` is equivalent to `d/dr
        linear_model_to_mean_fn(r)`.
      name: Python `str` used as TF namescope for ops created by member
        functions. Default value: `None` (i.e., the subclass name).
    """
    super(CustomExponentialFamily, self).__init__(name)
    self._distribution_fn = distribution_fn
    self._inverse_link_fn = (
        linear_model_to_mean_fn.forward
        if isinstance(linear_model_to_mean_fn, bijectors.Bijector)
        else linear_model_to_mean_fn)
    self._is_canonical = is_canonical

  @property
  def distribution_fn(self):
    return self._distribution_fn

  @property
  def linear_model_to_mean_fn(self):
    return self._inverse_link_fn

  def _call(self, r):
    if self._inverse_link_fn is None:
      # Interpret `None` as the identity function.
      mean, grad_mean = r, tf.ones_like(r)
    else:
      mean, grad_mean = value_and_gradient(self._inverse_link_fn, r)
    variance = self._distribution_fn(mean).variance()
    return mean, variance, grad_mean

  def _log_prob(self, y, r):
    mean = self._inverse_link_fn(r)
    return self._distribution_fn(mean).log_prob(y)


class Bernoulli(ExponentialFamily):
  """`Bernoulli(probs=mean)` where `mean = sigmoid(matmul(X, weights))`."""

  _is_canonical = True

  def _call(self, r):
    mean = tf.nn.sigmoid(r)
    variance = grad_mean = mean * tf.nn.sigmoid(-r)
    return mean, variance, grad_mean

  def _log_prob(self, y, r):
    return tfd.Bernoulli(logits=r).log_prob(y)


class BernoulliNormalCDF(ExponentialFamily):
  """`Bernoulli(probs=mean)` where
  `mean = Normal(0, 1).cdf(matmul(X, weights))`."""

  _is_canonical = False

  def _call(self, r):
    dtype = r.dtype.as_numpy_dtype
    d = tfd.Normal(loc=np.array(0, dtype), scale=np.array(1, dtype))
    mean = d.cdf(r)
    # var = cdf(r) * cdf(-r) but cdf(-r) = 1 - cdf(r) = survival_function(r).
    variance = mean * d.survival_function(r)
    grad_mean = d.prob(r)
    return mean, variance, grad_mean

  def _log_prob(self, y, r):
    dtype = r.dtype.as_numpy_dtype
    d = tfd.Normal(loc=np.array(0, dtype), scale=np.array(1, dtype))
    # logit(ncdf(r)) = log(ncdf(r)) - log(1-ncdf(r)) = logncdf(r) - lognsf(r).
    logits = d.log_cdf(r) - d.log_survival_function(r)
    return tfd.Bernoulli(logits=logits).log_prob(y)


class GammaExp(ExponentialFamily):
  """`Gamma(concentration=1, rate=1 / mean)` where
  `mean = exp(matmul(X, weights))`."""

  _is_canonical = False

  def _call(self, r):
    mean = grad_mean = tf.exp(r)
    variance = mean**2
    return mean, variance, grad_mean

  def _log_prob(self, y, r):
    dtype = r.dtype.as_numpy_dtype
    g = tfd.Gamma(concentration=np.array(1, dtype), rate=tf.exp(-r))
    return g.log_prob(y)


class GammaSoftplus(ExponentialFamily):
  """`Gamma(concentration=1, rate=1 / mean)` where
  `mean = softplus(matmul(X, weights))`."""

  _is_canonical = False

  def _call(self, r):
    mean = tf.nn.softplus(r)
    variance = mean**2
    grad_mean = tf.nn.sigmoid(r)
    return mean, variance, grad_mean

  def _log_prob(self, y, r):
    dtype = r.dtype.as_numpy_dtype
    mean = tf.nn.softplus(r)
    g = tfd.Gamma(concentration=np.array(1, dtype), rate=1./mean)
    return g.log_prob(y)


class Poisson(ExponentialFamily):
  """`Poisson(rate=mean)` where `mean = exp(matmul(X, weights))`."""

  _is_canonical = True

  def _call(self, r):
    mean = variance = grad_mean = tf.exp(r)
    return mean, variance, grad_mean

  def _log_prob(self, y, r):
    return tfd.Poisson(log_rate=r).log_prob(y)


class PoissonSoftplus(ExponentialFamily):
  """`Poisson(rate=mean)` where `mean = softplus(matmul(X, weights))`."""

  _is_canonical = False

  def _call(self, r):
    mean = variance = tf.nn.softplus(r)
    grad_mean = tf.nn.sigmoid(r)
    return mean, variance, grad_mean

  def _log_prob(self, y, r):
    return tfd.Poisson(rate=tf.nn.softplus(r)).log_prob(y)


class LogNormal(ExponentialFamily):
  """`LogNormal(loc=log(mean) - log(2) / 2, scale=sqrt(log(2)))` where
  `mean = exp(matmul(X, weights))`.
  """

  _is_canonical = False

  def _call(self, r):
    mean = grad_mean = tf.exp(r)
    variance = mean**2.
    return mean, variance, grad_mean

  def _log_prob(self, y, r):
    dtype = r.dtype.as_numpy_dtype
    log_y = tf.math.log(y)
    s2 = np.log(2.).astype(dtype)
    return -log_y + tfd.Normal(
        loc=r - 0.5 * s2,
        scale=np.sqrt(s2)).log_prob(log_y)


class LogNormalSoftplus(ExponentialFamily):
  """`LogNormal(loc=log(mean) - log(2) / 2, scale=sqrt(log(2)))`
  `mean = softplus(matmul(X, weights))`.
  """

  _is_canonical = False

  def _call(self, r):
    mean = tf.nn.softplus(r)
    variance = mean**2.
    grad_mean = tf.nn.sigmoid(r)
    return mean, variance, grad_mean

  def _log_prob(self, y, r):
    dtype = r.dtype.as_numpy_dtype
    log_y = tf.math.log(y)
    s2 = np.log(2.).astype(dtype)
    return tfd.Normal(
        loc=tf.math.log(tf.nn.softplus(r)) - 0.5 * s2,
        scale=np.sqrt(s2)).log_prob(log_y) - log_y


class Normal(ExponentialFamily):
  """`Normal(loc=mean, scale=1)` where `mean = matmul(X, weights)`."""

  _is_canonical = True

  def _call(self, r):
    mean = tf.identity(r)
    variance = grad_mean = tf.ones_like(r)
    return mean, variance, grad_mean

  def _log_prob(self, y, r):
    dtype = r.dtype.as_numpy_dtype
    return tfd.Normal(loc=r, scale=np.array(1, dtype)).log_prob(y)


class NormalReciprocal(ExponentialFamily):
  """`Normal(loc=mean, scale=1)` where `mean = 1 / matmul(X, weights)`."""

  _is_canonical = False

  def _call(self, r):
    mean = 1. / r
    variance = tf.ones_like(r)
    grad_mean = -1. / r**2
    return mean, variance, grad_mean

  def _log_prob(self, y, r):
    dtype = r.dtype.as_numpy_dtype
    return tfd.Normal(loc=1. / r, scale=np.array(1, dtype)).log_prob(y)
