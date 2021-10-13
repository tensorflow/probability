# Copyright 2019 The TensorFlow Probability Authors.
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
"""Base class for variational layers for building neural networks."""
import collections

import tensorflow.compat.v2 as tf

from tensorflow_probability.python import random as tfp_random
from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.distributions import independent as independent_lib
from tensorflow_probability.python.distributions import mvn_diag as mvn_diag_lib
from tensorflow_probability.python.distributions import normal as normal_lib
from tensorflow_probability.python.experimental.nn import layers as layers_lib
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.util.seed_stream import SeedStream


__all__ = [
    'VariationalLayer',
]


# The following aliases ensure docstrings read more succinctly.
tfd = distribution_lib


def unpack_kernel_and_bias(weights):
  """Returns `kernel`, `bias` tuple."""
  if isinstance(weights, collections.Mapping):
    kernel = weights.get('kernel', None)
    bias = weights.get('bias', None)
  elif len(weights) == 1:
    kernel, bias = weights, None
  elif len(weights) == 2:
    kernel, bias = weights
  else:
    raise ValueError('Unable to unpack weights: {}.'.format(weights))
  return kernel, bias


class VariationalLayer(layers_lib.Layer):
  """Base class for all variational layers."""

  def __init__(
      self,
      posterior,
      prior,
      activation_fn=None,
      posterior_value_fn=tfd.Distribution.sample,
      seed=None,
      dtype=tf.float32,
      validate_args=False,
      name=None):
    """Base class for variational layers.

    Args:
      posterior: ...
      prior: ...
      activation_fn: ...
      posterior_value_fn: ...
      seed: ...
      dtype: ...
      validate_args: ...
      name: Python `str` prepeneded to ops created by this object.
        Default value: `None` (i.e., `type(self).__name__`).
    """
    super(VariationalLayer, self).__init__(
        validate_args=validate_args, name=name)
    self._posterior = posterior
    self._prior = prior
    self._activation_fn = activation_fn
    self._posterior_value_fn = posterior_value_fn
    self._posterior_value = None
    self._seed = SeedStream(seed, salt=self.name)
    self._dtype = dtype
    tf.nest.assert_same_structure(prior.dtype, posterior.dtype,
                                  check_types=False)

  @property
  def dtype(self):
    return self._dtype

  @property
  def posterior(self):
    return self._posterior

  @property
  def prior(self):
    return self._prior

  @property
  def activation_fn(self):
    return self._activation_fn

  @property
  def posterior_value_fn(self):
    return self._posterior_value_fn

  @property
  def posterior_value(self):
    return self._posterior_value


class VariationalReparameterizationKernelBiasLayer(VariationalLayer):
  """Variational reparameterization linear layer."""

  def __init__(
      self,
      posterior,
      prior,
      apply_kernel_fn,
      activation_fn=None,
      posterior_value_fn=tfd.Distribution.sample,
      unpack_weights_fn=unpack_kernel_and_bias,
      seed=None,
      dtype=tf.float32,
      validate_args=False,
      name=None):
    super(VariationalReparameterizationKernelBiasLayer, self).__init__(
        posterior,
        prior,
        activation_fn=activation_fn,
        posterior_value_fn=posterior_value_fn,
        seed=seed,
        dtype=dtype,
        validate_args=validate_args,
        name=name)
    self._apply_kernel_fn = apply_kernel_fn
    self._unpack_weights_fn = unpack_weights_fn

  @property
  def unpack_weights_fn(self):
    return self._unpack_weights_fn

  def __call__(self, x, **kwargs):
    x = tf.convert_to_tensor(x, dtype=self.dtype, name='x')
    self._posterior_value = self.posterior_value_fn(
        self.posterior, seed=self._seed())  # pylint: disable=not-callable
    kernel, bias = self.unpack_weights_fn(self.posterior_value)  # pylint: disable=not-callable
    y = x
    if kernel is not None:
      y = self._apply_kernel_fn(y, kernel)
    if bias is not None:
      y = y + bias
    if self.activation_fn is not None:
      y = self.activation_fn(y)  # pylint: disable=not-callable
    return y


class VariationalFlipoutKernelBiasLayer(VariationalLayer):
  """Variational flipout linear layer."""

  def __init__(
      self,
      posterior,
      prior,
      apply_kernel_fn,
      activation_fn=None,
      posterior_value_fn=tfd.Distribution.sample,
      unpack_weights_fn=unpack_kernel_and_bias,
      seed=None,
      dtype=tf.float32,
      validate_args=False,
      name=None):
    super(VariationalFlipoutKernelBiasLayer, self).__init__(
        posterior,
        prior,
        activation_fn=activation_fn,
        posterior_value_fn=posterior_value_fn,
        seed=seed,
        dtype=dtype,
        validate_args=validate_args,
        name=name)
    self._apply_kernel_fn = apply_kernel_fn
    self._unpack_weights_fn = unpack_weights_fn

  @property
  def unpack_weights_fn(self):
    return self._unpack_weights_fn

  def __call__(self, x, **kwargs):
    x = tf.convert_to_tensor(x, dtype=self.dtype, name='x')
    self._posterior_value = self.posterior_value_fn(
        self.posterior, seed=self._seed())  # pylint: disable=not-callable
    kernel, bias = self.unpack_weights_fn(self.posterior_value)  # pylint: disable=not-callable
    y = x

    if kernel is not None:
      kernel_dist, _ = self.unpack_weights_fn(  # pylint: disable=not-callable
          self.posterior.sample_distributions(value=self.posterior_value)[0])
      kernel_loc, kernel_scale = get_spherical_normal_loc_scale(kernel_dist)

      # batch_size = tf.shape(x)[0]
      # sign_input_shape = ([batch_size] +
      #                     [1] * self._rank +
      #                     [self._input_channels])
      y *= tfp_random.rademacher(ps.shape(y),
                                 dtype=y.dtype,
                                 seed=self._seed())
      kernel_perturb = normal_lib.Normal(loc=0., scale=kernel_scale)
      y = self._apply_kernel_fn(   # E.g., tf.matmul.
          y,
          kernel_perturb.sample(seed=self._seed()))
      y *= tfp_random.rademacher(ps.shape(y),
                                 dtype=y.dtype,
                                 seed=self._seed())
      y += self._apply_kernel_fn(x, kernel_loc)

    if bias is not None:
      y = y + bias

    if self.activation_fn is not None:
      y = self.activation_fn(y)  # pylint: disable=not-callable

    return y


def get_spherical_normal_loc_scale(d):
  if isinstance(d, independent_lib.Independent):
    return get_spherical_normal_loc_scale(d.distribution)
  if isinstance(d, (normal_lib.Normal, mvn_diag_lib.MultivariateNormalDiag)):
    return d.loc, d.scale
  raise TypeError('Expected kernel `posterior` to be spherical Normal; '
                  'saw: "{}".'.format(type(d).__name__))
