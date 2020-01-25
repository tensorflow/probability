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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow.compat.v2 as tf

from discussion.nn import layers as layers_lib
from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.distributions import independent as independent_lib
from tensorflow_probability.python.distributions import kullback_leibler as kl_lib
from tensorflow_probability.python.distributions import mvn_diag as mvn_diag_lib
from tensorflow_probability.python.distributions import normal as normal_lib
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal.reparameterization import FULLY_REPARAMETERIZED
from tensorflow_probability.python.math.random_ops import random_rademacher
from tensorflow_probability.python.monte_carlo import expectation
from tensorflow_probability.python.util.seed_stream import SeedStream


__all__ = [
    'VariationalLayer',
]


# The following aliases ensure docstrings read more succinctly.
tfd = distribution_lib


def kl_divergence_monte_carlo(q, r, w):
  """Monte Carlo KL Divergence."""
  return expectation(
      lambda w: q.log_prob(w) - r.log_prob(w),
      samples=w,
      log_prob=q.log_prob,
      use_reparameterization=all(
          rt == FULLY_REPARAMETERIZED
          for rt in tf.nest.flatten(q.reparameterization_type)),
      axis=())


def kl_divergence_exact(q, r, w):  # pylint: disable=unused-argument
  """Exact KL Divergence."""
  return kl_lib.kl_divergence(q, r)


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
      penalty_weight=None,
      posterior_penalty_fn=kl_divergence_monte_carlo,
      posterior_value_fn=tfd.Distribution.sample,
      seed=None,
      dtype=tf.float32,
      name=None):
    """Base class for variational layers.

    # mean ==> penalty_weight =          1 / train_size
    # sum  ==> penalty_weight = batch_size / train_size

    Args:
      posterior: ...
      prior: ...
      penalty_weight: ...
      posterior_penalty_fn: ...
      posterior_value_fn: ...
      seed: ...
      dtype: ...
      name: Python `str` prepeneded to ops created by this object.
        Default value: `None` (i.e., `type(self).__name__`).
    """
    super(VariationalLayer, self).__init__(name=name)
    self._posterior = posterior
    self._prior = prior
    self._penalty_weight = penalty_weight
    self._posterior_penalty_fn = posterior_penalty_fn
    self._posterior_value_fn = posterior_value_fn
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
  def penalty_weight(self):
    return self._penalty_weight

  @property
  def posterior_penalty_fn(self):
    return self._posterior_penalty_fn

  @property
  def posterior_value_fn(self):
    return self._posterior_value_fn

  def eval(self, inputs, is_training=True, **kwargs):
    inputs = tf.convert_to_tensor(inputs, dtype=self.dtype, name='inputs')
    w = self.posterior_value_fn(self.posterior, seed=self._seed())  # pylint: disable=not-callable
    if is_training:
      penalty = self.posterior_penalty_fn(self.posterior, self.prior, w)  # pylint: disable=not-callable
      if penalty is not None and self.penalty_weight is not None:
        penalty *= tf.cast(self.penalty_weight, dtype=penalty.dtype)
    else:
      penalty = None
    outputs = self._eval(inputs, w, **kwargs)
    self._set_extra_loss(penalty)
    self._set_extra_result(w)
    return outputs

  def _eval(self, inputs, weights):
    raise NotImplementedError('Subclass failed to implement `_eval`.')


class VariationalReparameterizationKernelBiasLayer(VariationalLayer):
  """Variational reparameterization linear layer."""

  def __init__(
      self,
      posterior,
      prior,
      apply_kernel_fn,
      penalty_weight=None,
      posterior_penalty_fn=kl_divergence_monte_carlo,
      posterior_value_fn=tfd.Distribution.sample,
      unpack_weights_fn=unpack_kernel_and_bias,
      seed=None,
      dtype=tf.float32,
      name=None):
    super(VariationalReparameterizationKernelBiasLayer, self).__init__(
        posterior,
        prior,
        penalty_weight=penalty_weight,
        posterior_penalty_fn=posterior_penalty_fn,
        posterior_value_fn=posterior_value_fn,
        seed=seed,
        dtype=dtype,
        name=name)
    self._apply_kernel_fn = apply_kernel_fn
    self._unpack_weights_fn = unpack_weights_fn

  @property
  def unpack_weights_fn(self):
    return self._unpack_weights_fn

  def _eval(self, x, weights):
    kernel, bias = self.unpack_weights_fn(weights)  # pylint: disable=not-callable
    y = x
    if kernel is not None:
      y = self._apply_kernel_fn(y, kernel)
    if bias is not None:
      y = y + bias
    return y


class VariationalFlipoutKernelBiasLayer(VariationalLayer):
  """Variational flipout linear layer."""

  def __init__(
      self,
      posterior,
      prior,
      apply_kernel_fn,
      penalty_weight=None,
      posterior_penalty_fn=kl_divergence_monte_carlo,
      posterior_value_fn=tfd.Distribution.sample,
      unpack_weights_fn=unpack_kernel_and_bias,
      seed=None,
      dtype=tf.float32,
      name=None):
    super(VariationalFlipoutKernelBiasLayer, self).__init__(
        posterior,
        prior,
        penalty_weight=penalty_weight,
        posterior_penalty_fn=posterior_penalty_fn,
        posterior_value_fn=posterior_value_fn,
        seed=seed,
        dtype=dtype,
        name=name)
    self._apply_kernel_fn = apply_kernel_fn
    self._unpack_weights_fn = unpack_weights_fn

  @property
  def unpack_weights_fn(self):
    return self._unpack_weights_fn

  def _eval(self, x, weights):
    kernel, bias = self.unpack_weights_fn(weights)  # pylint: disable=not-callable
    y = x

    if kernel is not None:
      kernel_dist, _ = self.unpack_weights_fn(  # pylint: disable=not-callable
          self.posterior.sample_distributions(value=weights)[0])
      kernel_loc, kernel_scale = get_spherical_normal_loc_scale(kernel_dist)

      # batch_size = tf.shape(x)[0]
      # sign_input_shape = ([batch_size] +
      #                     [1] * self._rank +
      #                     [self._input_channels])
      y *= random_rademacher(prefer_static.shape(y),
                             dtype=y.dtype,
                             seed=self._seed())
      kernel_perturb = normal_lib.Normal(loc=0., scale=kernel_scale)
      y = self._apply_kernel_fn(   # E.g., tf.matmul.
          y,
          kernel_perturb.sample(seed=self._seed()))
      y *= random_rademacher(prefer_static.shape(y),
                             dtype=y.dtype,
                             seed=self._seed())
      y += self._apply_kernel_fn(x, kernel_loc)

    if bias is not None:
      y = y + bias

    return y


def get_spherical_normal_loc_scale(d):
  if isinstance(d, independent_lib.Independent):
    return get_spherical_normal_loc_scale(d.distribution)
  if isinstance(d, (normal_lib.Normal, mvn_diag_lib.MultivariateNormalDiag)):
    return d.loc, d.scale
  raise TypeError('Expected kernel `posterior` to be spherical Normal; '
                  'saw: "{}".'.format(type(d).__name__))
