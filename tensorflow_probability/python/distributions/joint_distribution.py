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
"""The `JointDistribution` base class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import tensorshape_util


__all__ = [
    'JointDistribution',
]


@six.add_metaclass(abc.ABCMeta)
class JointDistribution(distribution_lib.Distribution):
  """Joint distribution over one or more component distributions.

  This distribution enables both sampling and joint probability computation from
  a single model specification.

  A joint distribution is a collection of possibly interdependent distributions.


  #### Subclass Requirements

  Subclasses typically implement:

  - `_sample_distributions`: returns two `tuple`s: the first being an ordered
      sequence of `Distribution`-like instances the second being a sequence of
      `Tensor` samples, each one drawn from its corresponding
      `Distribution`-like instance.

  Subclasses also implement standard `Distribution` members:

  - `dtype`
  - `reparameterization_type`
  - `batch_shape_tensor`
  - `batch_shape`
  - `event_shape_tensor`
  - `event_shape`
  - `__getitem__`
  - `is_scalar_event`
  - `is_scalar_batch`

  Unlike usual `Distribution` subclasses, `JointDistribution` subclasses
  override the `Distribution` public-level functions so the subclass can return
  a `tuple` of values, one for each component `Distribution`-like instance.
  """

  @abc.abstractmethod
  def _sample_distributions(self, sample_shape=(), seed=None, value=None):
    raise NotImplementedError()

  def sample_distributions(self, sample_shape=(), seed=None, value=None,
                           name='sample_distributions'):
    """Generate samples and the (random) distributions.

    Note that a call to `sample()` without arguments will generate a single
    sample.

    Args:
      sample_shape: 0D or 1D `int32` `Tensor`. Shape of the generated samples.
      seed: Python integer seed for generating random numbers.
      value: `list` of `Tensor`s in `distribution_fn` order to use to
        parameterize other ("downstream") distribution makers.
        Default value: `None` (i.e., draw a sample from each distribution).
      name: name prepended to ops created by this function.
        Default value: `"sample_distributions"`.

    Returns:
      distributions: a `tuple` of `Distribution` instances for each of
        `distribution_fn`.
      samples: a `tuple` of `Tensor`s with prepended dimensions `sample_shape`
        for each of `distribution_fn`.
    """
    with self._name_scope(name):
      return self._sample_distributions(sample_shape, seed, value)

  def sample(self, sample_shape=(), seed=None, value=None, name='sample'):
    """Generate samples of the specified shape.

    Note that a call to `sample()` without arguments will generate a single
    sample.

    Args:
      sample_shape: 0D or 1D `int32` `Tensor`. Shape of the generated samples.
      seed: Python integer seed for generating random numbers.
      value: `list` of `Tensor`s in `distribution_fn` order to use to
        parameterize other ("downstream") distribution makers.
        Default value: `None` (i.e., draw a sample from each distribution).
      name: name prepended to ops created by this function.
        Default value: `"sample"`.

    Returns:
      samples: a `tuple` of `Tensor`s with prepended dimensions `sample_shape`
        for each of `distribution_fn`.
    """
    with self._name_scope(name):
      _, xs = self.sample_distributions(sample_shape, seed, value)
      return xs

  def log_prob_parts(self, value, name='log_prob_parts'):
    """Log probability density/mass function.

    Args:
      value: `list` of `Tensor`s in `distribution_fn` order for which we compute
        the `log_prob_parts` and to parameterize other ("downstream")
        distributions.
      name: name prepended to ops created by this function.
        Default value: `"log_prob_parts"`.

    Returns:
      log_prob_parts: a `tuple` of `Tensor`s representing the `log_prob` for
        each `distribution_fn` evaluated at each corresponding `value`.
    """
    if any(v is None for v in value):
      raise ValueError('No `value` part can be `None`.')
    with self._name_scope(name):
      return maybe_check_wont_broadcast(
          (d.log_prob(x) for d, x
           in zip(*self.sample_distributions(value=value))),
          self.validate_args)

  def prob_parts(self, value, name='prob_parts'):
    """Log probability density/mass function.

    Args:
      value: `list` of `Tensor`s in `distribution_fn` order for which we compute
        the `prob_parts` and to parameterize other ("downstream") distributions.
      name: name prepended to ops created by this function.
        Default value: `"prob_parts"`.

    Returns:
      prob_parts: a `tuple` of `Tensor`s representing the `prob` for
        each `distribution_fn` evaluated at each corresponding `value`.
    """
    if any(v is None for v in value):
      raise ValueError('No `value` part can be `None`.')
    with self._name_scope(name):
      return maybe_check_wont_broadcast(
          (d.prob(x) for d, x
           in zip(*self.sample_distributions(value=value))),
          self.validate_args)

  def _call_sample_n(self, sample_shape, seed, name, **kwargs):
    # Implemented here so generically calling `Distribution.sample` still works.
    # (This is needed for convenient Tensor coercion in tfp.layers.)
    return self.sample(sample_shape, seed, value=None, name=name)

  def _log_prob(self, xs):
    return sum(self.log_prob_parts(xs))


def maybe_check_wont_broadcast(parts, validate_args):
  """Verifies that `parts` dont broadcast."""
  parts = tuple(parts)  # So we can receive generators.
  if not validate_args:
    # Note: we don't try static validation because it is theoretically
    # possible that a user wants to take advantage of broadcasting.
    # Only when `validate_args` is `True` do we enforce the validation.
    return parts
  msg = 'Broadcasting probably indicates an error in model specification.'
  s = tuple(part.shape for part in parts)
  if all(tensorshape_util.is_fully_defined(s_) for s_ in s):
    if not all(a == b for a, b in zip(s[1:], s[:-1])):
      raise ValueError(msg)
    return parts
  assertions = [assert_util.assert_equal(a, b, message=msg)
                for a, b in zip(s[1:], s[:-1])]
  with tf.control_dependencies(assertions):
    return tuple(tf.identity(part) for part in parts)
