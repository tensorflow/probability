# Copyright 2021 The TensorFlow Probability Authors.
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
"""The `IncrementLogProb` class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.distributions import log_prob_ratio
from tensorflow_probability.python.internal import callable_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util

__all__ = [
    'IncrementLogProb',
]


class IncrementLogProb(object):
  """A distribution-like object for an unnormalized measure on a singleton set.

  `IncrementLogProb` is a distribution-like class that represents a
  "factor", which can also be thought of as a measure of the given size
  on a sample space consisting of a single element.  Its raison d'être
  is to provide a computed offset to the log probability of a
  `JointDistribution`.  A `JointDistribution` containing an
  `IncrementLogProb` still represents a measure, but that measure is
  no longer in general a probability measure (i.e., the probability
  may no longer integrate to 1).

  Even though sampling from any measure represented by
  `IncrementLogProb` is information-free, `IncrementLogProb` retains a
  `sample` method for API compatibility with other Distributions.
  This `sample` method returns a (batch) shape-[0] Tensor with the
  same `dtype` as the `increment_log_prob` argument provided
  originally.

  """

  def __init__(
      self,
      log_prob_increment,
      validate_args=False,
      allow_nan_stats=False,  # pylint: disable=unused-argument
      reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,  # pylint: disable=unused-argument
      log_prob_ratio_fn=None,
      name='IncrementLogProb',
      **kwargs):
    """Construct a `IncrementLogProb` distribution-like object.

    Args:
      log_prob_increment: Float Tensor or callable returning a float Tensor. Log
        probability/density to increment by.
      validate_args: This argument is ignored, but is present because it is used
        in certain situations where `Distribution`s are expected.
      allow_nan_stats: This argument is ignored, but is present because it is
        used in certain situations where `Distribution`s are expected.
      reparameterization_type: This argument is ignored, but is present because
        it is used in certain situations where `Distribution`s are expected.
      log_prob_ratio_fn: Optional callable with signature `(p_kwargs, q_kwargs)
        -> log_prob_ratio`, used to implement a custom `p_log_prob_increment -
        q_log_prob_increment` computation.
      name: Python `str` name prefixed to Ops created by this class.
      **kwargs: Passed to `log_prob_increment` if it is callable.
    """
    self._parameters = dict(locals())

    with tf.name_scope(name) as name:
      if callable(log_prob_increment):
        log_prob_increment_fn = lambda: tensor_util.convert_nonref_to_tensor(  # pylint: disable=g-long-lambda
            log_prob_increment(**kwargs))
        spec = callable_util.get_output_spec(log_prob_increment_fn)
      else:
        if kwargs:
          raise ValueError(
              '`kwargs` is only valid when `log_prob_increment` is callable.')
        log_prob_increment = tensor_util.convert_nonref_to_tensor(
            log_prob_increment)
        log_prob_increment_fn = lambda: log_prob_increment
        spec = log_prob_increment

      self._log_prob_increment_fn = log_prob_increment_fn
      self._log_prob_increment = log_prob_increment
      self._dtype = spec.dtype
      self._batch_shape = spec.shape
      self._name = name
      self._validate_args = validate_args
      self._log_prob_ratio_fn = log_prob_ratio_fn
      self._kwargs = kwargs

  @property
  def validate_args(self):
    return self._validate_args

  @property
  def log_prob_increment(self):
    return self._log_prob_increment

  @property
  def name(self):
    return self._name

  @property
  def parameters(self):
    return dict(self._parameters)

  @property
  def dtype(self):
    return self._dtype

  def log_prob(self, _):
    """Log probability mass function."""
    # TODO(axch): Arguably, this method should do some shape checking
    # on its argument to be really consistent with the Distribution
    # contract.  To wit, it should expect the argument to be a single
    # Tensor of shape `sample_shape + batch_shape + [0]`, and should
    # return `self.log_prob_increment` (which is of shape
    # `batch_shape`) broadcasted to shape `sample_shape +
    # batch_shape`.  This broadcasting only appears here because this
    # is the unique (currently present) Distribution whose `log_prob`
    # doesn't depend on the query point.  If the input Tensor's shape
    # is not compatible, we should raise an error, as we do with other
    # Distributions when they are queried at a point of inconsistent
    # shape.
    #
    # Status quo is OK for now, because IncrementLogProb is expected to
    # only be used in JointDistribution-type contexts, where the implicit
    # summation will produce the desired broadcasting.
    return self.log_prob_increment

  def unnormalized_log_prob(self, _):
    return self._log_prob_increment_fn()

  @property
  def batch_shape(self):
    return self._batch_shape

  def batch_shape_tensor(self, name='batch_shape_tensor'):
    """Shape of a single sample from a single event index as a 1-D `Tensor`.

    The batch dimensions are indexes into independent, non-identical
    parameterizations of this distribution.

    Args:
      name: name to give to the op

    Returns:
      batch_shape: `Tensor`.
    """
    with tf.name_scope(name):

      if tensorshape_util.is_fully_defined(self.batch_shape):
        batch_shape = self.batch_shape
      else:
        batch_shape = ps.shape(self._log_prob_increment_fn())

      return ps.identity(
          ps.convert_to_shape_tensor(batch_shape, name='batch_shape'))

  @property
  def event_shape(self):
    return tf.TensorShape([0])

  def event_shape_tensor(self, name='event_shape_tensor'):
    """Shape of a single sample from a single batch as a 1-D int32 `Tensor`.

    Args:
      name: name to give to the op

    Returns:
      event_shape: `Tensor`.
    """
    with tf.name_scope(name):

      return ps.convert_to_shape_tensor(self.event_shape, name='event_shape')

  @property
  def allow_nan_stats(self):
    return False

  @property
  def reparameterization_type(self):
    return False

  @classmethod
  def _parameter_control_dependencies(cls, *_, **__):
    return []

  @property
  def experimental_shard_axis_names(self):
    """The list or structure of lists of active shard axis names."""
    return []

  def sample(self, sample_shape=(), seed=None, name='sample'):  # pylint: disable=unused-argument
    return tf.zeros(
        ps.concat(
            [
                # sample_shape might be a scalar
                ps.reshape(
                    ps.convert_to_shape_tensor(sample_shape, tf.int32),
                    shape=[-1]),
                self.batch_shape_tensor(),
                self.event_shape_tensor()
            ],
            axis=0))

  def experimental_default_event_space_bijector(self):
    return identity_bijector.Identity(validate_args=self.validate_args)


@log_prob_ratio.RegisterLogProbRatio(IncrementLogProb)
def _increment_log_prob_log_prob_ratio(p, x, q, y, name=None):
  del x, y
  # pylint: disable=protected-access
  with tf.name_scope(name or 'increment_log_prob_log_prob_ratio'):
    if (p._log_prob_ratio_fn is not None and
        p._log_prob_ratio_fn is q._log_prob_ratio_fn):
      return p._log_prob_ratio_fn(p._kwargs, q._kwargs)
    else:
      return p.unnormalized_log_prob(()) - q.unnormalized_log_prob(())
