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

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import log_prob_ratio
from tensorflow_probability.python.internal import callable_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util

__all__ = [
    'IncrementLogProb',
]


class IncrementLogProb(distribution.AutoCompositeTensorDistribution):
  """A distribution representing an unnormalized measure on a singleton set.

  `IncrementLogProb` represents a "factor", which can also be thought of as a
  measure of the given size on a sample space consisting of a single element.
  Its raison d'être is to provide a computed offset to the log probability of a
  `JointDistribution`.  A `JointDistribution` containing an `IncrementLogProb`
  still represents a measure, but that measure is no longer in general a
  probability measure (i.e., the probability may no longer integrate to 1).

  Even though sampling from any measure represented by
  `IncrementLogProb` is information-free, `IncrementLogProb` retains a
  `sample` method for API compatibility with other Distributions.
  This `sample` method returns a (batch) shape-[0] Tensor with the
  same `dtype` as the `log_prob_increment` argument provided
  originally.

  """

  def __init__(
      self,
      log_prob_increment,
      validate_args=False,
      allow_nan_stats=False,  # pylint: disable=unused-argument
      log_prob_ratio_fn=None,
      name='IncrementLogProb',
      log_prob_increment_kwargs=None):
    """Construct a `IncrementLogProb` distribution-like object.

    Args:
      log_prob_increment: Float Tensor or callable returning a float Tensor. Log
        probability/density to increment by.
      validate_args: This argument is ignored, but is present because it is used
        in certain situations where `Distribution`s are expected.
      allow_nan_stats: This argument is ignored, but is present because it is
        used in certain situations where `Distribution`s are expected.
      log_prob_ratio_fn: Optional callable with signature `(p_kwargs, q_kwargs)
        -> log_prob_ratio`, used to implement a custom `p_log_prob_increment -
        q_log_prob_increment` computation.
      name: Python `str` name prefixed to Ops created by this class.
      log_prob_increment_kwargs: Passed to `log_prob_increment` if it is
        callable.
    """
    parameters = dict(locals())

    with tf.name_scope(name) as name:

      if log_prob_increment_kwargs is None:
        log_prob_increment_kwargs = {}

      if callable(log_prob_increment):
        log_prob_increment_fn = lambda: tensor_util.convert_nonref_to_tensor(  # pylint: disable=g-long-lambda
            log_prob_increment(**log_prob_increment_kwargs))
        spec = callable_util.get_output_spec(log_prob_increment_fn)
      else:
        if log_prob_increment_kwargs:
          raise ValueError('`log_prob_increment_kwargs` is only valid when '
                           '`log_prob_increment` is callable.')
        log_prob_increment = tensor_util.convert_nonref_to_tensor(
            log_prob_increment)
        log_prob_increment_fn = lambda: log_prob_increment
        spec = log_prob_increment

      self._log_prob_increment_fn = log_prob_increment_fn
      self._log_prob_increment = log_prob_increment
      self._dtype = spec.dtype
      self._static_batch_shape = spec.shape
      self._name = name
      self._validate_args = validate_args
      self._log_prob_ratio_fn = log_prob_ratio_fn
      self._log_prob_increment_kwargs = log_prob_increment_kwargs

      super().__init__(
          dtype=spec.dtype,
          reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # It's not obvious how to implement _parameter_properties for this
    # distribution, as we cannot determine the batch_ndims or event_ndims for
    # values in log_prob_increment_kwargs.
    raise NotImplementedError()

  @property
  def _composite_tensor_nonshape_params(self):
    params = ['log_prob_increment_kwargs']
    if not callable(self.log_prob_increment):
      params.append('log_prob_increment')
    return tuple(params)

  @property
  def _composite_tensor_shape_params(self):
    return ()

  @property
  def log_prob_increment(self):
    return self._log_prob_increment

  @property
  def log_prob_increment_kwargs(self):
    return self._log_prob_increment_kwargs

  def _log_prob(self, x):
    """Log probability mass function."""
    # TODO(axch): This method should do some shape checking on its argument to
    # be really consistent with the Distribution contract. If the input Tensor's
    # shape is not compatible, we should raise an error, as we do with other
    # Distributions when they are queried at a point of inconsistent shape.
    log_prob_increment = self._log_prob_increment_fn()
    return tf.broadcast_to(
        log_prob_increment,
        ps.broadcast_shape(ps.shape(log_prob_increment),
                           ps.shape(x)[:-1]))

  def _unnormalized_log_prob(self, x):
    return self._log_prob(x)

  def _batch_shape(self):
    return self._static_batch_shape

  def _batch_shape_tensor(self):
    if tensorshape_util.is_fully_defined(self._static_batch_shape):
      batch_shape = self._static_batch_shape
    else:
      batch_shape = ps.shape(self._log_prob_increment_fn())

    return batch_shape

  def _event_shape(self):
    return tf.TensorShape([0])

  def _event_shape_tensor(self):
    return [0]

  def _sample_n(self, n, seed=None):
    del seed
    return tf.zeros(
        ps.concat(
            [[n], self.batch_shape_tensor(),
             self.event_shape_tensor()], axis=0))

  def _default_event_space_bijector(self):
    return identity_bijector.Identity(validate_args=self.validate_args)


@log_prob_ratio.RegisterLogProbRatio(IncrementLogProb)
def _log_prob_increment_log_prob_ratio(p, x, q, y, name=None):
  """Computes the log-prob ratio."""
  del x, y
  # pylint: disable=protected-access
  with tf.name_scope(name or 'log_prob_increment_log_prob_ratio'):
    if (p._log_prob_ratio_fn is not None and
        p._log_prob_ratio_fn is q._log_prob_ratio_fn):
      return p._log_prob_ratio_fn(p._log_prob_increment_kwargs,
                                  q._log_prob_increment_kwargs)
    else:
      return p.unnormalized_log_prob(()) - q.unnormalized_log_prob(())
