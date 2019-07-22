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
"""FiniteDiscrete distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import categorical
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import distribution_util as dist_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization

__all__ = [
    'FiniteDiscrete',
]


class FiniteDiscrete(distribution.Distribution):
  """The finite discrete distribution.

  The FiniteDiscrete distribution is parameterized by either probabilities or
  log-probabilities of a set of `K` possible outcomes, which is defined by
  a strictly ascending list of `K` values.

  #### Mathematical Details

  The probability mass function (pmf) is,

  ```none
  pmf(x; pi, qi) = prod_j pi_j**[x == qi_j]
  ```

  #### Examples

  ```python

  # Initialize a discrete distribution with 4 possible outcomes and the 2nd
  # outcome being most likely.
  dist = FiniteDiscrete([1., 2., 4., 8.], probs=[0.1, 0.4, 0.3, 0.2])
  dist.prob(2.)
  # ==> 0.4

  # Using logits to initialize a discrete distribution with 4 possible outcomes
  # and the 2nd outcome being most likely.
  dist = FiniteDiscrete([1., 2., 4., 8.], logits=np.log([0.1, 0.4, 0.3, 0.2]))
  dist.prob(2.)
  # ==> 0.4
  ```

  """

  def __init__(self,
               outcomes,
               logits=None,
               probs=None,
               rtol=None,
               atol=None,
               validate_args=False,
               allow_nan_stats=True,
               name='FiniteDiscrete'):
    """Construct a finite discrete contribution.

    Args:
      outcomes: A 1-D floating or integer `Tensor`, representing a list of
        possible outcomes in strictly ascending order.
      logits: A floating N-D `Tensor`, `N >= 1`, representing the log
        probabilities of a set of FiniteDiscrete distributions. The first `N -
        1` dimensions index into a batch of independent distributions and the
        last dimension represents a vector of logits for each discrete value.
        Only one of `logits` or `probs` should be passed in.
      probs: A floating  N-D `Tensor`, `N >= 1`, representing the probabilities
        of a set of FiniteDiscrete distributions. The first `N - 1` dimensions
        index into a batch of independent distributions and the last dimension
        represents a vector of probabilities for each discrete value. Only one
        of `logits` or `probs` should be passed in.
      rtol: `Tensor` with same `dtype` as `outcomes`. The relative tolerance for
        floating number comparison. Only effective when `outcomes` is a floating
        `Tensor`. Default is `10 * eps`.
      atol: `Tensor` with same `dtype` as `outcomes`. The absolute tolerance for
        floating number comparison. Only effective when `outcomes` is a floating
        `Tensor`. Default is `10 * eps`.
      validate_args:  Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may render incorrect outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value '`NaN`' to indicate the
        result is undefined. When `False`, an exception is raised if one or more
        of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with tf1.name_scope(name) as name:
      self._outcomes = tf.convert_to_tensor(outcomes, name='outcomes')
      if validate_args:
        assertions = _maybe_validate_args(self._outcomes, logits, probs,
                                          validate_args)
        with tf.control_dependencies(assertions):
          self._outcomes = tf.identity(self._outcomes)
      if dtype_util.is_floating(self._outcomes.dtype):
        eps = np.finfo(dtype_util.as_numpy_dtype(self._outcomes.dtype)).eps
        self._rtol = 10 * eps if rtol is None else rtol
        self._atol = 10 * eps if atol is None else atol
      else:
        self._rtol = None
        self._atol = None
      self._categorical = categorical.Categorical(
          logits=logits,
          probs=probs,
          dtype=tf.int32,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats)
    super(FiniteDiscrete, self).__init__(
        dtype=self._outcomes.dtype,
        reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        name=name)

  @classmethod
  def _params_event_ndims(cls):
    # outcomes is currently not sliceable.
    return dict(logits=1, probs=1)

  @property
  def outcomes(self):
    return self._outcomes

  @property
  def logits(self):
    """Input argument `logits`."""
    return self._categorical.logits

  @property
  def probs(self):
    """Input argument `probs`."""
    return self._categorical.probs

  def _batch_shape_tensor(self):
    return self._categorical.batch_shape_tensor()

  def _batch_shape(self):
    return self._categorical.batch_shape

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _cdf(self, x):
    x = tf.convert_to_tensor(x, name='x')
    flat_x = tf.reshape(x, shape=[-1])
    upper_bound = tf.searchsorted(self.outcomes, values=flat_x, side='right')
    values_at_ub = tf.gather(
        self.outcomes,
        indices=tf.minimum(upper_bound,
                           dist_util.prefer_static_shape(self.outcomes)[-1] -
                           1))
    should_use_upper_bound = self._is_equal_or_close(flat_x, values_at_ub)
    indices = tf.where(should_use_upper_bound, upper_bound, upper_bound - 1)
    return self._categorical.cdf(
        tf.reshape(indices, shape=dist_util.prefer_static_shape(x)))

  def _entropy(self):
    return self._categorical.entropy()

  def _is_equal_or_close(self, a, b):
    if dtype_util.is_integer(self.outcomes.dtype):
      return tf.equal(a, b)
    return tf.abs(a - b) < self._atol + self._rtol * tf.abs(b)

  def _log_prob(self, x):
    x = tf.convert_to_tensor(x, name='x')
    right_indices = tf.minimum(
        tf.size(self.outcomes) - 1,
        tf.reshape(
            tf.searchsorted(
                self.outcomes, values=tf.reshape(x, shape=[-1]), side='right'),
            dist_util.prefer_static_shape(x)))
    use_right_indices = self._is_equal_or_close(
        x, tf.gather(self.outcomes, indices=right_indices))
    left_indices = tf.maximum(0, right_indices - 1)
    use_left_indices = self._is_equal_or_close(
        x, tf.gather(self.outcomes, indices=left_indices))
    log_probs = self._categorical.log_prob(
        tf.where(use_left_indices, left_indices, right_indices))
    return tf.where(
        tf.logical_not(use_left_indices | use_right_indices),
        dtype_util.as_numpy_dtype(log_probs.dtype)(-np.inf),
        log_probs)

  def _mean(self):
    probs = self.probs
    outcomes = self.outcomes
    if dtype_util.is_integer(outcomes.dtype):
      if self._validate_args:
        outcomes = dist_util.embed_check_integer_casting_closed(
            outcomes, target_dtype=probs.dtype)
      outcomes = tf.cast(outcomes, dtype=probs.dtype)
    return tf.tensordot(outcomes, probs, axes=[[0], [-1]])

  def _mode(self):
    return tf.gather(self.outcomes, indices=self._categorical.mode())

  def _sample_n(self, n, seed=None, **distribution_kwargs):
    return tf.gather(
        self.outcomes,
        indices=self._categorical.sample(
            sample_shape=[n], seed=seed, **distribution_kwargs))

  def _variance(self):
    probs = self._categorical.probs_parameter()
    outcomes = tf.broadcast_to(
        self.outcomes, shape=dist_util.prefer_static_shape(probs))
    if dtype_util.is_integer(outcomes.dtype):
      if self._validate_args:
        outcomes = dist_util.embed_check_integer_casting_closed(
            outcomes, target_dtype=probs.dtype)
      outcomes = tf.cast(outcomes, dtype=probs.dtype)
    square_d = tf.math.squared_difference(outcomes,
                                          tf.expand_dims(self.mean(), axis=-1))
    return tf.reduce_sum(probs * square_d, axis=-1)

  def logits_parameter(self, name=None):
    """Logits vec computed from non-`None` input arg (`probs` or `logits`)."""
    with self._name_and_control_scope(name or 'logits_parameter'):
      return self._categorical.logits_parameter()

  def probs_parameter(self, name=None):
    """Probs vec computed from non-`None` input arg (`probs` or `logits`)."""
    with self._name_and_control_scope(name or 'probs_parameter'):
      return self._categorical.probs_parameter()


def _maybe_validate_args(outcomes, logits, probs, validate_args):
  """Validate `outcomes`, `logits` and `probs`'s shapes."""
  assertions = []

  def validate_equal_last_dim(tensor_a, tensor_b, message):
    if tensor_a.shape.is_fully_defined() and tensor_b.shape.is_fully_defined():
      if tensor_a.shape[-1] != tensor_b.shape[-1]:
        raise ValueError(message)
    elif validate_args:
      assertions.append(
          tf1.assert_equal(
              tf.shape(tensor_a)[-1], tf.shape(tensor_b)[-1], message=message))

  if logits is not None:
    validate_equal_last_dim(
        outcomes,
        logits,
        message='Last dimension of outcomes and logits must be equal size.')
  if probs is not None:
    validate_equal_last_dim(
        outcomes,
        probs,
        message='Last dimension of outcomes and probs must be equal size.')

  message = 'Rank of outcomes must be 1.'
  if outcomes.shape.ndims is not None:
    if outcomes.shape.ndims != 1:
      raise ValueError(message)
  elif validate_args:
    assertions.append(tf1.assert_rank(outcomes, 1, message=message))

  message = 'Size of outcomes must be greater than 0.'
  if outcomes.shape.num_elements() is not None:
    if outcomes.shape.num_elements() == 0:
      raise ValueError(message)
  elif validate_args:
    assertions.append(tf1.assert_greater(tf.size(outcomes), 0, message=message))

  if validate_args:
    assertions.append(
        tf1.assert_equal(
            tf.math.is_strictly_increasing(outcomes),
            True,
            message='outcomes is not strictly increasing.'))

  return assertions
