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
"""The Categorical distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util


def _broadcast_cat_event_and_params(event, params, base_dtype):
  """Broadcasts the event or distribution parameters."""
  if dtype_util.is_integer(event.dtype):
    pass
  elif dtype_util.is_floating(event.dtype):
    # When `validate_args=True` we've already ensured int/float casting
    # is closed.
    event = tf.cast(event, dtype=tf.int32)
  else:
    raise TypeError('`value` should have integer `dtype` or '
                    '`self.dtype` ({})'.format(base_dtype))
  shape_known_statically = (
      tensorshape_util.rank(params.shape) is not None and
      tensorshape_util.is_fully_defined(params.shape[:-1]) and
      tensorshape_util.is_fully_defined(event.shape))
  if not shape_known_statically or params.shape[:-1] != event.shape:
    params = params * tf.ones_like(event[..., tf.newaxis],
                                   dtype=params.dtype)
    params_shape = tf.shape(params)[:-1]
    event = event * tf.ones(params_shape, dtype=event.dtype)
    if tensorshape_util.rank(params.shape) is not None:
      tensorshape_util.set_shape(event, params.shape[:-1])

  return event, params


class Categorical(distribution.Distribution):
  """Categorical distribution over integers.

  The Categorical distribution is parameterized by either probabilities or
  log-probabilities of a set of `K` classes. It is defined over the integers
  `{0, 1, ..., K-1}`.

  The Categorical distribution is closely related to the `OneHotCategorical` and
  `Multinomial` distributions.  The Categorical distribution can be intuited as
  generating samples according to `argmax{ OneHotCategorical(probs) }` itself
  being identical to `argmax{ Multinomial(probs, total_count=1) }`.

  #### Mathematical Details

  The probability mass function (pmf) is,

  ```none
  pmf(k; pi) = prod_j pi_j**[k == j]
  ```

  #### Pitfalls

  The number of classes, `K`, must not exceed:

  - the largest integer representable by `self.dtype`, i.e.,
    `2**(mantissa_bits+1)` (IEEE 754),
  - the maximum `Tensor` index, i.e., `2**31-1`.

  In other words,

  ```python
  K <= min(2**31-1, {
    tf.float16: 2**11,
    tf.float32: 2**24,
    tf.float64: 2**53 }[param.dtype])
  ```

  Note: This condition is validated only when `self.validate_args = True`.

  #### Examples

  Creates a 3-class distribution with the 2nd class being most likely.

  ```python
  dist = Categorical(probs=[0.1, 0.5, 0.4])
  n = 1e4
  empirical_prob = tf.cast(
      tf.histogram_fixed_width(
        dist.sample(int(n)),
        [0., 2],
        nbins=3),
      dtype=tf.float32) / n
  # ==> array([ 0.1005,  0.5037,  0.3958], dtype=float32)
  ```

  Creates a 3-class distribution with the 2nd class being most likely.
  Parameterized by [logits](https://en.wikipedia.org/wiki/Logit) rather than
  probabilities.

  ```python
  dist = Categorical(logits=np.log([0.1, 0.5, 0.4])
  n = 1e4
  empirical_prob = tf.cast(
      tf.histogram_fixed_width(
        dist.sample(int(n)),
        [0., 2],
        nbins=3),
      dtype=tf.float32) / n
  # ==> array([0.1045,  0.5047, 0.3908], dtype=float32)
  ```

  Creates a 3-class distribution with the 3rd class being most likely.
  The distribution functions can be evaluated on counts.

  ```python
  # counts is a scalar.
  p = [0.1, 0.4, 0.5]
  dist = Categorical(probs=p)
  dist.prob(0)  # Shape []

  # p will be broadcast to [[0.1, 0.4, 0.5], [0.1, 0.4, 0.5]] to match counts.
  counts = [1, 0]
  dist.prob(counts)  # Shape [2]

  # p will be broadcast to shape [3, 5, 7, 3] to match counts.
  counts = [[...]] # Shape [5, 7, 3]
  dist.prob(counts)  # Shape [5, 7, 3]
  ```

  """

  def __init__(
      self,
      logits=None,
      probs=None,
      dtype=tf.int32,
      validate_args=False,
      allow_nan_stats=True,
      name='Categorical'):
    """Initialize Categorical distributions using class log-probabilities.

    Args:
      logits: An N-D `Tensor`, `N >= 1`, representing the unnormalized
        log probabilities of a set of Categorical distributions. The first
        `N - 1` dimensions index into a batch of independent distributions
        and the last dimension represents a vector of logits for each class.
        Only one of `logits` or `probs` should be passed in.
      probs: An N-D `Tensor`, `N >= 1`, representing the probabilities
        of a set of Categorical distributions. The first `N - 1` dimensions
        index into a batch of independent distributions and the last dimension
        represents a vector of probabilities for each class. Only one of
        `logits` or `probs` should be passed in.
      dtype: The type of the event samples (default: int32).
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
    if (probs is None) == (logits is None):
      raise ValueError('Must pass probs or logits, but not both.')
    with tf.name_scope(name) as name:
      prob_logit_dtype = dtype_util.common_dtype([probs, logits], tf.float32)
      self._probs = tensor_util.convert_nonref_to_tensor(
          probs, dtype_hint=prob_logit_dtype, name='probs')
      self._logits = tensor_util.convert_nonref_to_tensor(
          logits, dtype_hint=prob_logit_dtype, name='logits')
      super(Categorical, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _params_event_ndims(cls):
    return dict(logits=1, probs=1)

  @property
  def logits(self):
    """Input argument `logits`."""
    return self._logits

  @property
  def probs(self):
    """Input argument `probs`."""
    return self._probs

  def _batch_shape_tensor(self, x=None):
    if x is None:
      x = tf.convert_to_tensor(
          self._probs if self._logits is None else self._logits)
    return ps.shape(x)[:-1]

  def _batch_shape(self):
    x = self._probs if self._logits is None else self._logits
    return x.shape[:-1]

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    logits = self._logits_parameter_no_checks()
    logits_2d = tf.reshape(logits, [-1, self._num_categories(logits)])
    sample_dtype = tf.int64 if dtype_util.size(self.dtype) > 4 else tf.int32
    # TODO(b/147874898): Remove workaround for seed-sensitive tests.
    if seed is None or isinstance(seed, six.integer_types):
      draws = tf.random.categorical(
          logits_2d, n, dtype=sample_dtype, seed=seed)
    else:
      draws = samplers.categorical(
          logits_2d, n, dtype=sample_dtype, seed=seed)
    draws = tf.cast(draws, self.dtype)
    return tf.reshape(
        tf.transpose(draws),
        shape=ps.concat([[n], self._batch_shape_tensor(logits)], axis=0))

  def _cdf(self, k):
    # TODO(b/135263541): Improve numerical precision of categorical.cdf.
    probs = self.probs_parameter()
    num_categories = self._num_categories(probs)

    k, probs = _broadcast_cat_event_and_params(
        k, probs, base_dtype=dtype_util.base_dtype(self.dtype))

    # Since the lowest number in the support is 0, any k < 0 should be zero in
    # the output.
    should_be_zero = k < 0

    # Will use k as an index in the gather below, so clip it to {0,...,K-1}.
    k = tf.clip_by_value(tf.cast(k, tf.int32), 0, num_categories - 1)

    batch_shape = tf.shape(k)

    # tf.gather(..., batch_dims=batch_dims) requires static batch_dims kwarg, so
    # to handle the case where the batch shape is dynamic, flatten the batch
    # dims (so we know batch_dims=1).
    k_flat_batch = tf.reshape(k, [-1])
    probs_flat_batch = tf.reshape(
        probs, tf.concat(([-1], [num_categories]), axis=0))

    cdf_flat = tf.gather(
        tf.cumsum(probs_flat_batch, axis=-1),
        k_flat_batch[..., tf.newaxis],
        batch_dims=1)

    cdf = tf.reshape(cdf_flat, shape=batch_shape)

    zero = np.array(0, dtype=dtype_util.as_numpy_dtype(cdf.dtype))
    return tf.where(should_be_zero, zero, cdf)

  def _log_prob(self, k):
    logits = self.logits_parameter()
    if self.validate_args:
      k = distribution_util.embed_check_integer_casting_closed(
          k, target_dtype=self.dtype)
    k, logits = _broadcast_cat_event_and_params(
        k, logits, base_dtype=dtype_util.base_dtype(self.dtype))
    return -tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=k, logits=logits)

  def _entropy(self):
    if self._logits is None:
      # If we only have probs, there's not much we can do to ensure numerical
      # precision.
      probs = tf.convert_to_tensor(self._probs)
      return -tf.reduce_sum(
          tf.math.multiply_no_nan(tf.math.log(probs), probs),
          axis=-1)
    # The following result can be derived as follows. Write log(p[i]) as:
    # s[i]-m-lse(s[i]-m) where m=max(s), then you have:
    #   sum_i exp(s[i]-m-lse(s-m)) (s[i] - m - lse(s-m))
    #   = -m - lse(s-m) + sum_i s[i] exp(s[i]-m-lse(s-m))
    #   = -m - lse(s-m) + (1/exp(lse(s-m))) sum_i s[i] exp(s[i]-m)
    #   = -m - lse(s-m) + (1/sumexp(s-m)) sum_i s[i] exp(s[i]-m)
    # Write x[i]=s[i]-m then you have:
    #   = -m - lse(x) + (1/sum_exp(x)) sum_i s[i] exp(x[i])
    # Negating all of this result is the Shanon (discrete) entropy.
    logits = tf.convert_to_tensor(self._logits)
    m = tf.reduce_max(logits, axis=-1, keepdims=True)
    x = logits - m
    sum_exp_x = tf.reduce_sum(tf.math.exp(x), axis=-1)
    lse_logits = m[..., 0] + tf.math.log(sum_exp_x)
    # TODO(b/161014180): Workaround to support correct gradient calculations
    # with -inf logits.
    is_inf_logits = tf.cast(tf.math.is_inf(logits), dtype=tf.float32)
    is_negative_logits = tf.cast(logits < 0, dtype=tf.float32)
    masked_logits = tf.where(
        tf.cast((is_inf_logits * is_negative_logits), dtype=bool),
        tf.cast(1.0, dtype=logits.dtype), logits)

    return lse_logits - tf.reduce_sum(
        tf.math.multiply_no_nan(masked_logits, tf.math.exp(x)),
        axis=-1) / sum_exp_x

  def _mode(self):
    x = self._probs if self._logits is None else self._logits
    mode = tf.cast(tf.argmax(x, axis=-1), self.dtype)
    tensorshape_util.set_shape(mode, x.shape[:-1])
    return mode

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

  def _num_categories(self, x=None):
    """Scalar `int32` tensor: the number of categories."""
    with tf.name_scope('num_categories'):
      if x is None:
        x = self._probs if self._logits is None else self._logits
      num_categories = tf.compat.dimension_value(x.shape[-1])
      if num_categories is not None:
        return num_categories
      # NOTE: In TF1, tf.shape(x) can call `tf.convert_to_tensor(x)` **twice**,
      # so we pre-emptively convert-to-tensor.
      return tf.shape(tf.convert_to_tensor(x))[-1]

  def _default_event_space_bijector(self):
    return

  def _parameter_control_dependencies(self, is_init):
    return maybe_assert_categorical_param_correctness(
        is_init, self.validate_args, self._probs, self._logits)

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.extend(distribution_util.assert_nonnegative_integer_form(x))
    assertions.append(
        assert_util.assert_less_equal(
            x, tf.cast(self._num_categories(), x.dtype),
            message=('Categorical samples must be between `0` and `n-1` '
                     'where `n` is the number of categories.')))
    return assertions


def maybe_assert_categorical_param_correctness(
    is_init, validate_args, probs, logits):
  """Return assertions for `Categorical`-type distributions."""
  assertions = []

  # In init, we can always build shape and dtype checks because
  # we assume shape doesn't change for Variable backed args.
  if is_init:
    x, name = (probs, 'probs') if logits is None else (logits, 'logits')

    if not dtype_util.is_floating(x.dtype):
      raise TypeError('Argument `{}` must having floating type.'.format(name))

    msg = 'Argument `{}` must have rank at least 1.'.format(name)
    ndims = tensorshape_util.rank(x.shape)
    if ndims is not None:
      if ndims < 1:
        raise ValueError(msg)
    elif validate_args:
      x = tf.convert_to_tensor(x)
      probs = x if logits is None else None  # Retain tensor conversion.
      logits = x if probs is None else None
      assertions.append(assert_util.assert_rank_at_least(x, 1, message=msg))

  if not validate_args:
    assert not assertions  # Should never happen.
    return []

  if logits is not None:
    if is_init != tensor_util.is_ref(logits):
      logits = tf.convert_to_tensor(logits)
      assertions.extend(
          distribution_util.assert_categorical_event_shape(logits))

  if probs is not None:
    if is_init != tensor_util.is_ref(probs):
      probs = tf.convert_to_tensor(probs)
      assertions.extend([
          assert_util.assert_non_negative(probs),
          assert_util.assert_near(
              tf.reduce_sum(probs, axis=-1),
              np.array(1, dtype=dtype_util.as_numpy_dtype(probs.dtype)),
              message='Argument `probs` must sum to 1.')
      ])
      assertions.extend(distribution_util.assert_categorical_event_shape(probs))

  return assertions


@kullback_leibler.RegisterKL(Categorical, Categorical)
def _kl_categorical_categorical(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) with a and b Categorical.

  Args:
    a: instance of a Categorical distribution object.
    b: instance of a Categorical distribution object.
    name: Python `str` name to use for created operations.
      Default value: `None` (i.e., `'kl_categorical_categorical'`).

  Returns:
    Batchwise KL(a || b)
  """
  with tf.name_scope(name or 'kl_categorical_categorical'):
    a_logits = a._logits_parameter_no_checks()  # pylint:disable=protected-access
    b_logits = b._logits_parameter_no_checks()  # pylint:disable=protected-access
    return tf.reduce_sum(
        (tf.math.softmax(a_logits) *
         (tf.math.log_softmax(a_logits) - tf.math.log_softmax(b_logits))),
        axis=-1)
