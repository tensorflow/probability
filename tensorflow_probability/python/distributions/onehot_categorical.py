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
"""The OneHotCategorical distribution class."""

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import softmax_centered as softmax_centered_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util


class OneHotCategorical(distribution.AutoCompositeTensorDistribution):
  """OneHotCategorical distribution.

  The categorical distribution is parameterized by the log-probabilities
  of a set of classes. The difference between OneHotCategorical and Categorical
  distributions is that OneHotCategorical is a discrete distribution over
  one-hot bit vectors whereas Categorical is a discrete distribution over
  positive integers. OneHotCategorical is equivalent to Categorical except
  Categorical has event_dim=() while OneHotCategorical has event_dim=K, where
  K is the number of classes.

  This class provides methods to create indexed batches of OneHotCategorical
  distributions. If the provided `logits` or `probs` is rank 2 or higher, for
  every fixed set of leading dimensions, the last dimension represents one
  single OneHotCategorical distribution. When calling distribution
  functions (e.g. `dist.prob(x)`), `logits` and `x` are broadcast to the
  same shape (if possible). In all cases, the last dimension of `logits,x`
  represents single OneHotCategorical distributions.

  #### Examples

  Creates a 3-class distribution, with the 2nd class, the most likely to be
  drawn from.

  ```python
  p = [0.1, 0.5, 0.4]
  dist = OneHotCategorical(probs=p)
  ```

  Creates a 3-class distribution, with the 2nd class the most likely to be
  drawn from, using logits.

  ```python
  logits = [-2, 2, 0]
  dist = OneHotCategorical(logits=logits)
  ```

  Creates a 3-class distribution, with the 3rd class is most likely to be drawn.

  ```python
  # counts is a scalar.
  p = [0.1, 0.4, 0.5]
  dist = OneHotCategorical(probs=p)
  dist.prob([0,1,0])  # Shape []

  # p will be broadcast to [[0.1, 0.4, 0.5], [0.1, 0.4, 0.5]] to match.
  samples = [[0,1,0], [1,0,0]]
  dist.prob(samples)  # Shape [2]
  ```

  """

  def __init__(self,
               logits=None,
               probs=None,
               dtype=tf.int32,
               validate_args=False,
               allow_nan_stats=True,
               name='OneHotCategorical'):
    """Initialize OneHotCategorical distributions using class log-probabilities.

    Args:
      logits: An N-D `Tensor`, `N >= 1`, representing the log probabilities of a
        set of Categorical distributions. The first `N - 1` dimensions index
        into a batch of independent distributions and the last dimension
        represents a vector of logits for each class. Only one of `logits` or
        `probs` should be passed in.
      probs: An N-D `Tensor`, `N >= 1`, representing the probabilities of a set
        of Categorical distributions. The first `N - 1` dimensions index into a
        batch of independent distributions and the last dimension represents a
        vector of probabilities for each class. Only one of `logits` or `probs`
        should be passed in.
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
    with tf.name_scope(name) as name:
      self._probs = tensor_util.convert_nonref_to_tensor(
          probs, dtype_hint=tf.float32, name='probs')
      self._logits = tensor_util.convert_nonref_to_tensor(
          logits, dtype_hint=tf.float32, name='logits')
      if (self._probs is None) == (self._logits is None):
        raise ValueError('Must pass `probs` or `logits`, but not both.')
      super(OneHotCategorical, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        logits=parameter_properties.ParameterProperties(event_ndims=1),
        probs=parameter_properties.ParameterProperties(
            event_ndims=1,
            default_constraining_bijector_fn=softmax_centered_bijector
            .SoftmaxCentered,
            is_preferred=False))

  def _event_size(self, param=None):
    if param is None:
      param = self._logits if self._logits is not None else self._probs
    if param.shape is not None:
      event_size = tf.compat.dimension_value(param.shape[-1])
      if event_size is not None:
        return event_size
    return tf.shape(param)[-1]

  @property
  def logits(self):
    """Input argument `logits`."""
    return self._logits

  @property
  def probs(self):
    """Input argument `probs`."""
    return self._probs

  def _event_shape_tensor(self):
    param = self._logits if self._logits is not None else self._probs
    # NOTE: If the last dimension of `param.shape` is statically-known, but
    # the `param.shape` is not statically-known, then we will *not* return a
    # statically-known event size here.  This could be fixed.
    return ps.shape(param)[-1:]

  def _event_shape(self):
    param = self._logits if self._logits is not None else self._probs
    return tensorshape_util.with_rank(param.shape[-1:], rank=1)

  def _sample_n(self, n, seed=None):
    logits = self._logits_parameter_no_checks()
    sample_shape = ps.concat([[n], ps.shape(logits)], 0)
    event_size = self._event_size(logits)
    if tensorshape_util.rank(logits.shape) == 2:
      logits_2d = logits
    else:
      logits_2d = tf.reshape(logits, [-1, event_size])
    samples = samplers.categorical(logits_2d, n, seed=seed)
    samples = tf.transpose(a=samples)
    samples = tf.one_hot(samples, event_size, dtype=self.dtype)
    ret = tf.reshape(samples, sample_shape)
    return ret

  def _log_prob(self, x):
    logits = self._logits_parameter_no_checks()
    event_size = self._event_size(logits)

    x = tf.cast(x, logits.dtype)

    # broadcast logits or x if need be.
    if (not tensorshape_util.is_fully_defined(x.shape) or
        not tensorshape_util.is_fully_defined(logits.shape) or
        x.shape != logits.shape):
      broadcast_shape = ps.broadcast_shape(ps.shape(logits), ps.shape(x))
      logits = tf.broadcast_to(logits, broadcast_shape)
      x = tf.broadcast_to(x, broadcast_shape)

    logits_shape = ps.shape(tf.reduce_sum(logits, axis=-1))
    logits_2d = tf.reshape(logits, [-1, event_size])
    x_2d = tf.reshape(x, [-1, event_size])
    ret = -tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.stop_gradient(x_2d),
        logits=logits_2d)

    # Reshape back to user-supplied batch and sample dims prior to 2D reshape.
    ret = tf.reshape(ret, logits_shape)
    return ret

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
    lse_logits = m[..., 0] + tf.reduce_logsumexp(x, axis=-1)
    sum_exp_x = tf.reduce_sum(tf.math.exp(x), axis=-1)
    return lse_logits - tf.reduce_sum(
        tf.math.multiply_no_nan(logits, tf.math.exp(x)), axis=-1) / sum_exp_x

  def _mean(self):
    return self._probs_parameter_no_checks()

  def _mode(self):
    logits = self._logits_parameter_no_checks()
    ret = tf.one_hot(
        tf.argmax(logits, axis=-1), self._event_size(logits), dtype=self.dtype)
    tensorshape_util.set_shape(ret, logits.shape)
    return ret

  def _covariance(self):
    p = self._probs_parameter_no_checks()
    ret = -tf.matmul(p[..., None], p[..., None, :])
    return tf.linalg.set_diag(ret, self._variance(p))

  def _variance(self, probs=None):
    if probs is None:
      probs = self._probs_parameter_no_checks()
    return probs * (1. - probs)

  def logits_parameter(self, name=None):
    """Logits vec computed from non-`None` input arg (`probs` or `logits`)."""
    with self._name_and_control_scope(name or 'logits_parameter'):
      return self._logits_parameter_no_checks()

  def _logits_parameter_no_checks(self):
    if self._logits is None:
      return tf.math.log(self._probs)
    return tensor_util.identity_as_tensor(self._logits)

  def probs_parameter(self, name=None):
    """Probs vec computed from non-`None` input arg (`probs` or `logits`)."""
    with self._name_and_control_scope(name or 'probs_parameter'):
      return self._probs_parameter_no_checks()

  def _probs_parameter_no_checks(self):
    if self._logits is None:
      return tensor_util.identity_as_tensor(self._probs)
    return tf.math.softmax(self._logits)

  def _default_event_space_bijector(self):
    return

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.extend(distribution_util.assert_nonnegative_integer_form(x))
    assertions.append(assert_util.assert_equal(
        tf.ones([], dtype=x.dtype), tf.reduce_sum(x, axis=[-1]),
        message='Last dimension of sample must sum to 1.'))
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
        with tf.control_dependencies(assertions):
          param = tf.identity(param)

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
            tf.shape(param)[-1], 1, message=msg1))
        # NOTE: For now, we leave out a runtime assertion that
        # `tf.shape(param)[-1] <= tf.int32.max`.  An earlier `tf.shape` call
        # will fail before we get to this point.

    if not self.validate_args:
      assert not assertions  # Should never happen.
      return []

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


@kullback_leibler.RegisterKL(OneHotCategorical, OneHotCategorical)
def _kl_categorical_categorical(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) with a, b OneHotCategorical.

  Args:
    a: instance of a OneHotCategorical distribution object.
    b: instance of a OneHotCategorical distribution object.
    name: Python `str` name to use for created operations.
      Default value: `None` (i.e., `'kl_categorical_categorical'`).

  Returns:
    Batchwise KL(a || b)
  """
  with tf.name_scope(name or 'kl_categorical_categorical'):
    # pylint: disable=protected-access
    a_logits = a._logits_parameter_no_checks()
    # pylint: disable=protected-access
    b_logits = b._logits_parameter_no_checks()
    # sum(p ln(p / q))
    return tf.reduce_sum(
        (tf.math.softmax(a_logits) *
         (tf.math.log_softmax(a_logits) - tf.math.log_softmax(b_logits))),
        axis=-1)
