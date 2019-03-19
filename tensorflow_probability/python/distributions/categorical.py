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

import tensorflow as tf

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import distribution_util as util
from tensorflow_probability.python.internal import reparameterization

from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import


def _broadcast_cat_event_and_params(event, params, base_dtype):
  """Broadcasts the event or distribution parameters."""
  if event.dtype.is_integer:
    pass
  elif event.dtype.is_floating:
    # When `validate_args=True` we've already ensured int/float casting
    # is closed.
    event = tf.cast(event, dtype=tf.int32)
  else:
    raise TypeError("`value` should have integer `dtype` or "
                    "`self.dtype` ({})".format(base_dtype))
  shape_known_statically = (
      params.shape.ndims is not None and
      params.shape[:-1].is_fully_defined() and
      event.shape.is_fully_defined())
  if not shape_known_statically or params.shape[:-1] != event.shape:
    params *= tf.ones_like(event[..., tf.newaxis],
                           dtype=params.dtype)
    params_shape = tf.shape(input=params)[:-1]
    event *= tf.ones(params_shape, dtype=event.dtype)
    if params.shape.ndims is not None:
      event.set_shape(tf.TensorShape(params.shape[:-1]))

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
      name="Categorical"):
    """Initialize Categorical distributions using class log-probabilities.

    Args:
      logits: An N-D `Tensor`, `N >= 1`, representing the log probabilities
        of a set of Categorical distributions. The first `N - 1` dimensions
        index into a batch of independent distributions and the last dimension
        represents a vector of logits for each class. Only one of `logits` or
        `probs` should be passed in.
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
    with tf.compat.v1.name_scope(name, values=[logits, probs]) as name:
      self._logits, self._probs = util.get_logits_and_probs(
          logits=logits,
          probs=probs,
          validate_args=validate_args,
          multidimensional=True,
          name=name)

      if validate_args:
        self._logits = util.embed_check_categorical_event_shape(
            self._logits)

      logits_shape_static = self._logits.shape.with_rank_at_least(1)
      if logits_shape_static.ndims is not None:
        self._batch_rank = tf.convert_to_tensor(
            value=logits_shape_static.ndims - 1,
            dtype=tf.int32,
            name="batch_rank")
      else:
        with tf.compat.v1.name_scope(name="batch_rank"):
          self._batch_rank = tf.rank(self._logits) - 1

      logits_shape = tf.shape(input=self._logits, name="logits_shape")
      num_categories = tf.compat.dimension_value(logits_shape_static[-1])
      if num_categories is not None:
        self._num_categories = tf.convert_to_tensor(
            value=num_categories, dtype=tf.int32, name="num_categories")
      else:
        with tf.compat.v1.name_scope(name="num_categories"):
          self._num_categories = logits_shape[self._batch_rank]

      if logits_shape_static[:-1].is_fully_defined():
        self._batch_shape_val = tf.constant(
            logits_shape_static[:-1].as_list(),
            dtype=tf.int32,
            name="batch_shape")
      else:
        with tf.compat.v1.name_scope(name="batch_shape"):
          self._batch_shape_val = logits_shape[:-1]
    super(Categorical, self).__init__(
        dtype=dtype,
        reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._logits,
                       self._probs],
        name=name)

  @classmethod
  def _params_event_ndims(cls):
    return dict(logits=1, probs=1)

  @property
  @deprecation.deprecated(
      "2019-05-19", "The `event_size` property is deprecated.  Use "
      "`num_categories` instead.  They have the same value, but `event_size` "
      "is misnamed.")
  def event_size(self):
    """Scalar `int32` tensor: the number of categories."""
    return self._num_categories

  @property
  def num_categories(self):
    """Scalar `int32` tensor: the number of categories."""
    return self._num_categories

  @property
  def logits(self):
    """Vector of coordinatewise logits."""
    return self._logits

  @property
  def probs(self):
    """Vector of coordinatewise probabilities."""
    return self._probs

  def _batch_shape_tensor(self):
    return tf.identity(self._batch_shape_val)

  def _batch_shape(self):
    return self.logits.shape[:-1]

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    if self.logits.shape.ndims == 2:
      logits_2d = self.logits
    else:
      logits_2d = tf.reshape(self.logits, [-1, self.num_categories])
    sample_dtype = tf.int64 if self.dtype.size > 4 else tf.int32
    draws = tf.random.categorical(
        logits_2d, n, dtype=sample_dtype, seed=seed)
    draws = tf.reshape(
        tf.transpose(a=draws), tf.concat([[n], self.batch_shape_tensor()], 0))
    return tf.cast(draws, self.dtype)

  def _cdf(self, k):
    k = tf.convert_to_tensor(value=k, name="k")

    k, probs = _broadcast_cat_event_and_params(
        k, self.probs, base_dtype=self.dtype.base_dtype)

    # Since the lowest number in the support is 0, any k < 0 should be zero in
    # the output.
    should_be_zero = k < 0

    # Will use k as an index in the gather below, so clip it to {0,...,K-1}.
    k = tf.clip_by_value(tf.cast(k, tf.int32), 0, self.num_categories - 1)

    batch_shape = tf.shape(input=k)

    # tf.gather(..., batch_dims=batch_dims) requires static batch_dims kwarg, so
    # to handle the case where the batch shape is dynamic, flatten the batch
    # dims (so we know batch_dims=1).
    k_flat_batch = tf.reshape(k, [-1])
    probs_flat_batch = tf.reshape(
        probs, tf.concat(([-1], [self.num_categories]), axis=0))

    cdf_flat = tf.gather(
        tf.cumsum(probs_flat_batch, axis=-1),
        k_flat_batch[..., tf.newaxis],
        batch_dims=1)

    cdf = tf.reshape(cdf_flat, shape=batch_shape)

    return tf.where(should_be_zero, tf.zeros_like(cdf), cdf)

  def _log_prob(self, k):
    k = tf.convert_to_tensor(value=k, name="k")
    if self.validate_args:
      k = util.embed_check_integer_casting_closed(
          k, target_dtype=tf.int32)
    k, logits = _broadcast_cat_event_and_params(
        k, self.logits, base_dtype=self.dtype.base_dtype)

    return -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=k,
                                                           logits=logits)

  def _entropy(self):
    return -tf.reduce_sum(
        input_tensor=tf.math.multiply_no_nan(
            tf.nn.log_softmax(self.logits), self.probs), axis=-1)

  def _mode(self):
    ret = tf.argmax(input=self.logits, axis=self._batch_rank)
    ret = tf.cast(ret, self.dtype)
    ret.set_shape(self.batch_shape)
    return ret


# TODO(b/117098119): Remove tf.distribution references once they're gone.
@kullback_leibler.RegisterKL(Categorical,
                             tf.compat.v1.distributions.Categorical)
@kullback_leibler.RegisterKL(tf.compat.v1.distributions.Categorical,
                             Categorical)
@kullback_leibler.RegisterKL(Categorical, Categorical)
def _kl_categorical_categorical(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) with a and b Categorical.

  Args:
    a: instance of a Categorical distribution object.
    b: instance of a Categorical distribution object.
    name: (optional) Name to use for created operations.
      default is "kl_categorical_categorical".

  Returns:
    Batchwise KL(a || b)
  """
  with tf.compat.v1.name_scope(
      name, "kl_categorical_categorical", values=[a.logits, b.logits]):
    # sum(probs log(probs / (1 - probs)))
    delta_log_probs1 = (tf.nn.log_softmax(a.logits) -
                        tf.nn.log_softmax(b.logits))
    return tf.reduce_sum(
        input_tensor=tf.nn.softmax(a.logits) * delta_log_probs1, axis=-1)
