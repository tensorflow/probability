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
"""The Empirical distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow.python.ops import gen_array_ops  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'Empirical'
]


def _broadcast_event_and_samples(event, samples, event_ndims):
  """Broadcasts the event or samples."""
  # This is the shape of self.samples, without the samples axis, i.e. the shape
  # of the result of a call to dist.sample(). This way we can broadcast it with
  # event to get a properly-sized event, then add the singleton dim back at
  # -event_ndims - 1.
  samples_shape = tf.concat(
      [
          tf.shape(samples)[:-event_ndims - 1],
          tf.shape(samples)[tf.rank(samples) - event_ndims:]
      ],
      axis=0)
  event *= tf.ones(samples_shape, dtype=event.dtype)
  event = tf.expand_dims(event, axis=-event_ndims - 1)
  samples *= tf.ones_like(event, dtype=samples.dtype)

  return event, samples


class Empirical(distribution.Distribution):
  """Empirical distribution.

  The Empirical distribution is parameterized by a [batch] multiset of samples.
  It describes the empirical measure (observations) of a variable.

  Note: some methods (log_prob, prob, cdf, mode, entropy) are not differentiable
  with regard to samples.

  #### Mathematical Details

  The probability mass function (pmf) and cumulative distribution function (cdf)
  are

  ```none
  pmf(k; s1, ..., sn) = sum_i I(k)^{k == si} / n
  I(k)^{k == si} == 1, if k == si, else 0.
  cdf(k; s1, ..., sn) = sum_i I(k)^{k >= si} / n
  I(k)^{k >= si} == 1, if k >= si, else 0.
  ```

  #### Examples

  ```python

  # Initialize a empirical distribution with 4 scalar samples.
  dist = Empirical(samples=[0., 1., 1., 2.])
  dist.cdf(1.)
  ==> 0.75
  dist.prob([0., 1.])
  ==> [0.25, 0.5] # samples will be broadcast to
                    [[0., 1., 1., 2.], [0., 1., 1., 2.]] to match event.

  # Initialize a empirical distribution with a [2] batch of scalar samples.
  dist = Empirical(samples=[[0., 1.], [1., 2.]])
  dist.cdf([0., 2.])
  ==> [0.5, 1.]
  dist.prob(0.)
  ==> [0.5, 0] # event will be broadcast to [0., 0.] to match samples.

  # Initialize a empirical distribution with 4 vector-like samples.
  dist = Empirical(samples=[[0., 0.], [0., 1.], [0., 1.], [1., 2.]],
                   event_ndims=1)
  dist.cdf([0., 1.])
  ==> 0.75
  dist.prob([[0., 1.], [1., 2.]])
  ==> [0.5, 0.25] # samples will be broadcast to shape [2, 4, 2] to match event.

  # Initialize a empirical distribution with a [2] batch of vector samples.
  dist = Empirical(samples=[[[0., 0.], [0., 1.]], [[0., 1.], [1., 2.]]],
                   event_ndims=1)
  dist.cdf([[0., 0.], [0., 1.]])
  ==> [0.5, 0.5]
  dist.prob([0., 1.])
  ==> [0.5, 1.] # event will be broadcast to shape [[0., 1.], [0., 1.]]
                  to match samples.
  ```

  """

  def __init__(self,
               samples,
               event_ndims=0,
               validate_args=False,
               allow_nan_stats=True,
               name='Empirical'):
    """Initialize `Empirical` distributions.

    Args:
      samples: Numeric `Tensor` of shape [B1, ..., Bk, S, E1, ..., En]`,
        `k, n >= 0`. Samples or batches of samples on which the distribution
        is based. The first `k` dimensions index into a batch of independent
        distributions. Length of `S` dimension determines number of samples
        in each multiset. The last `n` dimension represents samples for each
        distribution. n is specified by argument event_ndims.
      event_ndims: Python `int32`, default `0`. number of dimensions for each
        event. When `0` this distribution has scalar samples. When `1` this
        distribution has vector-like samples.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value `NaN` to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      ValueError: if the rank of `samples` < event_ndims + 1.
    """

    parameters = dict(locals())
    with tf.name_scope(name):
      self._samples = tf.convert_to_tensor(samples, name='samples')
      self._event_ndims = event_ndims
      self._samples_axis = (
          (tensorshape_util.rank(self.samples.shape) or tf.rank(self.samples)) -
          self._event_ndims - 1)
      with tf.control_dependencies(
          [assert_util.assert_rank_at_least(self._samples, event_ndims + 1)]):
        samples_shape = distribution_util.prefer_static_shape(self._samples)
        self._num_samples = samples_shape[self._samples_axis]

    super(Empirical, self).__init__(
        dtype=self._samples.dtype,
        reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._samples],
        name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return {'samples': tf.convert_to_tensor(sample_shape, dtype=tf.int32)}

  def _params_event_ndims(self):
    return dict(samples=self._event_ndims + 1)

  @property
  def samples(self):
    """Distribution parameter."""
    return self._samples

  @property
  def num_samples(self):
    """Number of samples."""
    return self._num_samples

  def _batch_shape_tensor(self):
    return tf.shape(self.samples)[:self._samples_axis]

  def _batch_shape(self):
    if tensorshape_util.rank(self.samples.shape) is None:
      return tf.TensorShape(None)
    return self.samples.shape[:self._samples_axis]

  def _event_shape_tensor(self):
    return tf.shape(self.samples)[self._samples_axis + 1:]

  def _event_shape(self):
    if tensorshape_util.rank(self.samples.shape) is None:
      return tf.TensorShape(None)
    return self.samples.shape[self._samples_axis + 1:]

  def _mean(self):
    return tf.reduce_mean(self.samples, axis=self._samples_axis)

  def _stddev(self):
    r = self.samples - tf.expand_dims(self.mean(), axis=self._samples_axis)
    var = tf.reduce_mean(tf.square(r), axis=self._samples_axis)
    return tf.sqrt(var)

  def _sample_n(self, n, seed=None):
    indices = tf.random.uniform([n], maxval=self.num_samples,
                                dtype=tf.int32, seed=seed)
    draws = tf.gather(self.samples, indices, axis=self._samples_axis)
    axes = tf.concat(
        [[self._samples_axis],
         tf.range(self._samples_axis),
         tf.range(self._event_ndims) + self._samples_axis + 1],
        axis=0)
    draws = tf.transpose(a=draws, perm=axes)
    return draws

  def _mode(self):
    # Samples count can vary by batch member. Use map_fn to compute mode for
    # each batch separately.
    def _get_mode(samples):
      # TODO(b/123985779): Swith to tf.unique_with_counts_v2 when exposed
      count = gen_array_ops.unique_with_counts_v2(samples, axis=[0]).count
      return tf.argmax(count)

    # Flatten samples for each batch.
    if self._event_ndims == 0:
      samples = tf.reshape(self.samples, [-1, self.num_samples])
      mode_shape = self.batch_shape_tensor()
    else:
      event_size = tf.reduce_prod(self.event_shape_tensor())
      samples = tf.reshape(self.samples, [-1, self.num_samples, event_size])
      mode_shape = tf.concat(
          [self.batch_shape_tensor(), self.event_shape_tensor()],
          axis=0)

    indices = tf.map_fn(_get_mode, samples, dtype=tf.int64)
    full_indices = tf.stack(
        [tf.range(tf.shape(indices)[0]),
         tf.cast(indices, tf.int32)], axis=1)

    mode = tf.gather_nd(samples, full_indices)
    return tf.reshape(mode, mode_shape)

  def _entropy(self):
    # Use map_fn to compute entropy for each batch separately.
    def _get_entropy(samples):
      # TODO(b/123985779): Swith to tf.unique_with_counts_v2 when exposed
      count = gen_array_ops.unique_with_counts_v2(samples, axis=[0]).count
      prob = count / self.num_samples
      entropy = tf.reduce_sum(-prob * tf.math.log(prob))
      return entropy

    # Flatten samples for each batch.
    if self._event_ndims == 0:
      samples = tf.reshape(self.samples, [-1, self.num_samples])
    else:
      event_size = tf.reduce_prod(self.event_shape_tensor())
      samples = tf.reshape(self.samples, [-1, self.num_samples, event_size])

    entropy = tf.map_fn(_get_entropy, samples)
    entropy_shape = self.batch_shape_tensor()
    if dtype_util.is_floating(self.dtype):
      entropy = tf.cast(entropy, self.dtype)
    return tf.reshape(entropy, entropy_shape)

  def _cdf(self, event):
    event = tf.convert_to_tensor(event, name='event', dtype=self.dtype)
    event, samples = _broadcast_event_and_samples(event, self.samples,
                                                  event_ndims=self._event_ndims)
    cdf = tf.reduce_sum(
        tf.cast(
            tf.reduce_all(
                samples <= event, axis=tf.range(-self._event_ndims, 0)),
            dtype=tf.int32),
        axis=-1) / self.num_samples
    if dtype_util.is_floating(self.dtype):
      cdf = tf.cast(cdf, self.dtype)
    return cdf

  def _prob(self, event):
    event = tf.convert_to_tensor(event, name='event', dtype=self.dtype)
    event, samples = _broadcast_event_and_samples(event, self.samples,
                                                  event_ndims=self._event_ndims)
    prob = tf.reduce_sum(
        tf.cast(
            tf.reduce_all(
                tf.equal(samples, event), axis=tf.range(-self._event_ndims, 0)),
            dtype=tf.int32),
        axis=-1) / self.num_samples
    if dtype_util.is_floating(self.dtype):
      prob = tf.cast(prob, self.dtype)
    return prob
