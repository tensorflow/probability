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

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util

NUMPY = False

__all__ = [
    'Empirical'
]


def _broadcast_event_and_samples(event, samples, event_ndims):
  """Broadcasts the event or samples."""
  # This is the shape of self.samples, without the samples axis, i.e. the shape
  # of the result of a call to dist.sample(). This way we can broadcast it with
  # event to get a properly-sized event, then add the singleton dim back at
  # -event_ndims - 1.
  samples_shape = ps.concat(
      [
          ps.shape(samples)[:-event_ndims - 1],
          ps.shape(samples)[ps.rank(samples) - event_ndims:]
      ],
      axis=0)
  event = event * tf.ones(samples_shape, dtype=event.dtype)
  event = tf.expand_dims(event, axis=-event_ndims - 1)
  samples = samples * tf.ones_like(event, dtype=samples.dtype)

  return event, samples


class Empirical(
    distribution.DiscreteDistributionMixin,
    distribution.AutoCompositeTensorDistribution):
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
      samples: Numeric `Tensor` of shape `[B1, ..., Bk, S, E1, ..., En]`,
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
      ValueError: if the rank of `samples` is statically known and less than
        event_ndims + 1.
    """

    parameters = dict(locals())
    with tf.name_scope(name):
      self._samples = tensor_util.convert_nonref_to_tensor(samples)
      dtype = dtype_util.common_dtype(
          [self._samples], dtype_hint=self._samples.dtype)
      self._event_ndims = event_ndims

      # Note: this tf.rank call affects the graph, but is ok in `__init__`
      # because we don't expect shapes (or ranks) to be runtime-variable, nor
      # ever need to differentiate with respect to them.
      samples_rank = ps.rank(self._samples)
      self._samples_axis = ps.cast(
          samples_rank - self._event_ndims - 1, tf.int32)

      super(Empirical, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        samples=parameter_properties.ParameterProperties(
            event_ndims=lambda self: self._event_ndims + 1))  # pylint: disable=protected-access

  @property
  def samples(self):
    """Distribution parameter."""
    return self._samples

  def compute_num_samples(self):
    """Compute and return the number of values in `self.samples`.

    Returns:
      num_samples: int32 `Tensor` containing the number of entries in
        `self.samples`. If `self.samples` has shape `[..., S, E1, ..., Ee]`
        where the `E`'s are event dims, this method returns a `Tensor` whose
        values is `S`.
    """
    with self._name_and_control_scope('compute_num_samples'):
      return self._compute_num_samples(self.samples)

  def _compute_num_samples(self, samples):
    samples_shape = distribution_util.prefer_static_shape(samples)
    return ps.convert_to_shape_tensor(
        samples_shape[self._samples_axis],
        dtype_hint=tf.int32,
        name='num_samples')

  def _event_shape_tensor(self, samples=None):
    if samples is None:
      samples = tf.convert_to_tensor(self.samples)
    return ps.shape(samples)[self._samples_axis + 1:]

  def _event_shape(self):
    if tensorshape_util.rank(self.samples.shape) is None:
      return tf.TensorShape(None)
    return self.samples.shape[self._samples_axis + 1:]

  def _mean(self, samples=None):
    if samples is None:
      samples = tf.convert_to_tensor(self._samples)
    return tf.reduce_mean(samples, axis=self._samples_axis)

  def _stddev(self):
    samples = tf.convert_to_tensor(self._samples)
    axis = self._samples_axis
    r = samples - tf.expand_dims(self._mean(samples), axis=axis)
    var = tf.reduce_mean(tf.square(r), axis=axis)
    return tf.sqrt(var)

  def _quantile(self, value, samples=None, **kwargs):
    if NUMPY:
      raise NotImplementedError()
    from tensorflow_probability.python import stats  # pylint: disable=g-import-not-at-top
    if samples is None:
      samples = tf.convert_to_tensor(self._samples)

    return stats.percentile(
        x=samples, q=value * 100, axis=self._samples_axis, **kwargs)

  def _sample_n(self, n, seed=None):
    samples = tf.convert_to_tensor(self._samples)
    indices = samplers.uniform([n], maxval=self._compute_num_samples(samples),
                               dtype=tf.int32, seed=seed)
    draws = tf.gather(samples, indices, axis=self._samples_axis)
    axes = ps.concat(
        [[self._samples_axis],
         ps.range(self._samples_axis, dtype=tf.int32),
         ps.range(self._event_ndims, dtype=tf.int32) + self._samples_axis + 1],
        axis=0)
    draws = tf.transpose(a=draws, perm=axes)
    return draws

  def _mode(self, samples=None):
    # Samples count can vary by batch member. Use map_fn to compute mode for
    # each batch separately.
    def _get_mode(samples):
      _, idx, count = tf.raw_ops.UniqueWithCountsV2(x=samples, axis=[0])
      # TODO(b/161402486): Remove this hack for fixing the wrong static shape
      # of `idx` in graph mode.
      idx = tf.vectorized_map(lambda x: tf.reshape(x, [-1])[0], idx)
      # NOTE:
      #  - `count` has shape `[K]`, where `K` is the number of unique elements,
      #    and `count[j]` is the number of times the j-th unique element occurs
      #    in `samples`.
      #  - `idx` has shape `[samples.shape[0]]`, and `idx[i] == j` means that
      #    `samples[i]` is equal to the `j`-th unique element.
      max_count_idx = tf.argmax(count, output_type=tf.int32)
      # Return an index `i` for which `idx[i] == max_count_idx`.
      return tf.argmax(
          tf.cast(tf.math.equal(idx, max_count_idx), dtype=tf.int32),
          output_type=tf.int32)

    if samples is None:
      samples = tf.convert_to_tensor(self._samples)
    num_samples = self._compute_num_samples(samples)

    # Flatten samples for each batch.
    if self._event_ndims == 0:
      flattened_samples = tf.reshape(samples, [-1, num_samples])
      mode_shape = self._batch_shape_tensor(samples=samples)
    else:
      event_size = tf.reduce_prod(self._event_shape_tensor(samples))
      mode_shape = ps.concat(
          [self._batch_shape_tensor(samples=samples),
           self._event_shape_tensor(samples=samples)],
          axis=0)
      flattened_samples = tf.reshape(samples, [-1, num_samples, event_size])

    indices = tf.map_fn(
        _get_mode, flattened_samples,
        fn_output_signature=tf.int32)
    full_indices = tf.stack([tf.range(tf.shape(indices)[0]), indices], axis=1)

    mode = tf.gather_nd(flattened_samples, full_indices)
    return tf.reshape(mode, mode_shape)

  def _entropy(self):
    samples = tf.convert_to_tensor(self.samples)
    num_samples = self._compute_num_samples(samples)
    entropy_shape = self._batch_shape_tensor(samples=samples)

    # Flatten samples for each batch.
    if self._event_ndims == 0:
      samples = tf.reshape(samples, [-1, num_samples])
    else:
      event_size = tf.reduce_prod(self.event_shape_tensor())
      samples = tf.reshape(samples, [-1, num_samples, event_size])

    # Use map_fn to compute entropy for each batch separately.
    def _get_entropy(samples):
      count = tf.raw_ops.UniqueWithCountsV2(x=samples, axis=[0]).count
      prob = tf.cast(count / num_samples, dtype=self.dtype)
      entropy = tf.reduce_sum(-prob * tf.math.log(prob))
      return entropy

    entropy = tf.map_fn(_get_entropy, samples, dtype=self.dtype)
    return tf.reshape(entropy, entropy_shape)

  def _cdf(self, event):
    samples = tf.convert_to_tensor(self._samples)
    num_samples = self._compute_num_samples(samples)
    event = tf.convert_to_tensor(event, name='event', dtype=self.dtype)
    event, samples = _broadcast_event_and_samples(event, samples,
                                                  event_ndims=self._event_ndims)
    cdf = tf.reduce_sum(
        tf.cast(
            tf.reduce_all(
                samples <= event, axis=ps.range(-self._event_ndims, 0)),
            dtype=tf.int32),
        axis=-1) / num_samples
    if dtype_util.is_floating(self.dtype):
      cdf = tf.cast(cdf, self.dtype)
    return cdf

  def _prob(self, event):
    samples = tf.convert_to_tensor(self._samples)
    num_samples = self._compute_num_samples(samples)
    event = tf.convert_to_tensor(event, name='event', dtype=self.dtype)
    event, samples = _broadcast_event_and_samples(event, samples,
                                                  event_ndims=self._event_ndims)
    prob = tf.reduce_sum(
        tf.cast(
            tf.reduce_all(
                tf.equal(samples, event), axis=ps.range(-self._event_ndims, 0)),
            dtype=tf.int32),
        axis=-1) / num_samples
    if dtype_util.is_floating(self.dtype):
      prob = tf.cast(prob, self.dtype)
    return prob

  def _default_event_space_bijector(self):
    return

  def _parameter_control_dependencies(self, is_init):
    assertions = []

    message = 'Rank of `samples` must be at least `event_ndims + 1`.'
    if is_init:
      samples_rank = tensorshape_util.rank(self.samples.shape)
      if samples_rank is not None:
        if samples_rank < self._event_ndims + 1:
          raise ValueError(message)
      elif self._validate_args:
        assertions.append(
            assert_util.assert_rank_at_least(
                self._samples, self._event_ndims + 1, message=message))

    if not self._validate_args:
      assert not assertions  # Should never happen.
      return []

    return assertions
