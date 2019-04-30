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
"""The Sample distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import tensorshape_util


def _make_summary_statistic(attr):
  """Factory for implementing summary statistics, eg, mean, stddev, mode."""
  def _fn(self, **kwargs):
    """Implements summary statistic, eg, mean, stddev, mode."""
    x = getattr(self.distribution, attr)(**kwargs)
    shape = prefer_static.concat([
        self.distribution.batch_shape_tensor(),
        prefer_static.ones(prefer_static.rank_from_shape(self.sample_shape),
                           dtype=self.sample_shape.dtype),
        self.distribution.event_shape_tensor(),
    ], axis=0)
    x = tf.reshape(x, shape=shape)
    shape = prefer_static.concat([
        self.distribution.batch_shape_tensor(),
        self.sample_shape,
        self.distribution.event_shape_tensor(),
    ], axis=0)
    return tf.broadcast_to(x, shape)
  return _fn


class Sample(distribution_lib.Distribution):
  """Sample distribution via independent draws.

  This distribution is useful for reducing over a collection of independent,
  identical draws. It is otherwise identical to the input distribution.

  #### Mathematical Details

  The probability function is,

  ```none
  p(x) = prod{ p(x[i]) : i = 0, ..., (n - 1) }
  ```

  #### Examples

  ```python
  tfd = tfp.distributions

  # Example 1: Five scalar draws.

  s = tfd.Sample(
      tfd.Normal(loc=0, scale=1),
      sample_shape=5)
  x = s.sample()
  # ==> x.shape: [5]

  lp = s.log_prob(x)
  # ==> lp.shape: []
  #     Equivalently: tf.reduce_sum(s.distribution.log_prob(x), axis=[0, 1])
  #
  # `Sample.log_prob` computes the per-{sample, batch} `log_prob`s then sums
  # over the `Sample.sample_shape` dimensions. In the above example `log_prob`
  # dims `[0, 1]` are summed out. Conceptually, first dim `1` is summed (this
  # being the intrinsic `event`) then we sum over `Sample.sample_shape` dims, in
  # this case dim `0`.

  # Example 2: `[5, 4]`-draws of a bivariate Normal.

  s = tfd.Sample(
      tfd.Independent(tfd.Normal(loc=tf.zeros([3, 2]), scale=1),
                      reinterpreted_batch_ndims=1),
      sample_shape=[5, 4])
  x = s.sample([6, 1])
  # ==> x.shape: [6, 1, 5, 4, 3, 2]

  lp = s.log_prob(x)
  # ==> lp.shape: [6, 1, 3]
  #     Equivalently: tf.reduce_sum(s.distribution.log_prob(x), axis=[2, 3])
  #
  # `s.log_prob` will reduce over (intrinsic) event dims, i.e., dim `5`, then
  # sums over `s.sample_shape` dims `[2, 3]` corresponding to shape (slice)
  # `[5, 4]`.
  ```

  """

  def __init__(
      self,
      distribution,
      sample_shape=(),
      validate_args=False,
      name=None):
    """Construct the `Sample` distribution.

    Args:
      distribution: The base distribution instance to transform. Typically an
        instance of `Distribution`.
      sample_shape: `int` scalar or vector `Tensor` representing the shape of a
        single sample.
      validate_args: Python `bool`.  Whether to validate input with asserts.
        If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
      name: The name for ops managed by the distribution.
        Default value: `None` (i.e., `'Sample' + distribution.name`).
    """
    parameters = dict(locals())
    name = name or 'Sample' + distribution.name
    self._distribution = distribution
    with tf.name_scope(name) as name:
      sample_shape = distribution_util.expand_to_vector(tf.convert_to_tensor(
          value=sample_shape, dtype_hint=tf.int32, name='sample_shape'))
      self._sample_shape = sample_shape
      super(Sample, self).__init__(
          dtype=self._distribution.dtype,
          reparameterization_type=self._distribution.reparameterization_type,
          validate_args=validate_args,
          allow_nan_stats=self._distribution.allow_nan_stats,
          parameters=parameters,
          graph_parents=([sample_shape] + distribution._graph_parents),  # pylint: disable=protected-access
          name=name)

  @property
  def distribution(self):
    return self._distribution

  @property
  def sample_shape(self):
    return self._sample_shape

  def _batch_shape_tensor(self):
    return self.distribution.batch_shape_tensor()

  def _batch_shape(self):
    return self.distribution.batch_shape

  def _event_shape_tensor(self):
    return prefer_static.concat([
        self.sample_shape,
        self.distribution.event_shape_tensor(),
    ], axis=0)

  def _event_shape(self):
    sample_shape = tf.TensorShape(tf.get_static_value(self.sample_shape))
    if (tensorshape_util.rank(sample_shape) is None or
        tensorshape_util.rank(self.distribution.event_shape) is None):
      return tf.TensorShape(None)
    return tensorshape_util.concatenate(sample_shape,
                                        self.distribution.event_shape)

  def _sample_n(self, n, seed, **kwargs):
    fake_sample_ndims = prefer_static.rank_from_shape(self.sample_shape)
    event_ndims = prefer_static.rank_from_shape(
        self.distribution.event_shape_tensor, self.distribution.event_shape)
    batch_ndims = prefer_static.rank_from_shape(
        self.distribution.batch_shape_tensor, self.distribution.batch_shape)
    perm = prefer_static.concat([
        [0],
        prefer_static.range(1 + fake_sample_ndims,
                            1 + fake_sample_ndims + batch_ndims),
        prefer_static.range(1, 1 + fake_sample_ndims),
        prefer_static.range(1 + fake_sample_ndims + batch_ndims,
                            1 + fake_sample_ndims + batch_ndims + event_ndims),
    ], axis=0)
    x = self.distribution.sample(
        prefer_static.concat([[n], self.sample_shape], axis=0),
        seed=seed,
        **kwargs)
    return tf.transpose(a=x, perm=perm)

  def _log_prob(self, x, **kwargs):
    batch_ndims = prefer_static.rank_from_shape(
        self.distribution.batch_shape_tensor,
        self.distribution.batch_shape)
    extra_sample_ndims = prefer_static.rank_from_shape(self.sample_shape)
    event_ndims = prefer_static.rank_from_shape(
        self.distribution.event_shape_tensor,
        self.distribution.event_shape)
    ndims = prefer_static.rank(x)
    # (1) Expand x's dims.
    d = ndims - batch_ndims - extra_sample_ndims - event_ndims
    x = tf.reshape(x, shape=tf.pad(
        tensor=tf.shape(input=x),
        paddings=[[prefer_static.maximum(0, -d), 0]],
        constant_values=1))
    sample_ndims = prefer_static.maximum(0, d)
    # (2) Transpose x's dims.
    sample_dims = prefer_static.range(0, sample_ndims)
    batch_dims = prefer_static.range(sample_ndims, sample_ndims + batch_ndims)
    extra_sample_dims = prefer_static.range(
        sample_ndims + batch_ndims,
        sample_ndims + batch_ndims + extra_sample_ndims)
    event_dims = prefer_static.range(
        sample_ndims + batch_ndims + extra_sample_ndims,
        ndims)
    perm = prefer_static.concat(
        [sample_dims, extra_sample_dims, batch_dims, event_dims], axis=0)
    x = tf.transpose(a=x, perm=perm)
    # (3) Compute x's log_prob.
    lp = self.distribution.log_prob(x, **kwargs)
    # (4) Make the final reduction in x.
    axis = prefer_static.range(sample_ndims, sample_ndims + extra_sample_ndims)
    return tf.reduce_sum(input_tensor=lp, axis=axis)

  def _entropy(self, **kwargs):
    h = self.distribution.entropy(**kwargs)
    n = prefer_static.reduce_prod(self.sample_shape)
    return tf.cast(x=n, dtype=h.dtype) * h

  _mean = _make_summary_statistic('mean')
  _stddev = _make_summary_statistic('stddev')
  _variance = _make_summary_statistic('variance')
  _mode = _make_summary_statistic('mode')


@kullback_leibler.RegisterKL(Sample, Sample)
def _kl_sample(a, b, name='kl_sample'):
  """Batched KL divergence `KL(a || b)` for Sample distributions.

  We can leverage the fact that:

  ```
  KL(Sample(a) || Sample(b)) = sum(KL(a || b))
  ```

  where the sum is over the `sample_shape` dims.

  Args:
    a: Instance of `Sample` distribution.
    b: Instance of `Sample` distribution.
    name: (optional) name to use for created ops.
      Default value: `"kl_sample"`'.

  Returns:
    kldiv: Batchwise `KL(a || b)`.

  Raises:
    ValueError: If the `sample_shape` of `a` and `b` don't match.
  """
  assertions = []
  a_ss = tf.get_static_value(a.sample_shape)
  b_ss = tf.get_static_value(b.sample_shape)
  msg = '`a.sample_shape` must be identical to `b.sample_shape`.'
  if a_ss is not None and b_ss is not None:
    if not np.array_equal(a_ss, b_ss):
      raise ValueError(msg)
  elif a.validate_args or b.validate_args:
    assertions.append(assert_util.assert_equal(
        a.sample_shape, b.sample_shape, message=msg))
  with tf.control_dependencies(assertions):
    kl = kullback_leibler.kl_divergence(
        a.distribution, b.distribution, name=name)
    n = prefer_static.reduce_prod(a.sample_shape)
    return tf.cast(x=n, dtype=kl.dtype) * kl
