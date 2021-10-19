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

import functools

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import bijector as bijector_lib
from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import log_prob_ratio
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util


def _make_summary_statistic(attr):
  """Factory for implementing summary statistics, eg, mean, stddev, mode."""
  def _fn(self, **kwargs):
    """Implements summary statistic, eg, mean, stddev, mode."""
    sample_shape = ps.reshape(self.sample_shape, shape=[-1])
    x = getattr(self.distribution, attr)(**kwargs)
    shape = ps.concat([
        self.distribution.batch_shape_tensor(),
        ps.ones(ps.rank_from_shape(sample_shape), dtype=sample_shape.dtype),
        self.distribution.event_shape_tensor(),
    ], axis=0)
    x = tf.reshape(x, shape=shape)
    shape = ps.concat([
        self.distribution.batch_shape_tensor(),
        sample_shape,
        self.distribution.event_shape_tensor(),
    ], axis=0)
    return tf.broadcast_to(x, shape)
  return _fn


class Sample(distribution_lib.Distribution):
  """Distribution over IID samples of a given shape.

  Given random variable `X`, one may make a new random variable by concatenating
  samples.  For example, if `X1` and `X2` are iid `Normal(0, 1)` samples then
  `[X1, X2]` is a bi-variate normal random vector.

  #### Mathematical Details

  With `p` the probability density/mass of the function being sampled, and
  `n` the samples taken, the density/mass of this distribution is

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
  #     Equivalently: tf.reduce_sum(s.distribution.log_prob(x), axis=0)
  #
  # `Sample.log_prob` computes the per-{sample, batch} `log_prob`s then sums
  # over the `Sample.sample_shape` dimensions. In the above example `log_prob`
  # dims `0` is summed out, since it is the `sample_shape` dimension.

  # Example 2: `[5, 4]`-draws of a bivariate Normal.

  mvn = tfd.MultivariateNormalDiag(loc=tf.zeros([3, 2]))
  mvn.batch_shape ==> [3]
  mvn.event_shape ==> [2]

  s = tfd.Sample(mvn, sample_shape=[5, 4])
  s.batch_shape ==> [3]
  s.event_shape ==> [5, 4, 2]

  x = s.sample([6, 1])
  # ==> x.shape: [6, 1, 3, 5, 4, 2]

  lp = s.log_prob(x)
  # ==> lp.shape: [6, 1, 3]
  #
  # `s.log_prob` will reduce over the event dims of `mvn`, i.e., dim `5`, then
  # sums over `s.sample_shape` dims `[3, 4]` corresponding to shape (slice)
  # `[5, 4]`.
  ```

  """

  def __init__(
      self,
      distribution,
      sample_shape=(),
      validate_args=False,
      experimental_use_kahan_sum=False,
      name=None):
    """Construct the `Sample` distribution.

    The `event_shape` and `batch_shape` of the `Sample` distribution are
    determined by the args `distribution` and `sample_shape`:

    ```
    s = Sample(distribution, sample_shape)
    ==> s.batch_shape: distribution.batch_shape
    ==> s.event_shape: sample_shape + distribution.event_shape
    ```

    Args:
      distribution: The base distribution instance to transform. Typically an
        instance of `Distribution`.
      sample_shape: `int` scalar or vector `Tensor` representing the shape of a
        single sample.
      validate_args: Python `bool`.  Whether to validate input with asserts.
        If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
      experimental_use_kahan_sum: Python `bool`. When `True`, we use Kahan
        summation to aggregate independent underlying log_prob values, which
        improves against the precision of a naive float32 sum. This can be
        noticeable in particular for large dimensions in float32. See CPU caveat
        on `tfp.math.reduce_kahan_sum`.
      name: The name for ops managed by the distribution.
        Default value: `None` (i.e., `'Sample' + distribution.name`).
    """
    parameters = dict(locals())
    self._experimental_use_kahan_sum = experimental_use_kahan_sum
    with tf.name_scope(name or 'Sample' + distribution.name) as name:
      self._distribution = distribution
      self._sample_shape = tensor_util.convert_nonref_to_tensor(
          sample_shape, dtype_hint=tf.int32, name='sample_shape',
          as_shape_tensor=True)
      super(Sample, self).__init__(
          dtype=self._distribution.dtype,
          reparameterization_type=self._distribution.reparameterization_type,
          validate_args=validate_args,
          allow_nan_stats=self._distribution.allow_nan_stats,
          parameters=parameters,
          name=name)

  @property
  def distribution(self):
    return self._distribution

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        distribution=parameter_properties.BatchedComponentProperties(),
        sample_shape=parameter_properties.ShapeParameterProperties())

  @property
  def sample_shape(self):
    return self._sample_shape

  @property
  def experimental_is_sharded(self):
    return self.distribution.experimental_is_sharded

  def _event_shape_tensor(self):
    return ps.concat([
        ps.reshape(self.sample_shape, shape=[-1]),
        self.distribution.event_shape_tensor(),
    ], axis=0)

  def _event_shape(self):
    s = tf.get_static_value(self.sample_shape)
    if tensorshape_util.rank(s) == 1:
      sample_shape = tf.TensorShape(s)
    else:
      sample_shape = tensorshape_util.constant_value_as_shape(self.sample_shape)
    if (tensorshape_util.rank(sample_shape) is None or
        tensorshape_util.rank(self.distribution.event_shape) is None):
      return tf.TensorShape(None)
    return tensorshape_util.concatenate(sample_shape,
                                        self.distribution.event_shape)

  def _sampling_permutation(self, sample_ndims):
    fake_sample_ndims = ps.rank_from_shape(
        ps.reshape(self.sample_shape, shape=[-1]))
    event_ndims = ps.rank_from_shape(
        self.distribution.event_shape_tensor, self.distribution.event_shape)
    batch_ndims = ps.rank_from_shape(
        self.distribution.batch_shape_tensor, self.distribution.batch_shape)
    return ps.concat([
        ps.range(sample_ndims),
        ps.range(sample_ndims + fake_sample_ndims,
                 sample_ndims + fake_sample_ndims + batch_ndims,
                 dtype=tf.int32),
        ps.range(sample_ndims, sample_ndims + fake_sample_ndims,
                 dtype=tf.int32),
        ps.range(sample_ndims + fake_sample_ndims + batch_ndims,
                 sample_ndims + fake_sample_ndims + batch_ndims + event_ndims,
                 dtype=tf.int32),
    ], axis=0)

  def _sample_n(self, n, seed, **kwargs):
    sample_shape = ps.reshape(self.sample_shape, shape=[-1])
    x = self.distribution.sample(ps.concat([[n], sample_shape], axis=0),
                                 seed=seed,
                                 **kwargs)
    return tf.transpose(a=x, perm=self._sampling_permutation(sample_ndims=1))

  def _sum_fn(self):
    if self._experimental_use_kahan_sum:
      return lambda x, axis: tfp_math.reduce_kahan_sum(x, axis).total
    return tf.math.reduce_sum

  def _prepare_for_underlying(self, x):
    batch_ndims = ps.rank_from_shape(
        self.distribution.batch_shape_tensor,
        self.distribution.batch_shape)
    extra_sample_ndims = ps.rank_from_shape(self.sample_shape)
    event_ndims = ps.rank_from_shape(
        self.distribution.event_shape_tensor,
        self.distribution.event_shape)
    ndims = ps.rank(x)
    # (1) Expand x's dims.
    d = ndims - batch_ndims - extra_sample_ndims - event_ndims
    x = tf.reshape(
        x,
        shape=ps.pad(
            ps.shape(x),
            paddings=[[ps.maximum(0, -d), 0]],
            constant_values=1))
    sample_ndims = ps.maximum(0, d)
    x = tf.transpose(
        x, perm=ps.invert_permutation(self._sampling_permutation(sample_ndims)))
    return x, (sample_ndims, extra_sample_ndims, batch_ndims)

  def _finish_log_prob(self, lp, aux):
    (sample_ndims, extra_sample_ndims, batch_ndims) = aux
    # (1) Ensure lp is fully broadcast in the sample dims, i.e. ensure lp has
    #     full sample shape in the sample axes, before we reduce.
    bcast_lp_shape = ps.broadcast_shape(
        ps.shape(lp),
        ps.concat([ps.ones([sample_ndims], tf.int32),
                   ps.reshape(self.sample_shape, shape=[-1]),
                   ps.ones([batch_ndims], tf.int32)], axis=0))
    lp = tf.broadcast_to(lp, bcast_lp_shape)
    # (2) Make the final reduction.
    axis = ps.range(sample_ndims, sample_ndims + extra_sample_ndims)
    return self._sum_fn()(lp, axis=axis)

  def _sample_and_log_prob(self, sample_shape, seed, **kwargs):
    sample_ndims = ps.rank_from_shape(sample_shape)
    batch_ndims = ps.rank_from_shape(
        self.distribution.batch_shape_tensor,
        self.distribution.batch_shape)
    extra_sample_shape = ps.reshape(self.sample_shape, shape=[-1])
    extra_sample_ndims = ps.rank_from_shape(extra_sample_shape)
    x, lp = self.distribution.experimental_sample_and_log_prob(
        ps.concat([sample_shape, extra_sample_shape], axis=0), seed=seed,
        **kwargs)
    return (
        tf.transpose(x, perm=self._sampling_permutation(sample_ndims)),
        self._finish_log_prob(
            lp, aux=(sample_ndims, extra_sample_ndims, batch_ndims)))

  def _log_prob(self, x, **kwargs):
    x, aux = self._prepare_for_underlying(x)
    return self._finish_log_prob(
        self.distribution.log_prob(x, **kwargs),
        aux)

  def _entropy(self, **kwargs):
    h = self.distribution.entropy(**kwargs)
    n = ps.reduce_prod(self.sample_shape)
    return tf.cast(x=n, dtype=h.dtype) * h

  _mean = _make_summary_statistic('mean')
  _stddev = _make_summary_statistic('stddev')
  _variance = _make_summary_statistic('variance')
  _mode = _make_summary_statistic('mode')

  def _default_event_space_bijector(self):
    # TODO(b/170405182): In scenarios where we can statically prove that it has
    #   no batch part, avoid the transposes by directly using
    #   `self.distribution.experimental_default_event_space_bijector()`.
    bijector = self.distribution.experimental_default_event_space_bijector()
    if bijector is None:
      return None
    bijector = _DefaultSampleBijector(
        self.distribution, self.sample_shape, self._sum_fn(), bijector=bijector)
    # TODO(b/191803645): Come up with an API to set this.
    bijector._use_kahan_sum = self._experimental_use_kahan_sum  # pylint: disable=protected-access
    return bijector

  def _parameter_control_dependencies(self, is_init):
    assertions = []
    sample_shape = None  # Memoize concretization.

    # Check valid shape.
    ndims_ = tensorshape_util.rank(self.sample_shape.shape)
    if is_init != (ndims_ is None):
      msg = 'Argument `sample_shape` must be either a scalar or a vector.'
      if ndims_ is not None:
        if ndims_ > 1:
          raise ValueError(msg)
      elif self.validate_args:
        if sample_shape is None:
          sample_shape = tf.convert_to_tensor(self.sample_shape)
        assertions.append(assert_util.assert_less(
            tf.rank(sample_shape), 2, message=msg))

    # Check valid dtype.
    if is_init:  # No xor check because `dtype` cannot change.
      dtype_ = self.sample_shape.dtype
      if dtype_ is None:
        if sample_shape is None:
          sample_shape = tf.convert_to_tensor(self.sample_shape)
        dtype_ = sample_shape.dtype
      if dtype_util.base_dtype(dtype_) not in {tf.int32, tf.int64}:
        raise TypeError('Argument `sample_shape` must be integer type; '
                        'saw {}.'.format(dtype_util.name(dtype_)))

    # Check valid "value".
    if is_init != tensor_util.is_ref(self.sample_shape):
      sample_shape_ = tf.get_static_value(self.sample_shape)
      msg = 'Argument `sample_shape` must have non-negative values.'
      if sample_shape_ is not None:
        if np.any(np.array(sample_shape_) < 0):
          raise ValueError('{} Saw: {}'.format(msg, sample_shape_))
      elif self.validate_args:
        if sample_shape is None:
          sample_shape = tf.convert_to_tensor(self.sample_shape)
        assertions.append(assert_util.assert_greater(
            sample_shape, -1, message=msg))

    return assertions


class _DefaultSampleBijector(bijector_lib.Bijector):
  """Since tfd.Sample uses transposes, it requires a custom event bijector."""

  def __init__(self, distribution, sample_shape, sum_fn, bijector=None):
    parameters = dict(locals())
    self.distribution = distribution
    if bijector is None:
      bijector = distribution.experimental_default_event_space_bijector()
    self.bijector = bijector
    self.sample_shape = sample_shape
    self._sum_fn = sum_fn
    sample_ndims = ps.rank_from_shape(self.sample_shape)
    super(_DefaultSampleBijector, self).__init__(
        forward_min_event_ndims=(
            self.bijector.forward_min_event_ndims + sample_ndims),
        inverse_min_event_ndims=(
            self.bijector.inverse_min_event_ndims + sample_ndims),
        parameters=parameters)

  def _forward_event_shape(self, shape):
    return self.bijector.forward_event_shape(shape)

  def _forward_event_shape_tensor(self, shape):
    return self.bijector.forward_event_shape_tensor(shape)

  def _inverse_event_shape(self, shape):
    return self.bijector.inverse_event_shape(shape)

  def _inverse_event_shape_tensor(self, shape):
    return self.bijector.inverse_event_shape_tensor(shape)

  def _transpose_around_bijector_fn(self,
                                    bijector_fn,
                                    arg,
                                    src_event_ndims,
                                    dest_event_ndims=None,
                                    fn_reduces_event=False,
                                    **kwargs):
    # This function moves the axes corresponding to `self.sample_shape` to the
    # left of the batch shape, then applies `bijector_fn`, then moves the axes
    # corresponding to `self.sample_shape` back to the event part of the shape.
    #
    # `src_event_ndims` and `dest_event_ndims` indicate the expected event rank
    # (omitting `self.sample_shape`) before and after applying `bijector_fn`.
    #
    # This function arose because forward and inverse ended up being quite
    # similar. It was then only a small generalization to also support {F/I}LDJ.
    batch_ndims = ps.rank_from_shape(self.distribution.batch_shape_tensor,
                                     self.distribution.batch_shape)
    extra_sample_ndims = ps.rank_from_shape(self.sample_shape)
    arg_ndims = ps.rank(arg)
    # (1) Expand arg's dims.
    d = arg_ndims - batch_ndims - extra_sample_ndims - src_event_ndims
    arg = tf.reshape(
        arg,
        shape=ps.pad(
            ps.shape(arg),
            paddings=[[ps.maximum(0, -d), 0]],
            constant_values=1))
    arg_ndims = ps.rank(arg)
    sample_ndims = ps.maximum(0, d)
    # (2) Transpose arg's dims.
    sample_dims = ps.range(0, sample_ndims)
    batch_dims = ps.range(sample_ndims, sample_ndims + batch_ndims)
    extra_sample_dims = ps.range(
        sample_ndims + batch_ndims,
        sample_ndims + batch_ndims + extra_sample_ndims)
    event_dims = ps.range(
        sample_ndims + batch_ndims + extra_sample_ndims,
        arg_ndims)
    perm = ps.concat(
        [sample_dims, extra_sample_dims, batch_dims, event_dims], axis=0)
    arg = tf.transpose(arg, perm=perm)
    # (3) Apply underlying bijector.
    result = bijector_fn(arg, **kwargs)
    # (4) Transpose sample_shape from the sample to the event shape.
    result_ndims = ps.rank(result)
    if fn_reduces_event:
      dest_event_ndims = 0
    d = result_ndims - batch_ndims - extra_sample_ndims - dest_event_ndims
    if fn_reduces_event:
      # In some cases, fn may reduce event too far, i.e. ildj may return a
      # scalar `0.`, which won't work with the transpose we do below.
      result = tf.reshape(
          result,
          shape=ps.pad(
              ps.shape(result),
              paddings=[[ps.maximum(0, -d), 0]],
              constant_values=1))
      result_ndims = ps.rank(result)
    sample_ndims = ps.maximum(0, d)
    sample_dims = ps.range(0, sample_ndims)
    extra_sample_dims = ps.range(sample_ndims,
                                 sample_ndims + extra_sample_ndims)
    batch_dims = ps.range(sample_ndims + extra_sample_ndims,
                          sample_ndims + extra_sample_ndims + batch_ndims)
    event_dims = ps.range(sample_ndims + extra_sample_ndims + batch_ndims,
                          result_ndims)
    perm = ps.concat(
        [sample_dims, batch_dims, extra_sample_dims, event_dims], axis=0)
    return tf.transpose(result, perm=perm)

  def _forward(self, x, **kwargs):
    dist = self.distribution
    event_ndims = ps.rank_from_shape(dist.event_shape_tensor, dist.event_shape)
    bij = self.bijector
    pullback_event_ndims = ps.rank_from_shape(
        lambda: bij.inverse_event_shape_tensor(dist.event_shape_tensor()),
        bij.inverse_event_shape(dist.event_shape))
    return self._transpose_around_bijector_fn(
        bij.forward, arg=x,
        src_event_ndims=pullback_event_ndims, dest_event_ndims=event_ndims)

  def _inverse(self, y, **kwargs):
    dist = self.distribution
    event_ndims = ps.rank_from_shape(dist.event_shape_tensor, dist.event_shape)
    bij = self.bijector
    pullback_event_ndims = ps.rank_from_shape(
        lambda: bij.inverse_event_shape_tensor(dist.event_shape_tensor()),
        bij.inverse_event_shape(dist.event_shape))
    return self._transpose_around_bijector_fn(
        bij.inverse, arg=y,
        src_event_ndims=event_ndims, dest_event_ndims=pullback_event_ndims)

  def _bcast_and_reduce_logdet(self, underlying_ldj):
    # Ensure ldj is fully broadcast in the sample dims, i.e. ensure ldj has
    # full sample shape in the sample axes, before we reduce.
    batch_ndims = ps.rank_from_shape(self.distribution.batch_shape_tensor,
                                     self.distribution.batch_shape)
    extra_sample_ndims = ps.rank_from_shape(self.sample_shape)
    sample_ndims = ps.rank(underlying_ldj) - extra_sample_ndims - batch_ndims
    bcast_ldj_shape = ps.broadcast_shape(
        ps.shape(underlying_ldj),
        ps.concat([ps.ones([sample_ndims], tf.int32),
                   ps.ones([batch_ndims], tf.int32),
                   ps.reshape(self.sample_shape, shape=[-1])], axis=0))
    ldj = tf.broadcast_to(underlying_ldj, bcast_ldj_shape)
    return self._sum_fn(ldj, axis=-1 - ps.range(extra_sample_ndims))

  def _forward_log_det_jacobian(self, x, **kwargs):
    dist = self.distribution
    bij = self.bijector
    pullback_event_ndims = ps.rank_from_shape(
        lambda: bij.inverse_event_shape_tensor(dist.event_shape_tensor()),
        bij.inverse_event_shape(dist.event_shape))
    fn = functools.partial(bij.forward_log_det_jacobian,
                           event_ndims=pullback_event_ndims)
    underlying_ldj = self._transpose_around_bijector_fn(
        fn, arg=x, src_event_ndims=pullback_event_ndims, fn_reduces_event=True)
    return self._bcast_and_reduce_logdet(underlying_ldj)

  def _inverse_log_det_jacobian(self, y, **kwargs):
    event_ndims = ps.rank_from_shape(
        self.distribution.event_shape_tensor, self.distribution.event_shape)
    fn = functools.partial(self.bijector.inverse_log_det_jacobian,
                           event_ndims=event_ndims)
    underlying_ldj = self._transpose_around_bijector_fn(
        fn, arg=y, src_event_ndims=event_ndims, fn_reduces_event=True)
    return self._bcast_and_reduce_logdet(underlying_ldj)


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
    n = ps.reduce_prod(a.sample_shape)
    return tf.cast(x=n, dtype=kl.dtype) * kl


@log_prob_ratio.RegisterLogProbRatio(Sample)
def _sample_log_prob_ratio(p, x, q, y, name=None):
  """Implements `log_prob_ratio` for tfd.Sample."""
  with tf.name_scope(name or 'sample_log_prob_ratio'):
    checks = []
    if p.validate_args or q.validate_args:
      checks.append(tf.debugging.assert_equal(p.sample_shape, q.sample_shape))
    with tf.control_dependencies(checks):
      # pylint: disable=protected-access
      x, aux = p._prepare_for_underlying(x)
      y, _ = q._prepare_for_underlying(y)
      return p._finish_log_prob(
          log_prob_ratio.log_prob_ratio(p.distribution, x, q.distribution, y),
          aux)
      # pylint: enable=protected-access
