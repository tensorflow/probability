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
"""The Independent distribution class."""

import collections

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import log_prob_ratio
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math import generic

from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import


class _Independent(distribution_lib.Distribution):
  """Independent distribution from batch of distributions.

  This distribution is useful for regarding a collection of independent,
  non-identical distributions as a single random variable. For example, the
  `Independent` distribution composed of a collection of `Bernoulli`
  distributions might define a distribution over an image (where each
  `Bernoulli` is a distribution over each pixel).

  More precisely, a collection of `B` (independent) `E`-variate random variables
  (rv) `{X_1, ..., X_B}`, can be regarded as a `[B, E]`-variate random variable
  `(X_1, ..., X_B)` with probability
  `p(x_1, ..., x_B) = p_1(x_1) * ... * p_B(x_B)` where `p_b(X_b)` is the
  probability of the `b`-th rv. More generally `B, E` can be arbitrary shapes.

  Similarly, the `Independent` distribution specifies a distribution over `[B,
  E]`-shaped events. It operates by reinterpreting the rightmost batch dims as
  part of the event dimensions. The `reinterpreted_batch_ndims` parameter
  controls the number of batch dims which are absorbed as event dims;
  `reinterpreted_batch_ndims <= len(batch_shape)`.  For example, the `log_prob`
  function entails a `reduce_sum` over the rightmost `reinterpreted_batch_ndims`
  after calling the base distribution's `log_prob`.  In other words, since the
  batch dimension(s) index independent distributions, the resultant multivariate
  will have independent components.

  #### Mathematical Details

  The probability function is,

  ```none
  prob(x; reinterpreted_batch_ndims) = tf.reduce_prod(
      dist.prob(x),
      axis=-1-range(reinterpreted_batch_ndims))
  ```

  #### Examples

  ```python
  tfd = tfp.distributions

  # Make independent distribution from a 2-batch Normal.
  ind = tfd.Independent(
      distribution=tfd.Normal(loc=[-1., 1], scale=[0.1, 0.5]),
      reinterpreted_batch_ndims=1)

  # All batch dims have been 'absorbed' into event dims.
  ind.batch_shape  # ==> []
  ind.event_shape  # ==> [2]

  # Make independent distribution from a 2-batch bivariate Normal.
  ind = tfd.Independent(
      distribution=tfd.MultivariateNormalDiag(
          loc=[[-1., 1], [1, -1]],
          scale_identity_multiplier=[1., 0.5]),
      reinterpreted_batch_ndims=1)

  # All batch dims have been 'absorbed' into event dims.
  ind.batch_shape  # ==> []
  ind.event_shape  # ==> [2, 2]
  ```

  """

  @deprecation.deprecated_arg_values(
      '2022-03-01',
      'Please pass an integer value for `reinterpreted_batch_ndims`. The '
      'current behavior corresponds to `reinterpreted_batch_ndims=tf.size('
      'distribution.batch_shape_tensor()) - 1`.',
      reinterpreted_batch_ndims=None)
  def __init__(self,
               distribution,
               reinterpreted_batch_ndims=None,
               validate_args=False,
               experimental_use_kahan_sum=False,
               name=None):
    """Construct an `Independent` distribution.

    Args:
      distribution: The base distribution instance to transform. Typically an
        instance of `Distribution`.
      reinterpreted_batch_ndims: Scalar, integer number of rightmost batch dims
        which will be regarded as event dims. When `None` all but the first
        batch axis (batch axis 0) will be transferred to event dimensions
        (analogous to `tf.layers.flatten`).
      validate_args: Python `bool`.  Whether to validate input with asserts.
        If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
      experimental_use_kahan_sum: Python `bool`. When `True`, we use Kahan
        summation to aggregate independent underlying log_prob values, which
        improves against the precision of a naive float32 sum. This can be
        noticeable in particular for large dimensions in float32. See CPU caveat
        on `tfp.math.reduce_kahan_sum`.
      name: The name for ops managed by the distribution.
        Default value: `Independent + distribution.name`.

    Raises:
      ValueError: if `reinterpreted_batch_ndims` exceeds
        `distribution.batch_ndims`
    """
    parameters = dict(locals())
    self._experimental_use_kahan_sum = experimental_use_kahan_sum
    with tf.name_scope(name or ('Independent' + distribution.name)) as name:
      self._distribution = distribution

      if reinterpreted_batch_ndims is None:
        # If possible, statically infer reinterpreted_batch_ndims.
        batch_ndims = tensorshape_util.rank(distribution.batch_shape)
        if batch_ndims is not None:
          self._static_reinterpreted_batch_ndims = max(0, batch_ndims - 1)
          self._reinterpreted_batch_ndims = ps.convert_to_shape_tensor(
              self._static_reinterpreted_batch_ndims,
              dtype_hint=tf.int32,
              name='reinterpreted_batch_ndims')
        else:
          self._reinterpreted_batch_ndims = None
          self._static_reinterpreted_batch_ndims = None

      else:
        self._reinterpreted_batch_ndims = tensor_util.convert_nonref_to_tensor(
            reinterpreted_batch_ndims,
            dtype_hint=tf.int32,
            as_shape_tensor=True,
            name='reinterpreted_batch_ndims')
        static_val = tf.get_static_value(self._reinterpreted_batch_ndims)
        self._static_reinterpreted_batch_ndims = (
            None if static_val is None else int(static_val))

      super(_Independent, self).__init__(
          dtype=self._distribution.dtype,
          reparameterization_type=self._distribution.reparameterization_type,
          validate_args=validate_args,
          allow_nan_stats=self._distribution.allow_nan_stats,
          parameters=parameters,
          name=name)

  @property
  def distribution(self):
    return self._distribution

  @property
  def reinterpreted_batch_ndims(self):
    return self._reinterpreted_batch_ndims

  @property
  def experimental_is_sharded(self):
    return self.distribution.experimental_is_sharded

  def _get_reinterpreted_batch_ndims(self,
                                     distribution_batch_shape_tensor=None):
    if self._static_reinterpreted_batch_ndims is not None:
      return self._static_reinterpreted_batch_ndims
    if self._reinterpreted_batch_ndims is not None:
      return tf.convert_to_tensor(self._reinterpreted_batch_ndims)

    if distribution_batch_shape_tensor is None:
      distribution_batch_shape_tensor = self.distribution.batch_shape_tensor()
    return ps.cast(
        ps.maximum(0, ps.size(distribution_batch_shape_tensor) - 1),
        np.int32)

  # TODO(davmre): Delete this override.
  # The default slicing machinery should work here after we remove support for
  # the deprecated init arg `reinterpreted_batch_ndims=None`.
  def __getitem__(self, slices):
    # Because slicing is parameterization-dependent, we only implement slicing
    # for instances of Independent, not subclasses thereof.
    if type(self) not in (_Independent, Independent):  # pylint: disable=unidiomatic-typecheck
      return super(_Independent, self).__getitem__(slices)

    if self._static_reinterpreted_batch_ndims is None:
      raise NotImplementedError(
          'Cannot slice Independent with non-static reinterpreted_batch_ndims')
    slices = (tuple(slices) if isinstance(slices, collections.abc.Sequence)
              else (slices,))
    if Ellipsis not in slices:
      slices = slices + (Ellipsis,)
    slices = slices + (slice(None),) * int(
        self._static_reinterpreted_batch_ndims)
    return self.copy(
        distribution=self.distribution[slices],
        reinterpreted_batch_ndims=self._static_reinterpreted_batch_ndims)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        distribution=parameter_properties.BatchedComponentProperties(
            # TODO(davmre): replace with `self.reinterpreted_batch_ndims` once
            # support for `reinterpreted_batch_ndims=None` has been removed.
            event_ndims=lambda self: self._get_reinterpreted_batch_ndims()),  # pylint: disable=protected-access
        reinterpreted_batch_ndims=(
            parameter_properties.ShapeParameterProperties()))

  def _batch_shape_tensor(self):
    batch_shape = self.distribution.batch_shape_tensor()
    batch_ndims = ps.rank_from_shape(
        batch_shape, self.distribution.batch_shape)
    return batch_shape[
        :batch_ndims - self._get_reinterpreted_batch_ndims(batch_shape)]

  def _batch_shape(self):
    batch_shape = self.distribution.batch_shape
    if (self._static_reinterpreted_batch_ndims is None or
        tensorshape_util.rank(batch_shape) is None):
      return tf.TensorShape(None)
    d = (tensorshape_util.rank(batch_shape) -
         self._static_reinterpreted_batch_ndims)
    return batch_shape[:d]

  def _event_shape_tensor(self):
    # If both `distribution.batch_shape` and `distribution.tensor_shape` are
    # known statically, then Distribution won't call this method.  But this
    # method may be called wheh only one of them is statically known.
    batch_shape = self.distribution.batch_shape
    if not tensorshape_util.is_fully_defined(batch_shape):
      batch_shape = self.distribution.batch_shape_tensor()
    batch_ndims = ps.rank_from_shape(batch_shape)
    event_shape = self.distribution.event_shape
    if not tensorshape_util.is_fully_defined(event_shape):
      event_shape = self.distribution.event_shape_tensor()
    return ps.concat([
        ps.convert_to_shape_tensor(batch_shape)[
            batch_ndims - self._get_reinterpreted_batch_ndims(batch_shape):],
        event_shape
    ], axis=0)

  def _event_shape(self):
    batch_shape = self.distribution.batch_shape
    if self._static_reinterpreted_batch_ndims is None:
      return tf.TensorShape(None)
    if tensorshape_util.rank(batch_shape) is not None:
      reinterpreted_batch_shape = batch_shape[
          tensorshape_util.rank(batch_shape) -
          self._static_reinterpreted_batch_ndims:]
    else:
      reinterpreted_batch_shape = tf.TensorShape(
          [None] * int(self._static_reinterpreted_batch_ndims))
    return tensorshape_util.concatenate(reinterpreted_batch_shape,
                                        self.distribution.event_shape)

  def _sample_n(self, n, seed, **kwargs):
    return self.distribution.sample(sample_shape=n, seed=seed, **kwargs)

  def _sum_fn(self):
    if self._experimental_use_kahan_sum:
      return lambda x, axis: generic.reduce_kahan_sum(x, axis).total
    return tf.math.reduce_sum

  def _sample_and_log_prob(self, sample_shape, seed, **kwargs):
    x, lp = self.distribution.experimental_sample_and_log_prob(
        sample_shape, seed=seed, **kwargs)
    return x, self._reduce(self._sum_fn(), lp)

  def _log_prob(self, x, **kwargs):
    return self._reduce(
        self._sum_fn(), self.distribution.log_prob(x, **kwargs))

  def _unnormalized_log_prob(self, x, **kwargs):
    return self._reduce(
        self._sum_fn(), self.distribution.unnormalized_log_prob(x, **kwargs))

  def _log_cdf(self, x, **kwargs):
    return self._reduce(self._sum_fn(), self.distribution.log_cdf(x, **kwargs))

  def _entropy(self, **kwargs):
    # NOTE: If self._reinterpreted_batch_ndims is None, we could avoid a read
    # of self.distribution.batch_shape_tensor() in `self._reduce` here by
    # passing in `tf.shape(self.distribution.entropy())` to use instead.
    return self._reduce(self._sum_fn(), self.distribution.entropy(**kwargs))

  def _mean(self, **kwargs):
    return self.distribution.mean(**kwargs)

  def _variance(self, **kwargs):
    return self.distribution.variance(**kwargs)

  def _stddev(self, **kwargs):
    return self.distribution.stddev(**kwargs)

  def _mode(self, **kwargs):
    return self.distribution.mode(**kwargs)

  def _default_event_space_bijector(self):
    bijector = self.distribution.experimental_default_event_space_bijector()
    if (bijector is not None and
        getattr(bijector,
                '_use_kahan_sum',
                False) != self._experimental_use_kahan_sum):
      # Copy in case the wrapped distribution doesn't construct a brand-new
      # bijector each time.
      bijector = bijector.copy()
      # TODO(b/191803645): Come up with an API to set this.
      bijector._use_kahan_sum = self._experimental_use_kahan_sum  # pylint: disable=protected-access
    return bijector

  def _parameter_control_dependencies(self, is_init):
    # self, distribution, reinterpreted_batch_ndims, validate_args):
    assertions = []

    batch_ndims = tensorshape_util.rank(self.distribution.batch_shape)
    if (batch_ndims is not None
        and self._static_reinterpreted_batch_ndims is not None):
      if is_init and self._static_reinterpreted_batch_ndims > batch_ndims:
        raise ValueError('reinterpreted_batch_ndims({}) cannot exceed '
                         'distribution.batch_ndims({})'.format(
                             self._static_reinterpreted_batch_ndims,
                             batch_ndims))
    elif self.validate_args:
      batch_shape_tensor = self.distribution.batch_shape_tensor()
      assertions.append(
          assert_util.assert_less_equal(
              self._get_reinterpreted_batch_ndims(batch_shape_tensor),
              ps.rank_from_shape(batch_shape_tensor),
              message=('reinterpreted_batch_ndims cannot exceed '
                       'distribution.batch_ndims')))
    return assertions

  def _reduce(self, op, stat):
    axis = 1 + ps.range(self._get_reinterpreted_batch_ndims())
    return op(stat, axis=-axis)


class Independent(
    _Independent, distribution_lib.AutoCompositeTensorDistribution):

  def __new__(cls, *args, **kwargs):
    """Maybe return a non-`CompositeTensor` `_Independent`."""

    if cls is Independent:
      if args:
        distribution = args[0]
      else:
        distribution = kwargs.get('distribution')

      if not isinstance(distribution, tf.__internal__.CompositeTensor):
        return _Independent(*args, **kwargs)
    return super(Independent, cls).__new__(cls)


Independent.__doc__ = _Independent.__doc__ + '\n' + (
    'If `distribution` is a `CompositeTensor`, then the resulting '
    '`Independent` instance is a `CompositeTensor` as well. Otherwise, a '
    'non-`CompositeTensor` `_Independent` instance is created instead. '
    'Distribution subclasses that inherit from `Independent` will also inherit '
    'from `CompositeTensor`.')


@kullback_leibler.RegisterKL(_Independent, _Independent)
def _kl_independent(a, b, name='kl_independent'):
  """Batched KL divergence `KL(a || b)` for Independent distributions.

  We can leverage the fact that
  ```
  KL(Independent(a) || Independent(b)) = sum(KL(a || b))
  ```
  where the sum is over the `reinterpreted_batch_ndims`.

  Args:
    a: Instance of `Independent`.
    b: Instance of `Independent`.
    name: (optional) name to use for created ops. Default 'kl_independent'.

  Returns:
    Batchwise `KL(a || b)`.

  Raises:
    ValueError: If the event space for `a` and `b`, or their underlying
      distributions don't match.
  """
  p = a.distribution
  q = b.distribution

  # The KL between any two (non)-batched distributions is a scalar.
  # Given that the KL between two factored distributions is the sum, i.e.
  # KL(p1(x)p2(y) || q1(x)q2(y)) = KL(p1 || q1) + KL(q1 || q2), we compute
  # KL(p || q) and do a `reduce_sum` on the reinterpreted batch dimensions.
  if (tensorshape_util.is_fully_defined(a.event_shape) and
      tensorshape_util.is_fully_defined(b.event_shape)):
    if a.event_shape == b.event_shape:
      if p.event_shape == q.event_shape:
        num_reduce_dims = (tensorshape_util.rank(a.event_shape) -
                           tensorshape_util.rank(p.event_shape))
        reduce_dims = [-i - 1 for i in range(0, num_reduce_dims)]

        return tf.reduce_sum(
            kullback_leibler.kl_divergence(p, q, name=name), axis=reduce_dims)
      else:
        raise NotImplementedError('KL between Independents with different '
                                  'event shapes not supported.')
    else:
      raise ValueError('Event shapes do not match.')
  else:
    p_event_shape_tensor = p.event_shape_tensor()
    q_event_shape_tensor = q.event_shape_tensor()
    # NOTE: We could optimize by passing the event_shape_tensor of p and q
    # to a.event_shape_tensor() and b.event_shape_tensor().
    a_event_shape_tensor = a.event_shape_tensor()
    b_event_shape_tensor = b.event_shape_tensor()
    with tf.control_dependencies(
        [
            assert_util.assert_equal(
                a_event_shape_tensor, b_event_shape_tensor,
                message='Event shapes do not match.'),
            assert_util.assert_equal(
                p_event_shape_tensor, q_event_shape_tensor,
                message='Event shapes do not match.'),
        ]):
      num_reduce_dims = (
          ps.rank_from_shape(
              a_event_shape_tensor, a.event_shape) -
          ps.rank_from_shape(
              p_event_shape_tensor, p.event_shape))
      reduce_dims = ps.range(-num_reduce_dims, 0, 1)
      return tf.reduce_sum(
          kullback_leibler.kl_divergence(p, q, name=name), axis=reduce_dims)


@log_prob_ratio.RegisterLogProbRatio(_Independent)
def _independent_log_prob_ratio(p, x, q, y, name=None):
  """Sum-of-diffs log(p(x)/q(y)) for `Independent`s."""
  with tf.name_scope(name or 'independent_log_prob_ratio'):
    checks = []
    if p.validate_args or q.validate_args:
      checks.append(tf.debugging.assert_equal(
          p.reinterpreted_batch_ndims, q.reinterpreted_batch_ndims))
    if p._experimental_use_kahan_sum or q._experimental_use_kahan_sum:  # pylint: disable=protected-access
      sum_fn = lambda x, axis: generic.reduce_kahan_sum(x, axis).total
    else:
      sum_fn = tf.reduce_sum
    with tf.control_dependencies(checks):
      return sum_fn(
          log_prob_ratio.log_prob_ratio(p.distribution, x, q.distribution, y),
          axis=-1 - ps.range(p.reinterpreted_batch_ndims))
