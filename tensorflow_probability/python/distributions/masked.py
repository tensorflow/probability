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
"""The MaskedIndependent distribution class."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector as bijector_lib
from tensorflow_probability.python.distributions import batch_broadcast
from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import log_prob_ratio
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util


def _add_event_dims_to_mask(validity_mask, *, dist=None, event_ndims=None):
  validity_mask = tf.convert_to_tensor(validity_mask)
  if event_ndims is None:
    event_ndims = ps.rank_from_shape(dist.event_shape_tensor())
  return tf.reshape(
      validity_mask,
      ps.concat([
          ps.shape(validity_mask),
          ps.ones(event_ndims, dtype=tf.int32)
      ], axis=0))


def _make_masked_fn(fn_name, n_event_shapes, safe_value,
                    make_arg0_safe=False):
  """Implements functions like mean, variance, etc.

  Args:
    fn_name: Name of the method called on the underlying distribution.
    n_event_shapes: Number of event shape repeats in the shape of the underlying
      function's output.
    safe_value: The value to be placed in invalid locations. May be
      `'safe_sample'` to specify we should use the "safe sample" value.
    make_arg0_safe: If `True`, we will apply `self.safe_sample_fn` to ensure the
      argument passed into the underlying routine is a "safe" sample.

  Returns:
    fn: Callable implementing the given function.
  """
  def fn(self, *args, **kwargs):
    if safe_value == 'safe_sample' or make_arg0_safe:  # Only if needed.
      safe_val = tf.stop_gradient(self.safe_sample_fn(self.distribution))

    validity_mask = tf.convert_to_tensor(self.validity_mask)
    if make_arg0_safe:
      x = args[0]
      safe_x = tf.where(
          _add_event_dims_to_mask(validity_mask, dist=self), x, safe_val)
      args = (safe_x,) + args[1:]

    val = getattr(self.distribution, fn_name)(*args, **kwargs)
    if n_event_shapes:
      validity_mask = tf.reshape(
          validity_mask,
          ps.concat(
              [ps.shape(validity_mask)] +
              [ps.ones_like(self.event_shape_tensor())] * n_event_shapes,
              axis=0))
    if safe_value == 'safe_sample':
      sentinel = tf.cast(safe_val, val.dtype)
    else:
      sentinel = tf.cast(safe_value, val.dtype)
    return tf.where(validity_mask, val, sentinel)

  fn.__name__ = f'_{fn_name}'
  return fn


def _fixed_sample(d):
  return d.sample(seed=samplers.zeros_seed())


class Masked(distribution_lib.Distribution):
  """A distribution that masks invalid underlying distributions.

  Sometimes we may want a way of masking out a subset of distributions. Perhaps
  we have labels for only a subset of batch members and want to evaluate a
  log_prob. Or we may want to encode a sparse random variable as a dense
  random variable with a mask applied. In single-program/multiple-data regimes,
  it can be necessary to pad Distributions and the samples thereof to a given
  size in order to achieve the "single-program" desideratum.

  When computing a probability density in this regime, we would like to mask out
  the contributions of invalid batch members. We may also want to ensure that
  the values being sampled are valid parameters for descendant distributions in
  a hierarchical model, even if they are ultimately masked out. This
  distribution answers those requirements. Specifically, for invalid batch
  elements:
  - `log_prob(x) == 0.` for all `x`, with no gradients back to `x`, nor any
    gradients to the parameters of `distribution`.
  - `sample() == tf.stop_gradient(safe_value_fn(distribution))`, with no
    gradients back to the parameters of `distribution`.

  The distribution accepts a mask specified by `validity_mask`, a boolean tensor
  broadcastable with the underlying distribution's batch shape which specifies
  for each batch element whether or not it is valid.

  Entries in `validity_mask` which are `False` denote missing distributions,
  which means that the corresponding entries in the measures (e.g. `prob`)
  and statistics (e.g. `mean`) must not be treated as coming from some real
  distribution. Whenever doing a reduction across those quantites, make sure to
  either mask out the invalid entries or make sure the returned value
  corresponds to the identity element of the reduction. For a couple examples:
  - OK: `reduce_sum(masked_dist.log_prob(x))`
  - OK: `tfd.Independent(masked_dist, ...)`
  - Not OK: `reduce_var(masked_dist.mean())` will underestimate the variance
    because it uses too large an `N`.
  - Not OK: `tf.linalg.cholesky(masked_dist.covariance())` will fail for invalid
    batch elements.

  The default `safe_value_fn` is to draw a fixed-seeded sample from the
  underlying `distribution`.  Since this may be expensive, it is suggested to
  specify a computationally cheaper method. Some options might include:
  - `tfd.Distribution.mode`
  - `tfd.Distribution.mean`
  - `lambda d: d.quantile(.5)` (median)
  - `lambda _: 0.` (if zero is always in the support of d)
  - `lambda d: d.experimental_default_event_space_bijector()(0.)`
  Besides the output of `sample`, results from `safe_value_fn` may also appear
  in (invalid batch members of) `masked.default_event_space_bijector().forward`.

  #### Examples

  ```
  # Use tf.sequence_mask for `range(n) < num_valid`.
  num_valid = 3
  num_entries = 4
  d = tfd.Masked(
      tfd.MultivariateNormalDiag(tf.zeros([2, num_entries, 5]), tf.ones([5])),
      tf.sequence_mask(num_valid, num_entries))
  d.batch_shape  # [2, 4]
  d.event_shape  # [5]
  d.log_prob(tf.zeros([5]))  # shape [2, 4]
  # => [[nonzero, nonzero, nonzero, 0.],
  #     [nonzero, nonzero, nonzero, 0.]]

  # Explicitly denote which elements are valid, adding a new batch dim of 2.
  d = tfd.Masked(tfd.MultivariateNormalDiag(tf.zeros([4, 5]), tf.ones([5])),
                 [[False], [True]])
  d.batch_shape  # [2, 4]
  d.event_shape  # [5]
  d.log_prob(tf.zeros([5]))  # shape [2, 4]
  # => [[0., 0., 0., 0.],
  #     [nonzero, nonzero, nonzero, nonzero]]

  # Use `BatchBroadcast` and `Independent` to achieve the equivalent of adding
  # positional mask functionality to `tfd.Sample`.
  # Suppose we wanted to achieve this:
  # `tfd.Sample(tfd.Normal(tf.zeros(2), 1), [3, 4], validity_mask=mask)`
  # We can write:
  d = tfd.Independent(
      tfd.Masked(tfd.BatchBroadcast(tfd.Normal(0, 1), [2, 3, 4]), mask),
      reinterpreted_batch_ndims=2)
  d.batch_shape  # [2]
  d.event_shape  # [3, 4]
  d.log_prob(tf.ones([3, 4]))  # shape [2]
  ```

  """

  def __init__(self,
               distribution,
               validity_mask,
               safe_sample_fn=_fixed_sample,
               validate_args=False,
               allow_nan_stats=True,
               name=None):
    """Constructs a Masked distribution.

    Args:
      distribution: The underlying distribution, which will be masked.
      validity_mask: Boolean mask where `True` indicates an element is valid.
        `validity_mask` must broadcast with the batch shape of the underlying
        distribution. Invalid batch elements are masked so that sampling returns
        `safe_sample_fn(dist)` in invalid positions and `log_prob(x)` returns
        `0.` for invalid positions.
      safe_sample_fn: A callable which takes a distribution (namely,
        the `distribution` argument) and returns a determinstic, safe sample
        value. This helps to avoid `nan` gradients and allows downstream usage
        of samples from a `Masked` distribution to assume a "safe" even if
        invalid value. (Be careful to ensure that such downstream usages are
        themselves masked!) Note that the result of this function will be
        wrapped in a `tf.stop_gradient` call.
      validate_args: Boolean indicating whether argument assertions should be
        run. May impose performance penalties.
      allow_nan_stats: Boolean indicating whether statistical functions may
        return `nan`, or should instead use asserts where possible.
      name: Optional name for operation scoping.
    """
    parameters = dict(locals())
    with tf.name_scope(name or f'Masked{distribution.name}') as name:
      self._distribution = distribution
      self._validity_mask = tensor_util.convert_nonref_to_tensor(
          validity_mask, dtype_hint=tf.bool)
      self._safe_sample_fn = safe_sample_fn
      super(Masked, self).__init__(
          dtype=distribution.dtype,
          reparameterization_type=distribution.reparameterization_type,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        distribution=parameter_properties.BatchedComponentProperties(),
        validity_mask=parameter_properties.ParameterProperties(
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED))

  @property
  def distribution(self):
    return self._distribution

  @property
  def validity_mask(self):
    return self._validity_mask

  @property
  def safe_sample_fn(self):
    return self._safe_sample_fn

  @property
  def experimental_is_sharded(self):
    return self.distribution.experimental_is_sharded

  def _event_shape(self):
    return self.distribution.event_shape

  def _event_shape_tensor(self):
    return self.distribution.event_shape_tensor()

  def _sample_n(self, n, seed=None, **kwargs):
    validity_mask = tf.convert_to_tensor(self.validity_mask)
    # To avoid the shape gymnastics of drawing extra samples, we delegate
    # sampling to the BatchBroadcast distribution.
    bb = batch_broadcast.BatchBroadcast(self.distribution,
                                        ps.shape(validity_mask))
    samples = bb.sample(n, seed=seed, **kwargs)
    safe_val = tf.stop_gradient(self.safe_sample_fn(self.distribution))
    return tf.where(_add_event_dims_to_mask(validity_mask, dist=self),
                    samples, safe_val)

  _log_prob = _make_masked_fn(
      'log_prob', n_event_shapes=0, safe_value=0., make_arg0_safe=True)
  _prob = _make_masked_fn(
      'prob', n_event_shapes=0, safe_value=1., make_arg0_safe=True)
  _log_cdf = _make_masked_fn(
      'log_cdf', n_event_shapes=0, safe_value=0., make_arg0_safe=True)
  _cdf = _make_masked_fn(
      'cdf', n_event_shapes=0, safe_value=1., make_arg0_safe=True)
  _log_survival_function = _make_masked_fn(
      'log_survival_function', n_event_shapes=0, safe_value=-float('inf'),
      make_arg0_safe=True)
  _survival_function = _make_masked_fn(
      'survival_function', n_event_shapes=0, safe_value=0.,
      make_arg0_safe=True)

  _entropy = _make_masked_fn(
      'entropy', n_event_shapes=0, safe_value=0.)
  _mode = _make_masked_fn(
      'mode', n_event_shapes=1, safe_value='safe_sample')
  _mean = _make_masked_fn(
      'mean', n_event_shapes=1, safe_value='safe_sample')
  _variance = _make_masked_fn(
      'variance', n_event_shapes=1, safe_value=0.)
  _stddev = _make_masked_fn(
      'stddev', n_event_shapes=1, safe_value=0.)
  _covariance = _make_masked_fn(
      'covariance', n_event_shapes=2, safe_value=0.)
  _quantile = _make_masked_fn(
      'quantile', n_event_shapes=1, safe_value='safe_sample')

  def _default_event_space_bijector(self, *args, **kwargs):
    underlying_bijector = (
        self.distribution.experimental_default_event_space_bijector())
    if underlying_bijector is None:
      return None
    return _MaskedBijector(self, underlying_bijector)


@kullback_leibler.RegisterKL(Masked, Masked)
def _kl_masked_masked(a, b, name=None):
  """KL divergence between Masked distributions."""
  with tf.name_scope(name or 'kl_masked_masked'):
    a_valid = tf.convert_to_tensor(a.validity_mask)
    b_valid = tf.convert_to_tensor(b.validity_mask)
    underlying_kl = kullback_leibler.kl_divergence(
        a.distribution, b.distribution)

    # The treatment for KL is as follows:
    # When both random variables are valid, the underlying KL applies.
    # When neither random variable is valid, the KL is 0., i.e.
    # `a log a - a log b = 0` because log a and log b are everywhere 0.
    # When exactly one is valid, we (a) raise an assertion error, if either
    # distribution's allow_nan_stats is set to False, or (b) return nan in
    # such positions.
    asserts = []
    if not (a.allow_nan_stats and b.allow_nan_stats):
      asserts.append(assert_util.assert_equal(
          a_valid, b_valid,
          message='KL is only valid for matching mask values'))
    with tf.control_dependencies(asserts):
      both_valid = (a_valid & b_valid)
      neither_valid = (~a_valid) & (~b_valid)
      dtype = underlying_kl.dtype
      return tf.where(both_valid, underlying_kl,
                      tf.where(neither_valid,
                               tf.zeros([], dtype), float('nan')))


@log_prob_ratio.RegisterLogProbRatio(Masked)
def _masked_log_prob_ratio(p, x, q, y, name=None):
  """Computes log p(x) - log q(y) for Masked p, q."""
  with tf.name_scope(name or 'masked_log_prob_ratio'):
    p_valid = tf.convert_to_tensor(p.validity_mask)
    safe_x = tf.where(_add_event_dims_to_mask(p_valid, dist=p),
                      x, tf.stop_gradient(p.safe_sample_fn(p.distribution)))
    q_valid = tf.convert_to_tensor(q.validity_mask)
    safe_y = tf.where(_add_event_dims_to_mask(q_valid, dist=q),
                      y, tf.stop_gradient(q.safe_sample_fn(q.distribution)))
    underlying = log_prob_ratio.log_prob_ratio(
        p.distribution, safe_x, q.distribution, safe_y)
    asserts = []
    # As with KL, we return the underlying log_prob_ratio where both are valid,
    # `0.` where neither is valid, and `nan` otherwise (or an assertion if
    # either distribution does not `allow_nan_stats`).
    if not (p.allow_nan_stats and p.allow_nan_stats):
      asserts.append(assert_util.assert_equal(
          p_valid, q_valid,
          message='Masked log_prob_ratio only valid for matching mask values'))
    with tf.control_dependencies(asserts):
      both_valid = (p_valid & q_valid)
      neither_valid = (~p_valid) & (~q_valid)
      return tf.where(both_valid, underlying,
                      tf.where(neither_valid,
                               tf.zeros([], dtype=underlying.dtype),
                               float('nan')))


class _MaskedBijector(bijector_lib.Bijector):
  """Event space bijector for Masked distributions."""

  def __init__(self, masked, underlying_bijector):
    self._masked = masked
    self._bijector = underlying_bijector
    super(_MaskedBijector, self).__init__(
        validate_args=underlying_bijector.validate_args,
        dtype=underlying_bijector.dtype,
        forward_min_event_ndims=underlying_bijector.forward_min_event_ndims,
        inverse_min_event_ndims=underlying_bijector.inverse_min_event_ndims)

  def _forward_event_shape(self, x):
    return self._bijector.forward_event_shape(x)

  def _forward_event_shape_tensor(self, x):
    return self._bijector.forward_event_shape_tensor(x)

  def _inverse_event_shape(self, y):
    return self._bijector.inverse_event_shape(y)

  def _inverse_event_shape_tensor(self, y):
    return self._bijector.inverse_event_shape_tensor(y)

  def _make_safe_x(self, x, validity_mask):
    bij = self._bijector
    masked = self._masked
    pullback_event_ndims = ps.rank_from_shape(
        lambda: bij.inverse_event_shape_tensor(masked.event_shape_tensor()),
        self._bijector.inverse_event_shape(masked.event_shape))
    pullback_event_mask = _add_event_dims_to_mask(
        validity_mask, event_ndims=pullback_event_ndims)
    # We presume that 0 in unconstrained space is safe.
    return tf.where(pullback_event_mask, x, 0.)

  def _forward(self, x):
    mask = self._masked.validity_mask
    safe_x = self._make_safe_x(x, mask)
    return self._make_safe_y(self._bijector.forward(safe_x), mask)

  def _forward_log_det_jacobian(self, x):
    validity_mask = tf.convert_to_tensor(self._masked.validity_mask)
    safe_x = self._make_safe_x(x, validity_mask)
    return tf.where(validity_mask,
                    self._bijector.forward_log_det_jacobian(safe_x),
                    0.)

  def _make_safe_y(self, y, validity_mask):
    safe_val = tf.stop_gradient(
        self._masked.safe_sample_fn(self._masked.distribution))
    event_mask = _add_event_dims_to_mask(validity_mask, dist=self._masked)
    return tf.where(event_mask, y, safe_val)

  def _inverse(self, y):
    safe_y = self._make_safe_y(y, self._masked.validity_mask)
    return self._bijector.inverse(safe_y)

  def _inverse_log_det_jacobian(self, y):
    validity_mask = tf.convert_to_tensor(self._masked.validity_mask)
    safe_y = self._make_safe_y(y, validity_mask)
    return tf.where(validity_mask,
                    self._bijector.inverse_log_det_jacobian(safe_y),
                    0.)
