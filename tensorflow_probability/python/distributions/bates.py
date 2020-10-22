# Copyright 2020 The TensorFlow Probability Authors.
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
"""The Bates distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import sys

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import sigmoid as sigmoid_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util

__all__ = [
    'Bates',
]


BATES_TOTAL_COUNT_STABILITY_LIMITS = {
    tf.float64: 75.,
    np.float64: 75.,
    np.finfo(np.float64).dtype: 75.,

    tf.float32: 25.,
    np.float32: 25.,
    np.finfo(np.float32).dtype: 25.,

    # Not an allowed type but we keep this here for the record.
    tf.float16: 7.,
    np.float16: 7.,
    np.finfo(np.float16).dtype: 7.,
}


class Bates(distribution.Distribution):
  """Bates distribution.

  The Bates distribution is the distribution of the average of `total_count`
  independent samples from `Uniform(low, high)`. It is parameterized by the
  interval bounds `low` and `high`, and `total_count`, the number of samples.

  Although some care has been taken to avoid numerical issues, the `pdf`, `cdf`,
  and log versions thereof may still exhibit numerical instability. They are
  relatively stable near the tails; however near the mode they are unstable if
  `total_count` is greater than about `75` for `tf.float64`, `25` for
  `tf.float32`, and `7` for `tf.float16`. Beyond these limits a warning will be
  shown if `validate_args=False`; otherwise an exception is thrown. For high
  `total_count`, consider using a `Normal` approximation.

  #### Mathematical Details

  The probability density function (pdf) is supported in the interval
  `[low, high]`. If `[low, high]` is the unit interval `[0, 1]`, the pdf
  is,

  ```none
  pdf(x; n, 0, 1) =
    ((n / (n-1)!) sum_{k=0}^j (-1)^k (n choose k) (nx - k)^{n-1}
  ```

  where
  * `total_count = n`,
  * `j = floor(nx)`
  * `n!` is the factorial of `n`,
  * `(n choose k)` is the binomial coefficient `n! / (k!(n - k)!),

  For arbitrary intervals `[low, high]`, the pdf is,

  ```none
  pdf(x; n, low, high) = pdf((x - low) / (high - low); n, 0, 1) / (high - low)
  ```

  #### Examples

  Create a single distribution for the mean of 5 uniform random variables on the
  interval `[-10, 5]`.

  ```python
  dist = tfd.Bates(total_count=5., low=-10., high=5.)
  ```

  Create a 3-batch of distributions with varying total counts and intervals.

  ```python
  counts = [1., 2., 5.]
  # high will be broadcast to [100., 100., 100.]
  dist = tfd.Bates(total_count=counts, low=[0., 5., 10.], high=100.)
  ```

  Compute some values for the pdf.

  ```python
  dist.probs(50.).eval()    # shape: [3]
  x = [[50., 50., 50.],
       [5., 10., 20.]]      # shape: [2, 3]
  dist.probs(x).eval()      # shape: [2]
  ```
  """

  def __init__(self,
               total_count,
               low=0.,
               high=1.,
               validate_args=False,
               allow_nan_stats=True,
               name='Bates'):
    """Construct a Bates distribution.

    Args:
      total_count: Non-negative integer-valued `Tensor` with shape broadcastable
        to the batch shape `[N1,..., Nm]`, `m >= 0`. This controls the number of
        samples of `Uniform(low, high)` to take the mean of.
      low: Floating point `Tensor` representing the lower bounds of the support.
        Should be broadcastable to `[N1,..., Nm]` with `m >= 0`, the same dtype
        as `total_count`, and `low < high` component-wise, after broadcasting.
        Defaults to `0`.
      high: Floating point `Tensor` representing the upper bounds of the
        support.  Should be broadcastable to `[N1,..., Nm]` with `m >= 0`, the
        same dtype as `total_count`, and `low < high` component-wise, after
        broadcasting.  Defaults to `1`.
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
      dtype = dtype_util.common_dtype([low, high], dtype_hint=tf.float32)
      assert dtype in (tf.float32, tf.float64), (
          '`Bates` only supports `tf.float32` or `tf.float64`')
      self._total_count = tensor_util.convert_nonref_to_tensor(
          total_count, name='total_count', dtype_hint=dtype)
      self._low = tensor_util.convert_nonref_to_tensor(
          low, dtype=dtype, name='low')
      self._high = tensor_util.convert_nonref_to_tensor(
          high, dtype=dtype, name='high')
      super(Bates, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        total_count=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=parameter_properties
            .BIJECTOR_NOT_IMPLEMENTED),
        low=parameter_properties.ParameterProperties(),
        # TODO(b/169874884): Support decoupled parameterization.
        high=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=parameter_properties
            .BIJECTOR_NOT_IMPLEMENTED,))

  @property
  def total_count(self):
    """Number of `Uniform` trials used to construct a sample."""
    return self._total_count

  @property
  def low(self):
    """Lower bound of the support."""
    return self._low

  @property
  def high(self):
    """Upper bound of the support."""
    return self._high

  def _params_list(self):
    return [('total_count', self._total_count),
            ('low', self._low),
            ('high', self._high)]

  def _batch_shape_tensor(self):
    return functools.reduce(
        ps.broadcast_shape,
        [ps.shape(param) for _, param in self._params_list()])

  def _batch_shape(self):
    return functools.reduce(
        tf.broadcast_static_shape,
        [param.shape for _, param in self._params_list()])

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    total_count = tf.cast(self.total_count, tf.int32)
    low = tf.convert_to_tensor(self.low)
    high = tf.convert_to_tensor(self.high)
    return _sample_bates(
        ps.broadcast_to(total_count, self._batch_shape_tensor()),
        low, high, n, seed=seed)

  def _prob(self, value):
    return _bates_pdf(self.total_count, self.low, self.high, self.dtype, value)

  def _cdf(self, value):
    return _bates_cdf(self.total_count, self.low, self.high, self.dtype, value)

  def _mean(self):
    return (self.low + self.high) / 2.

  @distribution_util.AppendDocstring(
      'For `n = 1`, any value in `(low, high)` is a mode; this gives the mean.')
  def _mode(self):
    return self._mean()

  def _variance(self):
    return tf.math.square(self.high - self.low) / (
        12. * tf.cast(self.total_count, self.dtype))

  def _default_event_space_bijector(self):
    return sigmoid_bijector.Sigmoid(
        low=self.low, high=self.high, validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []

    if is_init:
      try:
        self._batch_shape()
      except ValueError:
        raise ValueError(
            'Arguments `total_count`, `low` and `high` must have compatible '
            'shapes; total_count.shape={}, low.shape={}, '
            'high.shape={}.'.format(
                tf.shape(self.total_count),
                tf.shape(self.low),
                tf.shape(self.high)))

    assertions = []

    if is_init != tensor_util.is_ref(self.total_count):
      total_count = tf.convert_to_tensor(self.total_count)
      limit = BATES_TOTAL_COUNT_STABILITY_LIMITS[self.dtype]
      msg = '`total_count` must be representable as a 32-bit integer.'
      assertions.extend([
          assert_util.assert_positive(
              total_count,
              message='`total_count` must be positive.'),
          distribution_util.assert_casting_closed(
              total_count,
              target_dtype=tf.int32,
              message=msg),
          assert_util.assert_less_equal(
              tf.cast(total_count, self.dtype),
              tf.cast(limit, self.dtype),
              message='`total_count` > {} is numerically unstable.'.format(
                  limit))
      ])

    if is_init != (tensor_util.is_ref(self.low) or
                   tensor_util.is_ref(self.high)):
      assertions.append(assert_util.assert_less(
          self.low, self.high, message='`low` must be less than `high`.'))

    return assertions

  _composite_tensor_nonshape_params = ('low', 'high')

  _composite_tensor_shape_params = ('total_count',)


# TODO(b/157665707): Investigate alternative PDF formulas / computations.
def _bates_pdf(total_count, low, high, dtype, value):
  """Compute the Bates pdf.

  Internally, the (standard, unnormalized) pdf is computed by the formula

  ```none
  pdf = sum_{k=0}^j (-1)^k (n choose k) (nx - k)^{n - 1}
  ```

  where
  * `n = total_count`,
  * `x = value` the value to compute the probability of, and
  * `j = floor(nx)`.

  This is shifted to `[low, high]` and normalized. Since the pdf is symmetric,
  we only compute the left half, which keeps the number of terms lower.

  Computation is batched, using `tf.math.segment_sum()`. For this reason this is
  not compatible with `tf.vectorized_map()`.

  All input parameters should have compatible dtypes and shapes.

  Args:
    total_count: `Tensor` with integer values, as given to the `Bates`
      constructor.
    low: Float `Tensor`, as given to the `Bates` constructor.
    high: Float `Tensor`, as given to the `Bates` constructor.
    dtype: The dtype of the output.
    value: Float `Tensor`. Input value to `prob()`.
  Returns:
    pdf: Float `Tensor`. See above formula.
  """
  total_count = tf.cast(total_count, dtype)
  low = tf.convert_to_tensor(low)
  high = tf.convert_to_tensor(high)

  # Warn the user if they try to compute a pdf with high `total_count`.  This
  # warning is here instead of `_parameter_control_dependencies()` because
  # nested calls to `_name_and_control_scope` (e.g. `log_survival_function`) can
  # result in multiple warnings being added and multiple tensor
  # conversions. Also `sample()` does not have the same numerical issues.
  with tf.control_dependencies([_stability_limit_tensor(total_count, dtype)]):
    # Center and adjust `value` using limits and symmetry.
    range_ = high - low
    value_centered = (value - low) / range_
    value_adj = tf.clip_by_value(value_centered, 0., 1.)
    value_adj = tf.where(value_adj < .5, value_adj, 1. - value_adj)
    value_adj = tf.where(tf.math.is_finite(value_adj), value_adj, 0.)
    # Flatten to make segments; need to broadcast before flattening.
    shape = ps.broadcast_shape(ps.shape(value_adj), ps.shape(total_count))
    total_count_b = ps.broadcast_to(total_count, shape)
    total_count_x_value_adj_b = total_count * value_adj
    total_count_f = tf.reshape(total_count_b, [-1])
    total_count_x_value_adj_f = tf.reshape(total_count_x_value_adj_b, [-1])
    # Create segmented terms of summation.
    num_terms_f = tf.cast(tf.math.floor(total_count_x_value_adj_f + 1),
                          dtype=tf.int32)
    term_idx_s = tf.cast(_segmented_range(num_terms_f), dtype)  # aka `k`
    total_count_s = tf.repeat(total_count_f, num_terms_f)
    total_count_x_value_adj_s = tf.repeat(total_count_x_value_adj_f,
                                          num_terms_f)
    terms = (tf.cast(-1., dtype) ** term_idx_s
             * (1. / ((total_count_s + 1.) * tf.math.exp(
                 tfp_math.lbeta(total_count_s - term_idx_s + 1.,
                                term_idx_s + 1.))))
             * (total_count_x_value_adj_s - term_idx_s) ** (total_count_s - 1.))
    # Segment sum.
    segment_ids = tf.repeat(tf.range(tf.size(num_terms_f)), num_terms_f)
    pdf_s = tf.math.segment_sum(terms, segment_ids)
    # Reshape back.
    pdf = tf.reshape(pdf_s, shape)
    # Normalize.
    pdf = pdf * total_count_b / (
        range_ * tf.math.exp(tf.math.lgamma(total_count_b)))
    # Fix out-of-support queries.
    pdf = tf.where((value_centered < 0.) | (value_centered > 1.),
                   tf.cast(0., dtype), pdf)
    pdf = tf.where(tf.math.is_finite(value_centered), pdf, np.nan)
    return pdf


# TODO(b/157665707): Investigate alternative CDF formulas / computations.
def _bates_cdf(total_count, low, high, dtype, value):
  """Compute the Bates cdf.

  Internally, the (standard, unnormalized) cdf is computed by the formula

  ```none
  pdf = sum_{k=0}^j (-1)^k (n choose k) (nx - k)^n
  ```

  where
  * `n = total_count`,
  * `x = value` the value to compute the cumulative probability of, and
  * `j = floor(nx)`.

  This is shifted to `[low, high]` and normalized. Since the pdf is symmetric,
  we have `cdf(x) = 1 - cdf(1 - x)` for `x > .5`, hence we only compute the left
  half, which keeps the number of terms lower.

  Computation is batched, using `tf.math.segment_sum()`. For this reason this is
  not compatible with `tf.vectorized_map()`.

  All input parameters should have compatible dtypes and shapes.

  Args:
    total_count: `Tensor` with integer values, as given to the `Bates`
      constructor.
    low: Float `Tensor`, as given to the `Bates` constructor.
    high: Float `Tensor`, as given to the `Bates` constructor.
    dtype: The dtype of the output.
    value: Float `Tensor`. Input value to `cdf()`.
  Returns:
    cdf: Float `Tensor`. See above formula.
  """
  total_count = tf.cast(total_count, dtype)
  low = tf.convert_to_tensor(low)
  high = tf.convert_to_tensor(high)

  # Warn the user if they try to compute a pdf with high `total_count`.  This
  # warning is here instead of `_parameter_control_dependencies()` because
  # nested calls to `_name_and_control_scope` (e.g. `log_survival_function`) can
  # result in multiple warnings being added and multiple tensor
  # conversions. Also `sample()` does not have the same numerical issues.
  with tf.control_dependencies([_stability_limit_tensor(total_count, dtype)]):
    # Center and adjust `value` using limits and symmetry.
    value_centered = (value - low) / (high - low)
    value_adj = tf.clip_by_value(value_centered, 0., 1.)
    value_adj = tf.where(value_adj < .5, value_adj, 1. - value_adj)
    value_adj = tf.where(tf.math.is_finite(value_adj), value_adj, 0.)
    # Flatten to make segments; need to broadcast before flattening.
    shape = ps.broadcast_shape(ps.shape(value_adj), ps.shape(total_count))
    total_count_b = ps.broadcast_to(total_count, shape)
    total_count_x_value_adj_b = total_count * value_adj
    total_count_f = tf.reshape(total_count_b, [-1])
    total_count_x_value_adj_f = tf.reshape(total_count_x_value_adj_b, [-1])
    # Create segmented terms of summation.
    num_terms_f = tf.cast(tf.math.floor(total_count_x_value_adj_f + 1),
                          dtype=tf.int32)
    term_idx_s = tf.cast(_segmented_range(num_terms_f), dtype)  # aka `k`
    total_count_s = tf.repeat(total_count_f, num_terms_f)
    total_count_x_value_adj_s = tf.repeat(total_count_x_value_adj_f,
                                          num_terms_f)
    terms = (tf.cast(-1., dtype) ** term_idx_s
             * (1. / ((total_count_s + 1.) * tf.math.exp(
                 tfp_math.lbeta(total_count_s - term_idx_s + 1.,
                                term_idx_s + 1.))))
             * (total_count_x_value_adj_s - term_idx_s) ** total_count_s)
    # Segment sum.
    segment_ids = tf.repeat(tf.range(tf.size(num_terms_f)), num_terms_f)
    cdf_s = tf.math.segment_sum(terms, segment_ids)
    # Reshape back.
    cdf = tf.reshape(cdf_s, shape)
    # Normalize.
    cdf = cdf / tf.math.exp(tf.math.lgamma(total_count_b + tf.cast(1., dtype)))
    # cdf symmetry adjustment: cdf(x) = 1 - cdf(1 - x) for x > 0.5
    cdf = tf.where(value_centered > .5, 1. - cdf, cdf)
    # Fix out-of-support queries.
    cdf = tf.where(value_centered < 0., tf.cast(0., dtype), cdf)
    cdf = tf.where(value_centered > 1., tf.cast(1., dtype), cdf)
    cdf = tf.where(tf.math.is_finite(value_centered), cdf, np.nan)
    return cdf


def _stability_limit_tensor(total_count, dtype):
  limit = tf.cast(BATES_TOTAL_COUNT_STABILITY_LIMITS[dtype], dtype)
  return tf.cond(
      tf.math.reduce_any(total_count > limit),
      # pylint: disable=g-long-lambda
      lambda: tf.print(
          'WARNING: Bates PDF/CDF is unstable for `total_count` >', limit,
          output_stream=sys.stderr),
      tf.no_op)


def _segmented_range(limits):
  """Equivalent to `tf.ragged.range(limits).flat_values`.

  Ragged Tensors are are not supported by numpy.

  Args:
    limits: Integer `Tensor` of sizes of each range.

  Returns:
    segments: 1D `Tensor` of segment ranges.
  """
  return (tf.range(tf.reduce_sum(limits)) -
          tf.repeat(tf.concat([[0], tf.cumsum(limits[:-1])], axis=0), limits))


# TODO(b/157665707): Investigate rejection sampling for the Bates sampler.
def _sample_bates(total_count, low, high, n, seed=None):
  """Vectorized production of `Bates` samples.

  Args:
    total_count: (Batches of) counts of `Uniform`s to take means of.  Should
      have integer dtype and already be broadcasted to the batch shape.
    low: (Batches of) lower bounds of the `Uniform` variables to sample.  Should
      be the same floating dtype as `high` and broadcastable to the batch shape.
    high: (Batches of) upper bounds of the `Uniform` variables to sample. Should
      be the same floating dtype as `low` and broadcastable to the batch shape.
    n: `int32` number of samples to generate.
    seed: Random seed to pass to `Uniform` sampler.

  Returns:
    samples: Samples of (batches of) the `Bates` variable.  Will have same dtype
      as `low` and `high`. If the batch shape is `[B1,..., Bn]`, `samples` has
      shape `[n, B1,..., Bn]`.
  """

  # 1. Sample Uniform(0, 1)s, flattening the batch dimension into axis 0.
  uniform_sample_shape = ps.concat([[ps.reduce_sum(total_count)], [n]], axis=0)
  uniform_samples = samplers.uniform(
      uniform_sample_shape, minval=0., maxval=1., dtype=low.dtype, seed=seed)
  # 2. Produce segment means.
  segment_lengths = tf.reshape(total_count, [-1])
  segment_ids = tf.repeat(tf.range(tf.size(segment_lengths)), segment_lengths)
  flatmeans = tf.math.segment_mean(uniform_samples, segment_ids)
  # 3. Reshape and transpose segment means back to the original shape.
  outshape = tf.concat([tf.shape(total_count), [n]], axis=0)
  tmeans = tf.reshape(flatmeans, outshape)
  axes = tf.range(tf.rank(tmeans))
  means = tf.transpose(tmeans, tf.roll(axes, shift=1, axis=0))
  # 4. Shift/scale from (0, 1) to (low, high).
  return low + (high - low) * means
