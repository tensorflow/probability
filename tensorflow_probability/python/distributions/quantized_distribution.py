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
"""Quantized distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import distribution as distributions
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util


__all__ = ['QuantizedDistribution']


_prob_base_note = """
For whole numbers `y`,

```
P[Y = y] := P[X <= low],  if y == low,
         := P[X > high - 1],  y == high,
         := 0, if j < low or y > high,
         := P[y - 1 < X <= y],  all other y.
```

"""

_prob_note = _prob_base_note + """
The base distribution's `cdf` method must be defined on `y - 1`. If the
base distribution has a `survival_function` method, results will be more
accurate for large values of `y`, and in this case the `survival_function` must
also be defined on `y - 1`.
"""

_log_prob_note = _prob_base_note + """
The base distribution's `log_cdf` method must be defined on `y - 1`. If the
base distribution has a `log_survival_function` method results will be more
accurate for large values of `y`, and in this case the `log_survival_function`
must also be defined on `y - 1`.
"""


_cdf_base_note = """

For whole numbers `y`,

```
cdf(y) := P[Y <= y]
        = 1, if y >= high,
        = 0, if y < low,
        = P[X <= y], otherwise.
```

Since `Y` only has mass at whole numbers, `P[Y <= y] = P[Y <= floor(y)]`.
This dictates that fractional `y` are first floored to a whole number, and
then above definition applies.
"""

_cdf_note = _cdf_base_note + """
The base distribution's `cdf` method must be defined on `y - 1`.
"""

_log_cdf_note = _cdf_base_note + """
The base distribution's `log_cdf` method must be defined on `y - 1`.
"""


_sf_base_note = """

For whole numbers `y`,

```
survival_function(y) := P[Y > y]
                      = 0, if y >= high,
                      = 1, if y < low,
                      = P[X <= y], otherwise.
```

Since `Y` only has mass at whole numbers, `P[Y <= y] = P[Y <= floor(y)]`.
This dictates that fractional `y` are first floored to a whole number, and
then above definition applies.
"""

_sf_note = _sf_base_note + """
The base distribution's `cdf` method must be defined on `y - 1`.
"""

_log_sf_note = _sf_base_note + """
The base distribution's `log_cdf` method must be defined on `y - 1`.
"""


class QuantizedDistribution(distributions.Distribution):
  """Distribution representing the quantization `Y = ceiling(X)`.

  #### Definition in Terms of Sampling

  ```
  1. Draw X
  2. Set Y <-- ceiling(X)
  3. If Y < low, reset Y <-- low
  4. If Y > high, reset Y <-- high
  5. Return Y
  ```

  #### Definition in Terms of the Probability Mass Function

  Given scalar random variable `X`, we define a discrete random variable `Y`
  supported on the integers as follows:

  ```
  P[Y = j] := P[X <= low],  if j == low,
           := P[X > high - 1],  j == high,
           := 0, if j < low or j > high,
           := P[j - 1 < X <= j],  all other j.
  ```

  Conceptually, without cutoffs, the quantization process partitions the real
  line `R` into half open intervals, and identifies an integer `j` with the
  right endpoints:

  ```
  R = ... (-2, -1](-1, 0](0, 1](1, 2](2, 3](3, 4] ...
  j = ...      -1      0     1     2     3     4  ...
  ```

  `P[Y = j]` is the mass of `X` within the `jth` interval.
  If `low = 0`, and `high = 2`, then the intervals are redrawn
  and `j` is re-assigned:

  ```
  R = (-infty, 0](0, 1](1, infty)
  j =          0     1     2
  ```

  `P[Y = j]` is still the mass of `X` within the `jth` interval.

  #### Examples

  We illustrate a mixture of discretized logistic distributions
  [(Salimans et al., 2017)][1]. This is used, for example, for capturing 16-bit
  audio in WaveNet [(van den Oord et al., 2017)][2]. The values range in
  a 1-D integer domain of `[0, 2**16-1]`, and the discretization captures
  `P(x - 0.5 < X <= x + 0.5)` for all `x` in the domain excluding the endpoints.
  The lowest value has probability `P(X <= 0.5)` and the highest value has
  probability `P(2**16 - 1.5 < X)`.

  Below we assume a `wavenet` function. It takes as `input` right-shifted audio
  samples of shape `[..., sequence_length]`. It returns a real-valued tensor of
  shape `[..., num_mixtures * 3]`, i.e., each mixture component has a `loc` and
  `scale` parameter belonging to the logistic distribution, and a `logits`
  parameter determining the unnormalized probability of that component.

  ```python
  tfd = tfp.distributions
  tfb = tfp.bijectors

  net = wavenet(inputs)
  loc, unconstrained_scale, logits = tf.split(net,
                                              num_or_size_splits=3,
                                              axis=-1)
  scale = tf.math.softplus(unconstrained_scale)

  # Form mixture of discretized logistic distributions. Note we shift the
  # logistic distribution by -0.5. This lets the quantization capture 'rounding'
  # intervals, `(x-0.5, x+0.5]`, and not 'ceiling' intervals, `(x-1, x]`.
  discretized_logistic_dist = tfd.QuantizedDistribution(
      distribution=tfd.TransformedDistribution(
          distribution=tfd.Logistic(loc=loc, scale=scale),
          bijector=tfb.AffineScalar(shift=-0.5)),
      low=0.,
      high=2**16 - 1.)
  mixture_dist = tfd.MixtureSameFamily(
      mixture_distribution=tfd.Categorical(logits=logits),
      components_distribution=discretized_logistic_dist)

  neg_log_likelihood = -tf.reduce_sum(mixture_dist.log_prob(targets))
  train_op = tf.train.AdamOptimizer().minimize(neg_log_likelihood)
  ```

  After instantiating `mixture_dist`, we illustrate maximum likelihood by
  calculating its log-probability of audio samples as `target` and optimizing.

  #### References

  [1]: Tim Salimans, Andrej Karpathy, Xi Chen, and Diederik P. Kingma.
       PixelCNN++: Improving the PixelCNN with discretized logistic mixture
       likelihood and other modifications.
       _International Conference on Learning Representations_, 2017.
       https://arxiv.org/abs/1701.05517
  [2]: Aaron van den Oord et al. Parallel WaveNet: Fast High-Fidelity Speech
       Synthesis. _arXiv preprint arXiv:1711.10433_, 2017.
       https://arxiv.org/abs/1711.10433
  """

  def __init__(self,
               distribution,
               low=None,
               high=None,
               validate_args=False,
               name='QuantizedDistribution'):
    """Construct a Quantized Distribution representing `Y = ceiling(X)`.

    Some properties are inherited from the distribution defining `X`. Example:
    `allow_nan_stats` is determined for this `QuantizedDistribution` by reading
    the `distribution`.

    Args:
      distribution:  The base distribution class to transform. Typically an
        instance of `Distribution`.
      low: `Tensor` with same `dtype` as this distribution and shape
        that broadcasts to that of samples but does not result in additional
        batch dimensions after broadcasting. Should be a whole number. Default
        `None`. If provided, base distribution's `prob` should be defined at
        `low`.
      high: `Tensor` with same `dtype` as this distribution and shape
        that broadcasts to that of samples but does not result in additional
        batch dimensions after broadcasting. Should be a whole number. Default
        `None`. If provided, base distribution's `prob` should be defined at
        `high - 1`. `high` must be strictly greater than `low`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: If `dist_cls` is not a subclass of
          `Distribution` or continuous.
      NotImplementedError:  If the base distribution does not implement `cdf`.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([distribution, high, low],
                                      dtype_hint=tf.float32)
      self._dist = distribution
      self._low = tensor_util.convert_nonref_to_tensor(
          low, name='low', dtype=dtype)
      self._high = tensor_util.convert_nonref_to_tensor(
          high, name='high', dtype=dtype)
      super(QuantizedDistribution, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=self._dist.allow_nan_stats,
          parameters=parameters,
          name=name)

  @property
  def distribution(self):
    """Base distribution, p(x)."""
    return self._dist

  @property
  def low(self):
    """Lowest value that quantization returns."""
    return self._low

  @property
  def high(self):
    """Highest value that quantization returns."""
    return self._high

  def _batch_shape_tensor(self):
    return self.distribution.batch_shape_tensor()

  def _batch_shape(self):
    return self.distribution.batch_shape

  def _event_shape_tensor(self):
    return self.distribution.event_shape_tensor()

  def _event_shape(self):
    return self.distribution.event_shape

  def _sample_n(self, n, seed=None):
    with tf.name_scope('transform'):
      x_samps = self.distribution.sample(n, seed=seed)

      # Snap values to the intervals (j - 1, j].
      result_so_far = tf.math.ceil(x_samps)

      if self._low is not None:
        low = tf.convert_to_tensor(self._low)
        result_so_far = tf.where(result_so_far < low, low, result_so_far)

      if self._high is not None:
        high = tf.convert_to_tensor(self._high)
        result_so_far = tf.where(result_so_far > high, high, result_so_far)

      return result_so_far

  @distribution_util.AppendDocstring(_log_prob_note)
  def _log_prob(self, y):
    # Changes of mass are only at the integers, so we must use tf.floor in our
    # computation of log_cdf/log_sf.  Floor now, since
    # tf.floor(y - 1) can incur unwanted rounding near powers of two, but
    # tf.floor(y) - 1 can't.
    y = tf.floor(y)

    if not (hasattr(self.distribution, '_log_cdf') or
            hasattr(self.distribution, '_cdf')):
      raise NotImplementedError(
          '`log_prob` not implemented unless the base distribution implements '
          '`log_cdf`')
    try:
      return self._log_prob_with_logsf_and_logcdf(y)
    except NotImplementedError:
      return self._log_prob_with_logcdf(y)

  def _log_prob_with_logcdf(self, y):
    low = None if self._low is None else tf.convert_to_tensor(self._low)
    high = None if self._high is None else tf.convert_to_tensor(self._high)
    return _logsum_expbig_minus_expsmall(
        self.log_cdf(y, low=low, high=high),
        self.log_cdf(y - 1., low=low, high=high))

  def _log_prob_with_logsf_and_logcdf(self, y):
    """Compute log_prob(y) using log survival_function and cdf together."""
    # There are two options that would be equal if we had infinite precision:
    # Log[ sf(y - 1) - sf(y) ]
    #   = Log[ exp{logsf(y - 1)} - exp{logsf(y)} ]
    # Log[ cdf(y) - cdf(y - 1) ]
    #   = Log[ exp{logcdf(y)} - exp{logcdf(y - 1)} ]
    low = None if self._low is None else tf.convert_to_tensor(self._low)
    high = None if self._high is None else tf.convert_to_tensor(self._high)
    logsf_y = self._log_survival_function(y, low=low, high=high)
    logsf_y_minus_1 = self._log_survival_function(y - 1., low=low, high=high)
    logcdf_y = self._log_cdf(y, low=low, high=high)
    logcdf_y_minus_1 = self._log_cdf(y - 1., low=low, high=high)

    # Important:  Here we use select in a way such that no input is inf, this
    # prevents the troublesome case where the output of select can be finite,
    # but the output of grad(select) will be NaN.

    # In either case, we are doing Log[ exp{big} - exp{small} ]
    # We want to use the sf items precisely when we are on the right side of the
    # median, which occurs when logsf_y < logcdf_y.
    big = tf.where(logsf_y < logcdf_y, logsf_y_minus_1, logcdf_y)
    small = tf.where(logsf_y < logcdf_y, logsf_y, logcdf_y_minus_1)

    return _logsum_expbig_minus_expsmall(big, small)

  @distribution_util.AppendDocstring(_prob_note)
  def _prob(self, y):
    # Changes of mass are only at the integers, so we must use tf.floor in our
    # computation of log_cdf/log_sf.  Floor now, since
    # tf.floor(y - 1) can incur unwanted rounding near powers of two, but
    # tf.floor(y) - 1 can't.
    y = tf.floor(y)

    if not hasattr(self.distribution, '_cdf'):
      raise NotImplementedError(
          '`prob` not implemented unless the base distribution implements '
          '`cdf`')
    try:
      return self._prob_with_sf_and_cdf(y)
    except NotImplementedError:
      return self._prob_with_cdf(y)

  def _prob_with_cdf(self, y):
    low = None if self._low is None else tf.convert_to_tensor(self._low)
    high = None if self._high is None else tf.convert_to_tensor(self._high)
    return (self._cdf(y, low=low, high=high) -
            self._cdf(y - 1., low=low, high=high))

  def _prob_with_sf_and_cdf(self, y):
    # There are two options that would be equal if we had infinite precision:
    # sf(y - 1.) - sf(y)
    # cdf(y) - cdf(y - 1.)
    low = None if self._low is None else tf.convert_to_tensor(self._low)
    high = None if self._high is None else tf.convert_to_tensor(self._high)
    sf_y = self._survival_function(y, low=low, high=high)
    sf_y_minus_1 = self._survival_function(y - 1., low=low, high=high)
    cdf_y = self._cdf(y, low=low, high=high)
    cdf_y_minus_1 = self._cdf(y - 1., low=low, high=high)

    # sf_prob has greater precision iff we're on the right side of the median.
    return tf.where(
        sf_y < cdf_y,  # True iff we're on the right side of the median.
        sf_y_minus_1 - sf_y,
        cdf_y - cdf_y_minus_1)

  @distribution_util.AppendDocstring(_log_cdf_note)
  def _log_cdf(self, y, low=None, high=None):
    low = self._low if low is None else low
    high = self._high if high is None else high
    # Recall the promise:
    # cdf(y) := P[Y <= y]
    #         = 1, if y >= high,
    #         = 0, if y < low,
    #         = P[X <= y], otherwise.

    # P[Y <= j] = P[floor(Y) <= j] since mass is only at integers, not in
    # between.
    j = tf.floor(y)

    result_so_far = self.distribution.log_cdf(j)

    # Re-define values at the cutoffs.
    if low is not None:
      result_so_far = tf.where(
          j < low, tf.constant(-np.inf, self.dtype), result_so_far)

    if high is not None:
      result_so_far = tf.where(
          j < high, result_so_far, tf.zeros([], self.dtype))

    return result_so_far

  @distribution_util.AppendDocstring(_cdf_note)
  def _cdf(self, y, low=None, high=None):
    low = self._low if low is None else low
    high = self._high if high is None else high

    # Recall the promise:
    # cdf(y) := P[Y <= y]
    #         = 1, if y >= high,
    #         = 0, if y < low,
    #         = P[X <= y], otherwise.

    # P[Y <= j] = P[floor(Y) <= j] since mass is only at integers, not in
    # between.
    j = tf.floor(y)

    # P[X <= j], used when low < X < high.
    result_so_far = self.distribution.cdf(j)

    # Re-define values at the cutoffs.
    if low is not None:
      result_so_far = tf.where(
          j < low, tf.zeros([], self.dtype), result_so_far)

    if high is not None:
      result_so_far = tf.where(
          j < high, result_so_far, tf.ones([], self.dtype))

    return result_so_far

  @distribution_util.AppendDocstring(_log_sf_note)
  def _log_survival_function(self, y, low=None, high=None):
    low = self._low if low is None else low
    high = self._high if high is None else high

    # Recall the promise:
    # survival_function(y) := P[Y > y]
    #                       = 0, if y >= high,
    #                       = 1, if y < low,
    #                       = P[X > y], otherwise.

    # P[Y > j] = P[ceiling(Y) > j] since mass is only at integers, not in
    # between.
    j = tf.math.ceil(y)

    # P[X > j], used when low < X < high.
    result_so_far = self.distribution.log_survival_function(j)

    # Re-define values at the cutoffs.
    if low is not None:
      result_so_far = tf.where(
          j < low, tf.zeros([], self.dtype), result_so_far)

    if high is not None:
      result_so_far = tf.where(
          j < high, result_so_far, tf.constant(-np.inf, self.dtype))

    return result_so_far

  @distribution_util.AppendDocstring(_sf_note)
  def _survival_function(self, y, low=None, high=None):
    low = self._low if low is None else low
    high = self._high if high is None else high

    # Recall the promise:
    # survival_function(y) := P[Y > y]
    #                       = 0, if y >= high,
    #                       = 1, if y < low,
    #                       = P[X > y], otherwise.

    # P[Y > j] = P[ceiling(Y) > j] since mass is only at integers, not in
    # between.
    j = tf.math.ceil(y)

    # P[X > j], used when low < X < high.
    result_so_far = self.distribution.survival_function(j)

    # Re-define values at the cutoffs.
    if low is not None:
      result_so_far = tf.where(
          j < low, tf.ones([], self.dtype), result_so_far)

    if high is not None:
      result_so_far = tf.where(
          j < high, result_so_far, tf.zeros([], self.dtype))

    return result_so_far

  def _default_event_space_bijector(self):
    return

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []

    sample_shape = tf.concat(
        [self._batch_shape_tensor(), self._event_shape_tensor()], axis=0)

    low = None if self._low is None else tf.convert_to_tensor(self._low)
    high = None if self._high is None else tf.convert_to_tensor(self._high)

    assertions = []
    if self._low is not None and is_init != tensor_util.is_ref(self._low):
      low_shape = ps.shape(low)
      broadcast_shape = ps.broadcast_shape(sample_shape, low_shape)
      assertions.extend(
          [distribution_util.assert_integer_form(
              low, message='`low` has non-integer components.'),
           assert_util.assert_equal(
               tf.reduce_prod(broadcast_shape),
               tf.reduce_prod(sample_shape),
               message=('Shape of `low` adds extra batch dimensions to '
                        'sample shape.'))])
    if self._high is not None and is_init != tensor_util.is_ref(self._high):
      high_shape = ps.shape(high)
      broadcast_shape = ps.broadcast_shape(sample_shape, high_shape)
      assertions.extend(
          [distribution_util.assert_integer_form(
              high, message='`high` has non-integer components.'),
           assert_util.assert_equal(
               tf.reduce_prod(broadcast_shape),
               tf.reduce_prod(sample_shape),
               message=('Shape of `high` adds extra batch dimensions to '
                        'sample shape.'))])
    if (self._low is not None and self._high is not None and
        (is_init != (tensor_util.is_ref(self._low)
                     or tensor_util.is_ref(self._high)))):
      assertions.append(assert_util.assert_less(
          low, high,
          message='`low` must be strictly less than `high`.'))

    return assertions

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(distribution_util.assert_integer_form(
        x, message='Sample has non-integer components.'))
    return assertions

  _composite_tensor_nonshape_params = ('distribution', 'low', 'high')


def _logsum_expbig_minus_expsmall(big, small):
  """Stable evaluation of `Log[exp{big} - exp{small}]`.

  To work correctly, we should have the pointwise relation:  `small <= big`.

  Args:
    big: Floating-point `Tensor`
    small: Floating-point `Tensor` with same `dtype` as `big` and broadcastable
      shape.

  Returns:
    log_sub_exp: `Tensor` of same `dtype` of `big` and broadcast shape.
  """
  with tf.name_scope('logsum_expbig_minus_expsmall'):
    return big + tf.math.log1p(-tf.exp(small - big))
