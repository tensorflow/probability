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
"""Functions for computing moving statistics of a value stream."""

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps


__all__ = [
    'assign_log_moving_mean_exp',
    'assign_moving_mean_variance',
    'moving_mean_variance_zero_debiased',
]


def assign_moving_mean_variance(value, moving_mean, moving_variance=None,
                                zero_debias_count=None, decay=0.99, axis=(),
                                name=None):
  """Compute one update to the exponentially weighted moving mean and variance.

  The `value` updated exponentially weighted moving `moving_mean` and
  `moving_variance` are conceptually given by the following recurrence
  relations ([Welford (1962)][1]):

  ```python
  new_mean = old_mean + (1 - decay) * (value - old_mean)
  new_var  = old_var  + (1 - decay) * (value - old_mean) * (value - new_mean)
  ```

  This function implements the above recurrences in a numerically stable manner
  and also uses the `assign_add` op to allow concurrent lockless updates to the
  supplied variables.

  For additional references see
  [John D. Cook's Blog](https://www.johndcook.com/blog/standard_deviation),
  whereas we use `1 - decay = 1 / k`, and
  [Finch (2009; Eq.  143)][2], whereas we use `1 - decay = alpha`.

  Since variables that are initialized to a `0` value will be `0` biased,
  providing `zero_debias_count` triggers scaling the `moving_mean` and
  `moving_variance` by the factor of `1 - decay ** (zero_debias_count + 1)`.
  For more details, see `tfp.stats.moving_mean_variance_zero_debiased`.

  Args:
    value: `float`-like `Tensor` representing one or more streaming
      observations. When `axis` is non-empty `value ` is reduced (by mean) for
      updated `moving_mean` and `moving-variance`. Presumed to have same shape
      as `moving_mean` and `moving_variance`.
    moving_mean: `float`-like `tf.Variable` representing the exponentially
      weighted moving mean. Same shape as `moving_variance` and `value`. This
      function presumes the `tf.Variable` was created with all zero initial
      value(s).
    moving_variance: `float`-like `tf.Variable` representing the exponentially
      weighted moving variance. Same shape as `moving_mean` and `value`.  This
      function presumes the `tf.Variable` was created with all zero initial
      value(s).
      Default value: `None` (i.e., no moving variance is computed).
    zero_debias_count: `int`-like `tf.Variable` representing the number of times
      this function has been called on streaming input (*not* the number of
      reduced values used in this functions computation). When not `None` (the
      default) the returned values for `moving_mean` and `moving_variance` are
      "zero debiased", i.e., corrected for their presumed all zeros
      intialization. Note: the `tf.Variable`s `moving_mean` and
      `moving_variance` *always* store the unbiased calculation, regardless of
      setting this argument. To obtain unbiased calculations from these
      `tf.Variable`s, see `tfp.stats.moving_mean_variance_zero_debiased`.
      Default value: `None` (i.e., no zero debiasing calculation is made).
    decay: A `float`-like `Tensor` representing the moving mean decay. Typically
      close to `1.`, e.g., `0.99`.
      Default value: `0.99`.
    axis: The dimensions to reduce. If `()` (the default) no dimensions are
      reduced. If `None` all dimensions are reduced. Must be in the range
      `[-rank(value), rank(value))`.
      Default value: `()` (i.e., no reduction is made).
    name: Python `str` prepended to op names created by this function.
      Default value: `None` (i.e., 'assign_moving_mean_variance').

  Returns:
    moving_mean: The `value`-updated exponentially weighted moving mean.
      Debiased if `zero_debias_count is not None`.
    moving_variance: The `value`-updated exponentially weighted moving variance.
      Debiased if `zero_debias_count is not None`.

  Raises:
    TypeError: if `moving_mean` does not have float type `dtype`.
    TypeError: if `moving_mean`, `moving_variance`, `value`, `decay` have
      different `base_dtype`.

  #### Examples

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp
  tfd = tfp.distributions
  d = tfd.MultivariateNormalTriL(
      loc=[-1., 1.],
      scale_tril=tf.linalg.cholesky([[0.75, 0.05],
                                     [0.05, 0.5]]))
  d.mean()
  # ==> [-1.,  1.]
  d.variance()
  # ==> [0.75, 0.5]
  moving_mean = tf.Variable(tf.zeros(2))
  moving_variance = tf.Variable(tf.zeros(2))
  zero_debias_count = tf.Variable(0)
  for _ in range(100):
    m, v = tfp.stats.assign_moving_mean_variance(
      value=d.sample(3),
      moving_mean=moving_mean,
      moving_variance=moving_variance,
      zero_debias_count=zero_debias_count,
      decay=0.99,
      axis=-2)
    print(m.numpy(), v.numpy())
  # ==> [-1.0334632  0.9545268] [0.8126194 0.5118788]
  # ==> [-1.0293456   0.96070296] [0.8115873  0.50947404]
  # ...
  # ==> [-1.025172  0.96351 ] [0.7142789  0.48570773]

  m1, v1 = tfp.stats.moving_mean_variance_zero_debiased(
    moving_mean,
    moving_variance,
    zero_debias_count,
    decay=0.99)
  print(m.numpy(), v.numpy())
  # ==> [-1.025172  0.96351 ] [0.7142789  0.48570773]
  assert(all(m == m1))
  assert(all(v == v1))
  ```

  #### References

  [1]  B. P. Welford. Note on a Method for Calculating Corrected Sums of
       Squares and Products. Technometrics, Vol. 4, No. 3 (Aug., 1962), p419-20.
       http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.302.7503&rep=rep1&type=pdf
       http://www.jstor.org/stable/1266577

  [2]: Tony Finch. Incremental calculation of weighted mean and variance.
       _Technical Report_, 2009.
       http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf
  """
  with tf.name_scope(name or 'assign_moving_mean_variance'):
    base_dtype = dtype_util.base_dtype(moving_mean.dtype)
    if not dtype_util.is_floating(base_dtype):
      raise TypeError(
          'Argument `moving_mean` is not float type (saw {}).'.format(
              dtype_util.name(moving_mean.dtype)))

    value = tf.convert_to_tensor(value, dtype=base_dtype, name='value')
    decay = tf.convert_to_tensor(decay, dtype=base_dtype, name='decay')
    # Force a read of `moving_mean` as  we'll need it twice.
    old_mean = tf.convert_to_tensor(
        moving_mean, dtype=base_dtype, name='old_mean')

    updated_mean = moving_mean.assign_add(
        (1. - decay) * (tf.reduce_mean(value, axis=axis) - old_mean))

    if zero_debias_count is not None:
      t = tf.cast(zero_debias_count.assign_add(1), base_dtype)
      # Could have used:
      #   bias_correction = -tf.math.expm1(t * tf.math.log(decay))
      # however since we expect decay to be nearly 1, we don't expect this to
      # bear a significant improvement, yet would incur higher computational
      # cost.
      bias_correction = 1. - decay**t
      with tf.control_dependencies([updated_mean]):
        updated_mean = updated_mean / bias_correction

    if moving_variance is None:
      return updated_mean

    if base_dtype != dtype_util.base_dtype(moving_variance.dtype):
      raise TypeError('Arguments `moving_mean` and `moving_variance` do not '
                      'have same base `dtype` (saw {}, {}).'.format(
                          dtype_util.name(moving_mean.dtype),
                          dtype_util.name(moving_variance.dtype)))

    if zero_debias_count is not None:
      old_t = tf.where(t > 1., t - 1., tf.constant(np.inf, base_dtype))
      old_bias_correction = 1. - decay**old_t
      old_mean = old_mean / old_bias_correction

    mean_sq_diff = tf.reduce_mean(
        tf.math.squared_difference(value, old_mean),
        axis=axis)
    updated_variance = moving_variance.assign_add(
        (1. - decay) * (decay * mean_sq_diff - moving_variance))

    if zero_debias_count is not None:
      with tf.control_dependencies([updated_variance]):
        updated_variance = updated_variance / bias_correction

    return updated_mean, updated_variance


def moving_mean_variance_zero_debiased(moving_mean, moving_variance=None,
                                       zero_debias_count=None, decay=0.99,
                                       name=None):
  """Compute zero debiased versions of `moving_mean` and `moving_variance`.

  Since `moving_*` variables initialized with `0`s will be biased (toward `0`),
  this function rescales the `moving_mean` and `moving_variance` by the factor
  `1 - decay**zero_debias_count`, i.e., such that the `moving_mean` is unbiased.
  For more details, see [Kingma (2014)][1].

  Args:
    moving_mean: `float`-like `tf.Variable` representing the exponentially
      weighted moving mean. Same shape as `moving_variance` and `value`. This
      function presumes the `tf.Variable` was created with all zero initial
      value(s).
    moving_variance: `float`-like `tf.Variable` representing the exponentially
      weighted moving variance. Same shape as `moving_mean` and `value`.  This
      function presumes the `tf.Variable` was created with all zero initial
      value(s).
      Default value: `None` (i.e., no moving variance is computed).
    zero_debias_count: `int`-like `tf.Variable` representing the number of times
      this function has been called on streaming input (*not* the number of
      reduced values used in this functions computation). When not `None` (the
      default) the returned values for `moving_mean` and `moving_variance` are
      "zero debiased", i.e., corrected for their presumed all zeros
      intialization. Note: the `tf.Variable`s `moving_mean` and
      `moving_variance` *always* store the unbiased calculation, regardless of
      setting this argument. To obtain unbiased calculations from these
      `tf.Variable`s, see `tfp.stats.moving_mean_variance_zero_debiased`.
      Default value: `None` (i.e., no zero debiasing calculation is made).
    decay: A `float`-like `Tensor` representing the moving mean decay. Typically
      close to `1.`, e.g., `0.99`.
      Default value: `0.99`.
    name: Python `str` prepended to op names created by this function.
      Default value: `None` (i.e., 'moving_mean_variance_zero_debiased').

  Returns:
    moving_mean: The zero debiased exponentially weighted moving mean.
    moving_variance: The zero debiased exponentially weighted moving variance.

  Raises:
    TypeError: if `moving_mean` does not have float type `dtype`.
    TypeError: if `moving_mean`, `moving_variance`, `decay` have different
      `base_dtype`.

  #### References

  [1]: Diederik P. Kingma, Jimmy Ba. Adam: A Method for Stochastic Optimization.
        _arXiv preprint arXiv:1412.6980_, 2014.
       https://arxiv.org/abs/1412.6980
  """
  with tf.name_scope(name or 'zero_debias_count'):
    if zero_debias_count is None:
      raise ValueError()
    base_dtype = dtype_util.base_dtype(moving_mean.dtype)
    if not dtype_util.is_floating(base_dtype):
      raise TypeError(
          'Argument `moving_mean` is not float type (saw {}).'.format(
              dtype_util.name(moving_mean.dtype)))
    t = tf.cast(zero_debias_count, dtype=base_dtype)
    # Could have used:
    #   bias_correction = -tf.math.expm1(t * tf.math.log(decay))
    # however since we expect decay to be nearly 1, we don't expect this to bear
    # a significant improvement, yet would incur higher computational cost.
    t = tf.where(t > 0., t, tf.constant(np.inf, base_dtype))
    bias_correction = 1. - decay**t
    unbiased_mean = moving_mean / bias_correction
    if moving_variance is None:
      return unbiased_mean
    if base_dtype != dtype_util.base_dtype(moving_variance.dtype):
      raise TypeError('Arguments `moving_mean` and `moving_variance` do not '
                      'have same base `dtype` (saw {}, {}).'.format(
                          dtype_util.name(moving_mean.dtype),
                          dtype_util.name(moving_variance.dtype)))
    unbiased_variance = moving_variance / bias_correction
    return unbiased_mean, unbiased_variance


def assign_log_moving_mean_exp(log_value, moving_log_mean_exp,
                               zero_debias_count=None, decay=0.99, name=None):
  """Compute the log of the exponentially weighted moving mean of the exp.

  If `log_value` is a draw from a stationary random variable, this function
  approximates `log(E[exp(log_value)])`, i.e., a weighted log-sum-exp. More
  precisely, a `tf.Variable`, `moving_log_mean_exp`, is updated by `log_value`
  using the following identity:

  ```none
  moving_log_mean_exp =
  = log(decay exp(moving_log_mean_exp) + (1 - decay) exp(log_value))
  = log(exp(moving_log_mean_exp + log(decay)) + exp(log_value + log1p(-decay)))
  = moving_log_mean_exp
    + log(  exp(moving_log_mean_exp   - moving_log_mean_exp + log(decay))
          + exp(log_value - moving_log_mean_exp + log1p(-decay)))
  = moving_log_mean_exp
    + log_sum_exp([log(decay), log_value - moving_log_mean_exp +
    log1p(-decay)]).
  ```

  In addition to numerical stability, this formulation is advantageous because
  `moving_log_mean_exp` can be updated in a lock-free manner, i.e., using
  `assign_add`. (Note: the updates are not thread-safe; it's just that the
  update to the tf.Variable is presumed efficient due to being lock-free.)

  Args:
    log_value: `float`-like `Tensor` representing a new (streaming) observation.
      Same shape as `moving_log_mean_exp`.
    moving_log_mean_exp: `float`-like `Variable` representing the log of the
      exponentially weighted moving mean of the exp. Same shape as `log_value`.
    zero_debias_count: `int`-like `tf.Variable` representing the number of times
      this function has been called on streaming input (*not* the number of
      reduced values used in this functions computation). When not `None` (the
      default) the returned values for `moving_mean` and `moving_variance` are
      "zero debiased", i.e., corrected for their presumed all zeros
      intialization. Note: the `tf.Variable`s `moving_mean` and
      `moving_variance` *always* store the unbiased calculation, regardless of
      setting this argument. To obtain unbiased calculations from these
      `tf.Variable`s, see `tfp.stats.moving_mean_variance_zero_debiased`.
      Default value: `None` (i.e., no zero debiasing calculation is made).
    decay: A `float`-like `Tensor` representing the moving mean decay. Typically
      close to `1.`, e.g., `0.99`.
      Default value: `0.99`.
    name: Python `str` prepended to op names created by this function.
      Default value: `None` (i.e., 'assign_log_moving_mean_exp').

  Returns:
    moving_log_mean_exp: A reference to the input 'Variable' tensor with the
      `log_value`-updated log of the exponentially weighted moving mean of exp.

  Raises:
    TypeError: if `moving_log_mean_exp` does not have float type `dtype`.
    TypeError: if `moving_log_mean_exp`, `log_value`, `decay` have different
      `base_dtype`.
  """
  if zero_debias_count is not None:
    raise NotImplementedError(
        'Argument `zero_debias_count` is not yet supported. If you require '
        'this feature please create a new issue on '
        '`https://github.com/tensorflow/probability` or email '
        '`tfprobability@tensorflow.org`.')
  with tf.name_scope(name or 'assign_log_moving_mean_exp'):
    # We want to update the variable in a numerically stable and lock-free way.
    # To do this, observe that variable `x` updated by `v` is:
    # x = log(w exp(x) + (1-w) exp(v))
    #   = log(exp(x + log(w)) + exp(v + log1p(-w)))
    #   = x + log(exp(x - x + log(w)) + exp(v - x + log1p(-w)))
    #   = x + lse([log(w), v - x + log1p(-w)])
    base_dtype = dtype_util.base_dtype(moving_log_mean_exp.dtype)
    if not dtype_util.is_floating(base_dtype):
      raise TypeError(
          'Argument `moving_log_mean_exp` is not float type (saw {}).'.format(
              dtype_util.name(moving_log_mean_exp.dtype)))
    log_value = tf.convert_to_tensor(
        log_value, dtype=base_dtype, name='log_value')
    decay = tf.convert_to_tensor(decay, dtype=base_dtype, name='decay')
    delta = (log_value - moving_log_mean_exp)[tf.newaxis, ...]
    x = tf.concat([
        tf.broadcast_to(
            tf.math.log(decay),
            ps.broadcast_shape(ps.shape(decay), ps.shape(delta))),
        delta + tf.math.log1p(-decay)
    ], axis=0)
    update = tf.reduce_logsumexp(x, axis=0)
    return moving_log_mean_exp.assign_add(update)
