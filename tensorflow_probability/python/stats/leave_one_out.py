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
"""Functions for generic calculations.

Note: Many of these functions will eventually be migrated to core TensorFlow.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math.generic import log_add_exp
from tensorflow_probability.python.math.generic import softplus_inverse


__all__ = [
    'log_loomean_exp',
    'log_loosum_exp',
    'log_soomean_exp',
    'log_soosum_exp',
]


def log_loosum_exp(logx, axis=None, keepdims=False, name=None):
  """Computes the log-leave-one-out-sum of `exp(logx)`.

  Args:
    logx: Floating-type `Tensor` representing `log(x)` where `x` is some
      positive value.
    axis: The dimensions to sum across. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(logx), rank(logx)]`.
      Default value: `None` (i.e., reduce over all dims).
    keepdims: If true, retains reduced dimensions with length 1.
      Default value: `False` (i.e., keep all dims in `log_sum_x`).
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., `"log_loosum_exp"`).

  Returns:
    log_loosum_exp: `Tensor` with same shape and dtype as `logx` representing
      the natural-log of the sum of `exp(logx)` except that the element
      `logx[i]` is removed.
    log_sum_x: `logx.dtype` `Tensor` corresponding to the natural-log of the
      sum of `exp(logx)`. Has reduced shape of `logx` (per `axis` and
      `keepdims`).
  """
  with tf.name_scope(name or 'log_loosum_exp'):
    return _log_loosum_exp_impl(logx, axis, keepdims, compute_mean=False)[:2]


def log_loomean_exp(logx, axis, keepdims=False, name=None):
  """Computes the log-leave-one-out-mean of `exp(logx)`.

  Args:
    logx: Floating-type `Tensor` representing `log(x)` where `x` is some
      positive value.
    axis: The dimensions to sum across. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(logx), rank(logx)]`.
      Default value: `None` (i.e., reduce over all dims).
    keepdims: If true, retains reduced dimensions with length 1.
      Default value: `False` (i.e., keep all dims in `log_mean_x`).
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., `"log_loomean_exp"`).

  Returns:
    log_loomean_exp: `Tensor` with same shape and dtype as `logx` representing
      the natural-log of the mean of `exp(logx)` except that the element
      `logx[i]` is removed.
    log_mean_x: `logx.dtype` `Tensor` corresponding to the natural-log of the
      arithmetic mean of `x`. Has reduced shape of `logx` (per `axis` and
      `keepdims`).
  """
  with tf.name_scope(name or 'log_loomean_exp'):
    return _log_loosum_exp_impl(logx, axis, keepdims, compute_mean=True)[:2]


def log_soosum_exp(logx, axis, keepdims=False, name=None):
  """Computes the log-swap-one-out-sum of `exp(logx)`.

  The swapped out element `logx[i]` is replaced with the log-leave-`i`-out
  geometric mean of `logx`.

  Args:
    logx: Floating-type `Tensor` representing `log(x)` where `x` is some
      positive value.
    axis: The dimensions to sum across. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(logx), rank(logx)]`.
      Default value: `None` (i.e., reduce over all dims).
    keepdims: If true, retains reduced dimensions with length 1.
      Default value: `False` (i.e., keep all dims in `log_mean_x`).
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., `"log_soomean_exp"`).

  Returns:
    log_soomean_x: `logx.dtype` `Tensor` characterized by the natural-log of the
      sum of `x`` except that the element `logx[i]` is replaced with the
      log of the leave-`i`-out Geometric-average. The sum of the gradient of
      `log_soosum_x` is `n`, i.e., the number of reduced elements.
      Mathematically `log_soomean_x` is,
      ```none
      log_soomean_x[i] = log(Avg{h[j ; i] : j=0, ..., m-1})
      h[j ; i] = { u[j]                              j!=i
                 { GeometricAverage{u[k] : k != i}   j==i
      ```
    log_sum_x: `logx.dtype` `Tensor` corresponding to the natural-log of the
      average of `x`. The sum of the gradient of `log_mean_x` is `1`. Has
      reduced shape of `logx` (per `axis` and `keepdims`).
  """
  with tf.name_scope(name or 'log_soosum_exp'):
    return _log_soosum_exp_impl(logx, axis, keepdims, compute_mean=False)


def log_soomean_exp(logx, axis, keepdims=False, name=None):
  """Computes the log-swap-one-out-mean of `exp(logx)`.

  The swapped out element `logx[i]` is replaced with the log-leave-`i`-out
  geometric mean of `logx`.

  Args:
    logx: Floating-type `Tensor` representing `log(x)` where `x` is some
      positive value.
    axis: The dimensions to sum across. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(logx), rank(logx)]`.
      Default value: `None` (i.e., reduce over all dims).
    keepdims: If true, retains reduced dimensions with length 1.
      Default value: `False` (i.e., keep all dims in `log_mean_x`).
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., `"log_soomean_exp"`).

  Returns:
    log_soomean_x: ``Tensor` with same shape and dtype as `logx` representing
      the natural-log of the average of `x`` except that the element `logx[i]`
      is replaced with the log of the leave-`i`-out Geometric-average. The mean
      of the gradient of `log_soomean_x` is `1`. Mathematically `log_soomean_x`
      is,
      ```none
      log_soomean_x[i] = log(Avg{h[j ; i] : j=0, ..., m-1})
      h[j ; i] = { u[j]                              j!=i
                 { GeometricAverage{u[k] : k != i}   j==i
      ```
    log_mean_x: `logx.dtype` `Tensor` corresponding to the natural-log of the
      average of `x`. The sum of the gradient of `log_mean_x` is `1`. Has
      reduced shape of `logx` (per `axis` and `keepdims`).
  """
  with tf.name_scope(name or 'log_soomean_exp'):
    return _log_soosum_exp_impl(logx, axis, keepdims, compute_mean=True)


def _log_soosum_exp_impl(logx, axis, keepdims, compute_mean):
  """Implementation for `*soosum*` functions."""
  with tf.name_scope('log_soosum_exp_impl'):
    logx = tf.convert_to_tensor(logx, name='logx')
    log_loosum_x, log_sum_x, n = _log_loosum_exp_impl(
        logx, axis, keepdims, compute_mean=False)
    # The swap-one-out-sum ('soosum') is n different sums, each of which
    # replaces the i-th item with the i-th-left-out average (or the user
    # specified value), i.e.,
    # soo_sum_x[i] = [exp(logx) - exp(logx[i])] + exp(mean(logx[!=i]))
    #              =  exp(log_loosum_x[i])      + exp(loo_log_swap_in[i])
    n = tf.cast(n, logx.dtype)
    loo_log_swap_in = (
        (tf.reduce_sum(logx, axis=axis, keepdims=True) - logx) /
        (n - 1.))
    log_soosum_x = log_add_exp(log_loosum_x, loo_log_swap_in)
    if not compute_mean:
      return log_soosum_x, log_sum_x
    log_n = ps.log(n)
    return log_soosum_x - log_n, log_sum_x - log_n


def _log_loosum_exp_impl(logx, axis, keepdims, compute_mean):
  """Implementation for `*loosum*` functions."""
  with tf.name_scope('log_loosum_exp_impl'):
    logx = tf.convert_to_tensor(logx, name='logx')
    dtype = dtype_util.as_numpy_dtype(logx.dtype)

    if axis is not None:
      x = np.array(axis)
      axis = (tf.convert_to_tensor(axis, name='axis', dtype_hint=tf.int32)
              if x.dtype is np.object_ else x.astype(np.int32))

    log_sum_x = tf.reduce_logsumexp(logx, axis=axis, keepdims=True)

    # Later we'll want to compute the mean from a sum so we calculate the number
    # of reduced elements, n.
    n = ps.size(logx) // ps.size(log_sum_x)
    n = ps.cast(n, dtype)

    # log_loosum_x[i] =
    # = logsumexp(logx[j] : j != i)
    # = log( exp(logsumexp(logx)) - exp(logx[i]) )
    # = log( exp(logsumexp(logx - logx[i])) exp(logx[i])  - exp(logx[i]))
    # = logx[i] + log(exp(logsumexp(logx - logx[i])) - 1)
    # = logx[i] + log(exp(logsumexp(logx) - logx[i]) - 1)
    # = logx[i] + softplus_inverse(logsumexp(logx) - logx[i])
    d = log_sum_x - logx
    # We use `d != 0` rather than `d > 0.` because `d < 0.` should never happen;
    # if it does we want to complain loudly (which `softplus_inverse` will).
    d_ok = tf.not_equal(d, 0.)
    safe_d = tf.where(d_ok, d, 1.)
    d_ok_result = logx + softplus_inverse(safe_d)

    neg_inf = tf.constant(-np.inf, dtype=dtype)

    # When not(d_ok) and is_largest then we manually compute the
    # log_loosum_x. (We can efficiently do this for any one point but not all,
    # hence we still need the above calculation.) This is good because when
    # this condition is met, we cannot use the above calculation; its -inf.
    # We now compute the log-leave-out-max-sum, replicate it to every
    # point and make sure to select it only when we need to.
    max_logx = tf.reduce_max(logx, axis=axis, keepdims=True)
    is_largest = tf.equal(logx, max_logx)
    log_lomsum_x = tf.reduce_logsumexp(
        tf.where(is_largest, neg_inf, logx),
        axis=axis,
        keepdims=True)
    d_not_ok_result = tf.where(is_largest, log_lomsum_x, neg_inf)

    log_loosum_x = tf.where(d_ok, d_ok_result, d_not_ok_result)

    # We now squeeze log_sum_x so as if we used `keepdims=False`.
    # TODO(b/136176077): These mental gymnastics could all be replaced with
    # `tf.squeeze(log_sum_x, axis)` if tf.squeeze supported Tensor valued `axis`
    # arguments.
    if not keepdims:
      if axis is None:
        keepdims = np.array([], dtype=np.int32)
      else:
        rank = ps.rank(logx)
        keepdims = ps.setdiff1d(
            ps.range(rank),
            ps.non_negative_axis(axis, rank))
      squeeze_shape = tf.gather(ps.shape(logx), indices=keepdims)
      log_sum_x = tf.reshape(log_sum_x, shape=squeeze_shape)
      if ps.is_numpy(keepdims):
        tensorshape_util.set_shape(log_sum_x, np.array(logx.shape)[keepdims])

    # Set static shapes just in case we lost them.
    tensorshape_util.set_shape(n, [])
    tensorshape_util.set_shape(log_loosum_x, logx.shape)

    if not compute_mean:
      return log_loosum_x, log_sum_x, n

    log_nm1 = ps.log(max(1., n - 1.))
    log_n = ps.log(n)
    return log_loosum_x - log_nm1, log_sum_x - log_n, n
