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
"""Functions for computing moving statistics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    "assign_moving_mean_variance",
    "assign_log_moving_mean_exp",
    "moving_mean_variance",
]


def assign_moving_mean_variance(
    mean_var, variance_var, value, decay, name=None):
  """Compute exponentially weighted moving {mean,variance} of a streaming value.

  The `value` updated exponentially weighted moving `mean_var` and
  `variance_var` are given by the following recurrence relations:

  ```python
  variance_var = decay * (variance_var + (1 - decay) * (value - mean_var)**2)
  mean_var     = decay * mean_var + (1 - decay) * value
  ```

  Note: `mean_var` is updated *after* `variance_var`, i.e., `variance_var` uses
  the lag-1 mean.

  For derivation justification, see [Finch (2009; Eq. 143)][1].
  Parameterization: Finch's `alpha` is `1 - decay`.

  Args:
    mean_var: `float`-like `Variable` representing the exponentially weighted
      moving mean. Same shape as `variance_var` and `value`.
    variance_var: `float`-like `Variable` representing the
      exponentially weighted moving variance. Same shape as `mean_var` and
      `value`.
    value: `float`-like `Tensor`. Same shape as `mean_var` and `variance_var`.
    decay: A `float`-like `Tensor`. The moving mean decay. Typically close to
      `1.`, e.g., `0.999`.
    name: Optional name of the returned operation.

  Returns:
    mean_var: `Variable` representing the `value`-updated exponentially weighted
      moving mean.
    variance_var: `Variable` representing the `value`-updated
      exponentially weighted moving variance.

  Raises:
    TypeError: if `mean_var` does not have float type `dtype`.
    TypeError: if `mean_var`, `variance_var`, `value`, `decay` have different
      `base_dtype`.

  #### References

  [1]: Tony Finch. Incremental calculation of weighted mean and variance.
       _Technical Report_, 2009.
       http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf
  """
  with tf.compat.v1.name_scope(name, "assign_moving_mean_variance",
                               [variance_var, mean_var, value, decay]):
    with tf.compat.v1.colocate_with(variance_var):
      with tf.compat.v1.colocate_with(mean_var):
        base_dtype = mean_var.dtype.base_dtype
        if not base_dtype.is_floating:
          raise TypeError(
              "mean_var.base_dtype({}) does not have float type "
              "`dtype`.".format(base_dtype.name))
        if base_dtype != variance_var.dtype.base_dtype:
          raise TypeError(
              "mean_var.base_dtype({}) != variance_var.base_dtype({})".format(
                  base_dtype.name,
                  variance_var.dtype.base_dtype.name))
        value = tf.convert_to_tensor(
            value=value, dtype=base_dtype, name="value")
        decay = tf.convert_to_tensor(
            value=decay, dtype=base_dtype, name="decay")
        delta = value - mean_var
        with tf.control_dependencies([delta]):
          # We want mean_{t+1} = decay * mean_t + (1. - decay) * value
          # We compute mean += decay * mean_t - mean_t + (1. - decay) * value =
          #   = (1. - decay) * (value - mean_t)
          mean_var = mean_var.assign_add((1. - decay) * delta)
          # We want variance_{t+1} = decay * (variance_t +
          #   + (1 - decay) * (value - mean_var)**2).
          # We compute variance -= variance_t - decay * (variance_t +
          #     + (1 - decay) * (value - mean_var)**2) =
          #   = (1 - decay) * variance_t
          #     - decay * (1 - decay) * (value - mean_var)**2
          #   = (1 - decay) * (variance_t - decay * (value - mean_var)**2).
          variance_var = variance_var.assign_sub(
              (1. - decay) * (variance_var - decay * tf.square(delta)))
        return mean_var, variance_var


def assign_log_moving_mean_exp(
    log_mean_exp_var, log_value, decay, name=None):
  """Compute the log of the exponentially weighted moving mean of the exp.

  If `log_value` is a draw from a stationary random variable, this function
  approximates `log(E[exp(log_value)])`, i.e., a weighted log-sum-exp. More
  precisely, a `tf.Variable`, `log_mean_exp_var`, is updated by `log_value`
  using the following identity:

  ```none
  log_mean_exp_var =
  = log(decay exp(log_mean_exp_var) + (1 - decay) exp(log_value))
  = log(exp(log_mean_exp_var + log(decay)) + exp(log_value + log1p(-decay)))
  = log_mean_exp_var
    + log(  exp(log_mean_exp_var   - log_mean_exp_var + log(decay))
          + exp(log_value - log_mean_exp_var + log1p(-decay)))
  = log_mean_exp_var
    + log_sum_exp([log(decay), log_value - log_mean_exp_var + log1p(-decay)]).
  ```

  In addition to numerical stability, this formulation is advantageous because
  `log_mean_exp_var` can be updated in a lock-free manner, i.e., using
  `assign_add`. (Note: the updates are not thread-safe; it's just that the
  update to the tf.Variable is presumed efficient due to being lock-free.)

  Args:
    log_mean_exp_var: `float`-like `Variable` representing the log of the
      exponentially weighted moving mean of the exp. Same shape as `log_value`.
    log_value: `float`-like `Tensor` representing a new (streaming) observation.
      Same shape as `log_mean_exp_var`.
    decay: A `float`-like `Tensor`. The moving mean decay. Typically close to
      `1.`, e.g., `0.999`.
    name: Optional name of the returned operation.

  Returns:
    log_mean_exp_var: A reference to the input 'Variable' tensor with the
      `log_value`-updated log of the exponentially weighted moving mean of exp.

  Raises:
    TypeError: if `log_mean_exp_var` does not have float type `dtype`.
    TypeError: if `log_mean_exp_var`, `log_value`, `decay` have different
      `base_dtype`.
  """
  with tf.compat.v1.name_scope(name, "assign_log_moving_mean_exp",
                               [log_mean_exp_var, log_value, decay]):
    # We want to update the variable in a numerically stable and lock-free way.
    # To do this, observe that variable `x` updated by `v` is:
    # x = log(w exp(x) + (1-w) exp(v))
    #   = log(exp(x + log(w)) + exp(v + log1p(-w)))
    #   = x + log(exp(x - x + log(w)) + exp(v - x + log1p(-w)))
    #   = x + lse([log(w), v - x + log1p(-w)])
    with tf.compat.v1.colocate_with(log_mean_exp_var):
      base_dtype = log_mean_exp_var.dtype.base_dtype
      if not base_dtype.is_floating:
        raise TypeError(
            "log_mean_exp_var.base_dtype({}) does not have float type "
            "`dtype`.".format(base_dtype.name))
      log_value = tf.convert_to_tensor(
          value=log_value, dtype=base_dtype, name="log_value")
      decay = tf.convert_to_tensor(value=decay, dtype=base_dtype, name="decay")
      delta = (log_value - log_mean_exp_var)[tf.newaxis, ...]
      x = tf.concat([
          tf.math.log(decay) * tf.ones_like(delta),
          delta + tf.math.log1p(-decay)
      ],
                    axis=0)
      x = tf.reduce_logsumexp(input_tensor=x, axis=0)
      return log_mean_exp_var.assign_add(x)


@deprecation.deprecated(
    "2019-03-22", "The `moving_mean_variance` function is deprecated. "
    "Use `assign_moving_mean_variance` and construct the Variables explicitly.")
def moving_mean_variance(value, decay, name=None):
  """Compute exponentially weighted moving {mean,variance} of a streaming value.

  The exponentially-weighting moving `mean_var` and `variance_var` are updated
  by `value` according to the following recurrence:

  ```python
  variance_var = decay * (variance_var + (1-decay) * (value - mean_var)**2)
  mean_var     = decay * mean_var + (1 - decay) * value
  ```

  Note: `mean_var` is updated *after* `variance_var`, i.e., `variance_var` uses
  the lag-`1` mean.

  For derivation justification, see [Finch (2009; Eq. 143)][1].

  Unlike `assign_moving_mean_variance`, this function handles
  variable creation.

  Args:
    value: `float`-like `Tensor`. Same shape as `mean_var` and `variance_var`.
    decay: A `float`-like `Tensor`. The moving mean decay. Typically close to
      `1.`, e.g., `0.999`.
    name: Optional name of the returned operation.

  Returns:
    mean_var: `Variable` representing the `value`-updated exponentially weighted
      moving mean.
    variance_var: `Variable` representing the `value`-updated
      exponentially weighted moving variance.

  Raises:
    TypeError: if `value_var` does not have float type `dtype`.
    TypeError: if `value`, `decay` have different `base_dtype`.

  #### References

  [1]: Tony Finch. Incremental calculation of weighted mean and variance.
       _Technical Report_, 2009.
       http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf
  """
  with tf.compat.v1.variable_scope(name, "moving_mean_variance",
                                   [value, decay]):
    value = tf.convert_to_tensor(value=value, name="value")
    base_dtype = value.dtype.base_dtype
    if not base_dtype.is_floating:
      raise TypeError(
          "value.base_dtype({}) does not have float type `dtype`.".format(
              base_dtype.name))
    decay = tf.convert_to_tensor(value=decay, dtype=base_dtype, name="decay")
    variance_var = tf.compat.v2.Variable(
        name="moving_variance",
        initial_value=tf.zeros(shape=value.shape, dtype=value.dtype),
        trainable=False)
    mean_var = tf.compat.v2.Variable(
        name="moving_mean",
        initial_value=tf.zeros(shape=value.shape, dtype=value.dtype),
        trainable=False)
    return assign_moving_mean_variance(
        mean_var, variance_var, value, decay)
