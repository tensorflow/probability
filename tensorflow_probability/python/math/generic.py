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


__all__ = [
    'log_add_exp',
    'log_combinations',
    'reduce_weighted_logsumexp',
    'soft_threshold',
    'softplus_inverse',
]


def log_combinations(n, counts, name='log_combinations'):
  """Multinomial coefficient.

  Given `n` and `counts`, where `counts` has last dimension `k`, we compute
  the multinomial coefficient as:

  ```n! / sum_i n_i!```

  where `i` runs over all `k` classes.

  Args:
    n: Floating-point `Tensor` broadcastable with `counts`. This represents `n`
      outcomes.
    counts: Floating-point `Tensor` broadcastable with `n`. This represents
      counts in `k` classes, where `k` is the last dimension of the tensor.
    name: A name for this operation (optional).

  Returns:
    log_combinations: `Tensor` representing the multinomial coefficient between
      `n` and `counts`.
  """
  # First a bit about the number of ways counts could have come in:
  # E.g. if counts = [1, 2], then this is 3 choose 2.
  # In general, this is (sum counts)! / sum(counts!)
  # The sum should be along the last dimension of counts. This is the
  # 'distribution' dimension. Here n a priori represents the sum of counts.
  with tf.name_scope(name):
    n = tf.convert_to_tensor(n, name='n')
    counts = tf.convert_to_tensor(counts, name='counts')
    total_permutations = tf.math.lgamma(n + 1)
    counts_factorial = tf.math.lgamma(counts + 1)
    redundant_permutations = tf.reduce_sum(counts_factorial, axis=-1)
    return total_permutations - redundant_permutations


def reduce_weighted_logsumexp(logx,
                              w=None,
                              axis=None,
                              keep_dims=False,
                              return_sign=False,
                              name=None):
  """Computes `log(abs(sum(weight * exp(elements across tensor dimensions))))`.

  If all weights `w` are known to be positive, it is more efficient to directly
  use `reduce_logsumexp`, i.e., `tf.reduce_logsumexp(logx + tf.log(w))` is more
  efficient than `du.reduce_weighted_logsumexp(logx, w)`.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keep_dims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` has no entries, all dimensions are reduced, and a
  tensor with a single element is returned.

  This function is more numerically stable than log(sum(w * exp(input))). It
  avoids overflows caused by taking the exp of large inputs and underflows
  caused by taking the log of small inputs.

  For example:

  ```python
  x = tf.constant([[0., 0, 0],
                   [0, 0, 0]])

  w = tf.constant([[-1., 1, 1],
                   [1, 1, 1]])

  du.reduce_weighted_logsumexp(x, w)
  # ==> log(-1*1 + 1*1 + 1*1 + 1*1 + 1*1 + 1*1) = log(4)

  du.reduce_weighted_logsumexp(x, w, axis=0)
  # ==> [log(-1+1), log(1+1), log(1+1)]

  du.reduce_weighted_logsumexp(x, w, axis=1)
  # ==> [log(-1+1+1), log(1+1+1)]

  du.reduce_weighted_logsumexp(x, w, axis=1, keep_dims=True)
  # ==> [[log(-1+1+1)], [log(1+1+1)]]

  du.reduce_weighted_logsumexp(x, w, axis=[0, 1])
  # ==> log(-1+5)
  ```

  Args:
    logx: The tensor to reduce. Should have numeric type.
    w: The weight tensor. Should have numeric type identical to `logx`.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keep_dims: If true, retains reduced dimensions with length 1.
    return_sign: If `True`, returns the sign of the result.
    name: A name for the operation (optional).

  Returns:
    lswe: The `log(abs(sum(weight * exp(x))))` reduced tensor.
    sign: (Optional) The sign of `sum(weight * exp(x))`.
  """
  with tf.name_scope(name or 'reduce_weighted_logsumexp'):
    logx = tf.convert_to_tensor(logx, name='logx')
    if w is None:
      lswe = tf.reduce_logsumexp(logx, axis=axis, keepdims=keep_dims)
      if return_sign:
        sgn = tf.ones_like(lswe)
        return lswe, sgn
      return lswe
    w = tf.convert_to_tensor(w, dtype=logx.dtype, name='w')
    log_absw_x = logx + tf.math.log(tf.abs(w))
    max_log_absw_x = tf.reduce_max(log_absw_x, axis=axis, keepdims=True)
    # If the largest element is `-inf` or `inf` then we don't bother subtracting
    # off the max. We do this because otherwise we'd get `inf - inf = NaN`. That
    # this is ok follows from the fact that we're actually free to subtract any
    # value we like, so long as we add it back after taking the `log(sum(...))`.
    max_log_absw_x = tf.where(
        tf.math.is_inf(max_log_absw_x),
        tf.zeros([], max_log_absw_x.dtype),
        max_log_absw_x)
    wx_over_max_absw_x = (tf.sign(w) * tf.exp(log_absw_x - max_log_absw_x))
    sum_wx_over_max_absw_x = tf.reduce_sum(
        wx_over_max_absw_x, axis=axis, keepdims=keep_dims)
    if not keep_dims:
      max_log_absw_x = tf.squeeze(max_log_absw_x, axis)
    sgn = tf.sign(sum_wx_over_max_absw_x)
    lswe = max_log_absw_x + tf.math.log(sgn * sum_wx_over_max_absw_x)
    if return_sign:
      return lswe, sgn
    return lswe


def soft_threshold(x, threshold, name=None):
  """Soft Thresholding operator.

  This operator is defined by the equations

  ```none
                                { x[i] - gamma,  x[i] >   gamma
  SoftThreshold(x, gamma)[i] =  { 0,             x[i] ==  gamma
                                { x[i] + gamma,  x[i] <  -gamma
  ```

  In the context of proximal gradient methods, we have

  ```none
  SoftThreshold(x, gamma) = prox_{gamma L1}(x)
  ```

  where `prox` is the proximity operator.  Thus the soft thresholding operator
  is used in proximal gradient descent for optimizing a smooth function with
  (non-smooth) L1 regularization, as outlined below.

  The proximity operator is defined as:

  ```none
  prox_r(x) = argmin{ r(z) + 0.5 ||x - z||_2**2 : z },
  ```

  where `r` is a (weakly) convex function, not necessarily differentiable.
  Because the L2 norm is strictly convex, the above argmin is unique.

  One important application of the proximity operator is as follows.  Let `L` be
  a convex and differentiable function with Lipschitz-continuous gradient.  Let
  `R` be a convex lower semicontinuous function which is possibly
  nondifferentiable.  Let `gamma` be an arbitrary positive real.  Then

  ```none
  x_star = argmin{ L(x) + R(x) : x }
  ```

  if and only if the fixed-point equation is satisfied:

  ```none
  x_star = prox_{gamma R}(x_star - gamma grad L(x_star))
  ```

  Proximal gradient descent thus typically consists of choosing an initial value
  `x^{(0)}` and repeatedly applying the update

  ```none
  x^{(k+1)} = prox_{gamma^{(k)} R}(x^{(k)} - gamma^{(k)} grad L(x^{(k)}))
  ```

  where `gamma` is allowed to vary from iteration to iteration.  Specializing to
  the case where `R(x) = ||x||_1`, we minimize `L(x) + ||x||_1` by repeatedly
  applying the update

  ```
  x^{(k+1)} = SoftThreshold(x - gamma grad L(x^{(k)}), gamma)
  ```

  (This idea can also be extended to second-order approximations, although the
  multivariate case does not have a known closed form like above.)

  Args:
    x: `float` `Tensor` representing the input to the SoftThreshold function.
    threshold: nonnegative scalar, `float` `Tensor` representing the radius of
      the interval on which each coordinate of SoftThreshold takes the value
      zero.  Denoted `gamma` above.
    name: Python string indicating the name of the TensorFlow operation.
      Default value: `'soft_threshold'`.

  Returns:
    softthreshold: `float` `Tensor` with the same shape and dtype as `x`,
      representing the value of the SoftThreshold function.

  #### References

  [1]: Yu, Yao-Liang. The Proximity Operator.
       https://www.cs.cmu.edu/~suvrit/teach/yaoliang_proximity.pdf

  [2]: Wikipedia Contributors. Proximal gradient methods for learning.
       _Wikipedia, The Free Encyclopedia_, 2018.
       https://en.wikipedia.org/wiki/Proximal_gradient_methods_for_learning

  """
  # https://math.stackexchange.com/questions/471339/derivation-of-soft-thresholding-operator
  with tf.name_scope(name or 'soft_threshold'):
    x = tf.convert_to_tensor(x, name='x')
    threshold = tf.convert_to_tensor(threshold, dtype=x.dtype, name='threshold')
    return tf.sign(x) * tf.maximum(tf.abs(x) - threshold, 0.)


# TODO(jvdillon): Merge this test back into:
# tensorflow/python/ops/softplus_op_test.py
# once TF core is accepting new ops.
def softplus_inverse(x, name=None):
  """Computes the inverse softplus, i.e., x = softplus_inverse(softplus(x)).

  Mathematically this op is equivalent to:

  ```none
  softplus_inverse = log(exp(x) - 1.)
  ```

  Args:
    x: `Tensor`. Non-negative (not enforced), floating-point.
    name: A name for the operation (optional).

  Returns:
    `Tensor`. Has the same type/shape as input `x`.
  """
  with tf.name_scope(name or 'softplus_inverse'):
    x = tf.convert_to_tensor(x, name='x')
    # We begin by deriving a more numerically stable softplus_inverse:
    # x = softplus(y) = Log[1 + exp{y}], (which means x > 0).
    # ==> exp{x} = 1 + exp{y}                                (1)
    # ==> y = Log[exp{x} - 1]                                (2)
    #       = Log[(exp{x} - 1) / exp{x}] + Log[exp{x}]
    #       = Log[(1 - exp{-x}) / 1] + Log[exp{x}]
    #       = Log[1 - exp{-x}] + x                           (3)
    # (2) is the "obvious" inverse, but (3) is more stable than (2) for large x.
    # For small x (e.g. x = 1e-10), (3) will become -inf since 1 - exp{-x} will
    # be zero. To fix this, we use 1 - exp{-x} approx x for small x > 0.
    #
    # In addition to the numerically stable derivation above, we clamp
    # small/large values to be congruent with the logic in:
    # tensorflow/core/kernels/softplus_op.h
    #
    # Finally, we set the input to one whenever the input is too large or too
    # small. This ensures that no unchosen codepath is +/- inf. This is
    # necessary to ensure the gradient doesn't get NaNs. Recall that the
    # gradient of `where` behaves like `pred*pred_true + (1-pred)*pred_false`
    # thus an `inf` in an unselected path results in `0*inf=nan`. We are careful
    # to overwrite `x` with ones only when we will never actually use this
    # value. Note that we use ones and not zeros since `log(expm1(0.)) = -inf`.
    threshold = np.log(np.finfo(dtype_util.as_numpy_dtype(x.dtype)).eps) + 2.
    is_too_small = x < np.exp(threshold)
    is_too_large = x > -threshold
    too_small_value = tf.math.log(x)
    too_large_value = x
    # This `where` will ultimately be a NOP because we won't select this
    # codepath whenever we used the surrogate `ones_like`.
    x = tf.where(is_too_small | is_too_large, tf.ones([], x.dtype), x)
    y = x + tf.math.log(-tf.math.expm1(-x))  # == log(expm1(x))
    return tf.where(is_too_small,
                    too_small_value,
                    tf.where(is_too_large, too_large_value, y))


def log_add_exp(x, y, name=None):
  """Computes `log(exp(x) + exp(y))` in a numerically stable way.

  Args:
    x: `float` `Tensor` broadcastable with `y`.
    y: `float` `Tensor` broadcastable with `x`.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., `'log_add_exp'`).

  Returns:
    log_add_exp: `log(exp(x) + exp(y))` computed in a numerically stable way.
  """
  with tf.name_scope(name or 'log_add_exp'):
    dtype = dtype_util.common_dtype([x, y], dtype_hint=tf.float32)
    x = tf.convert_to_tensor(x, dtype=dtype, name='x')
    y = tf.convert_to_tensor(y, dtype=dtype, name='y')
    return tf.maximum(x, y) + tf.math.softplus(-abs(x - y))
