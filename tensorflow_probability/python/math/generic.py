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

import collections

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import custom_gradient as tfp_custom_gradient
from tensorflow_probability.python.internal import distribute_lib
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import variadic_reduce
from tensorflow_probability.python.math.scan_associative import scan_associative


__all__ = [
    'log_add_exp',
    'log_cosh',
    'log_sub_exp',
    'log_combinations',
    'log_cumsum_exp',
    'log1mexp',
    'reduce_kahan_sum',
    'reduce_logmeanexp',
    'reduce_weighted_logsumexp',
    'smootherstep',
    'soft_sorting_matrix',
    'soft_threshold',
    'softplus_inverse',
    'sqrt1pm1',
]


def log_combinations(n, counts, name='log_combinations'):
  """Log multinomial coefficient.

  Given `n` and `counts`, where `counts` has last dimension `k`, we define
  the multinomial coefficient as:

  ```n! / prod_i n_i!```

  where `i` runs over all `k` classes.

  This function computes the natural logarithm of the multinomial coefficient.

  Args:
    n: Floating-point `Tensor` broadcastable with `counts`. This represents `n`
      outcomes.
    counts: Floating-point `Tensor` broadcastable with `n`. This represents
      counts in `k` classes, where `k` is the last dimension of the tensor.
    name: A name for this operation (optional).

  Returns:
    log_combinations: `Tensor` representing the log of the multinomial
      coefficient between `n` and `counts`.
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


# TODO(b/154562929): Remove this once the built-in op supports XLA.
# TODO(b/156297366): Derivatives of this function may not always be correct.
def log_cumsum_exp(x, axis=-1, name=None):
  """Computes log(cumsum(exp(x))).

  This is a pure-TF implementation of `tf.math.cumulative_logsumexp`; unlike
  the built-in op, it supports XLA compilation. It uses a similar algorithmic
  technique (parallel prefix sum) as the built-in op, so it has similar numerics
  and asymptotic performace. However, this implemenentation currently has higher
  overhead, so it is significantly slower on smaller inputs (`n < 10000`).

  Args:
    x: the `Tensor` to sum over.
    axis: int `Tensor` axis to sum over.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., `'cumulative_logsumexp'`).
  Returns:
    cumulative_logsumexp: `Tensor` of the same shape as `x`.
  """
  with tf.name_scope(name or 'cumulative_logsumexp'):
    x = tf.convert_to_tensor(x, name='x')
    def safe_logsumexp(x, y):
      result = log_add_exp(x, y)
      # Remove spurious `NaN`s that arise from subtracting infinities.
      return tf.where(tf.math.is_finite(result), result, -np.inf)
    return scan_associative(safe_logsumexp, x, axis=axis)


def _kahan_reduction(x, y):
  """Implements the Kahan summation reduction."""
  (s, c), (s1, c1) = x, y
  for val in -c1, s1:
    u = val - c
    t = s + u
    # TODO(b/173158845): XLA:CPU reassociates-to-zero the correction term.
    c = (t - s) - u
    s = t
  return s, c


def _kahan_reduce_bwd(axis, reducer, unsqueezed_shape, aux, grads):
  operands, inits = aux
  del axis, inits, reducer  # unused
  # Return (None, None) for gradients w.r.t. inits
  return (tf.broadcast_to(tf.reshape(grads[0], unsqueezed_shape),
                          ps.shape(operands[0])),
          None), (None, None)


def _kahan_reduce_tangents(axis, primals, tangents):
  del primals  # unused
  doperands, _ = tangents
  reduced_tangent = tf.reduce_sum(doperands[0], axis)
  return (reduced_tangent, tf.zeros_like(reduced_tangent))


_reduce_kahan_sum = variadic_reduce.make_variadic_reduce(
    _kahan_reduction, _kahan_reduce_bwd, _kahan_reduce_tangents)


class Kahan(collections.namedtuple('Kahan', ['total', 'correction'])):
  """Result of Kahan summation, i.e., `sum = total - correction`.

  All the high-order bits of `sum` are held in the `total` field,
  so the correction can be dropped when returning to ordinary floating-point.
  """
  __slots__ = ()

  def __add__(self, x):
    return Kahan._make(_kahan_reduction(
        self, x if isinstance(x, Kahan) else (x, 0)))

  def __radd__(self, x):
    return Kahan._make(_kahan_reduction(
        self, x if isinstance(x, Kahan) else (x, 0)))

  def __neg__(self):
    return Kahan(-self.total, -self.correction)

  def __sub__(self, y):
    return Kahan._make(_kahan_reduction(
        self, -y if isinstance(y, Kahan) else (-y, 0)))

  def __rsub__(self, x):
    return Kahan._make(_kahan_reduction(
        x if isinstance(x, Kahan) else (x, 0), -self))


def reduce_kahan_sum(input_tensor, axis=None, keepdims=False, name=None):
  """Reduces the input tensor along the given axis using Kahan summation.

  Returns both the total and the correction term, as a `namedtuple`,
  representing the sum in higher precision as `total - correction`.

  A practical use-case is computing the difference of two large (magnitude) sums
  we expect to be nearly equal. If instead we take their difference as
  `(s0.total - s1.total) - (s0.correction - s1.correction)`, we can retain more
  precision in computing their difference.

  Note that `total` holds all the high-order bits of the sum, so the correction
  can be safely neglected if further enhanced precision computations are not
  required.

  Note: (TF + JAX) This function does not work properly on XLA:CPU without the
  environment variable: `XLA_FLAGS=--xla_cpu_enable_fast_math=false`, due to
  LLVM's reassociation optimizations, which simplify error terms to zero.

  Args:
    input_tensor: The tensor to sum.
    axis: One of `None`, a Python `int`, or a sequence of Python `int`. The axes
      to be reduced. `None` is taken as "reduce all axes".
    keepdims: Python `bool` indicating whether we return a tensor with singleton
      dimensions in the reduced axes (`True`), or squeeze the axes out (default,
      `False`).
    name: Optional name for ops in scope.

  Returns:
    reduced: A `Kahan(total, correction)` namedtuple.
  """
  with tf.name_scope(name or 'reduce_kahan_sum'):
    t = tf.convert_to_tensor(input_tensor)
    operands = (t, tf.zeros_like(t))
    inits = (tf.zeros([], dtype=t.dtype),) * 2
    return Kahan._make(
        _reduce_kahan_sum(operands, inits, axis=axis, keepdims=keepdims))


def reduce_logmeanexp(input_tensor, axis=None, keepdims=False,
                      experimental_named_axis=None, name=None):
  """Computes `log(mean(exp(input_tensor)))`.

  Reduces `input_tensor` along the dimensions given in `axis`.  Unless
  `keepdims` is true, the rank of the tensor is reduced by 1 for each entry in
  `axis`. If `keepdims` is true, the reduced dimensions are retained with length
  1.

  If `axis` has no entries, all dimensions are reduced, and a tensor with a
  single element is returned.

  This function is more numerically stable than `log(reduce_mean(exp(input)))`.
  It avoids overflows caused by taking the exp of large inputs and underflows
  caused by taking the log of small inputs.

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims:  Boolean.  Whether to keep the axis as singleton dimensions.
      Default value: `False` (i.e., squeeze the reduced dimensions).
    experimental_named_axis: A `str or list of `str` axis names to additionally
      reduce over. Providing `None` will not reduce over any axes.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., `'reduce_logmeanexp'`).

  Returns:
    log_mean_exp: The reduced tensor.
  """
  with tf.name_scope(name or 'reduce_logmeanexp'):
    named_axes = distribute_lib.canonicalize_named_axis(experimental_named_axis)
    lse = distribute_lib.reduce_logsumexp(input_tensor, axis=axis,
                                          keepdims=keepdims,
                                          named_axis=named_axes)
    n = ps.size(input_tensor) // ps.size(lse)
    for named_axis in named_axes:
      n = n * distribute_lib.get_axis_size(named_axis)
    log_n = tf.math.log(tf.cast(n, lse.dtype))
    return lse - log_n


def reduce_weighted_logsumexp(logx,
                              w=None,
                              axis=None,
                              keep_dims=False,
                              return_sign=False,
                              experimental_named_axis=None,
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
    experimental_named_axis: A `str or list of `str` axis names to additionally
      reduce over. Providing `None` will not reduce over any axes.
    name: A name for the operation (optional).

  Returns:
    lswe: The `log(abs(sum(weight * exp(x))))` reduced tensor.
    sign: (Optional) The sign of `sum(weight * exp(x))`.
  """
  with tf.name_scope(name or 'reduce_weighted_logsumexp'):
    logx = tf.convert_to_tensor(logx, name='logx')
    if w is None:
      lswe = distribute_lib.reduce_logsumexp(logx, axis=axis,
                                             keepdims=keep_dims,
                                             named_axis=experimental_named_axis)
      if return_sign:
        sgn = tf.ones_like(lswe)
        return lswe, sgn
      return lswe
    w = tf.convert_to_tensor(w, dtype=logx.dtype, name='w')
    log_absw_x = logx + tf.math.log(tf.abs(w))
    max_log_absw_x = distribute_lib.reduce_max(
        log_absw_x, axis=axis, keepdims=True,
        named_axis=experimental_named_axis)
    # If the largest element is `-inf` or `inf` then we don't bother subtracting
    # off the max. We do this because otherwise we'd get `inf - inf = NaN`. That
    # this is ok follows from the fact that we're actually free to subtract any
    # value we like, so long as we add it back after taking the `log(sum(...))`.
    max_log_absw_x = tf.where(
        tf.math.is_inf(max_log_absw_x),
        tf.zeros([], max_log_absw_x.dtype),
        max_log_absw_x)
    wx_over_max_absw_x = (tf.sign(w) * tf.exp(log_absw_x - max_log_absw_x))
    sum_wx_over_max_absw_x = distribute_lib.reduce_sum(
        wx_over_max_absw_x, axis=axis, keepdims=keep_dims,
        named_axis=experimental_named_axis)
    if not keep_dims:
      max_log_absw_x = tf.squeeze(max_log_absw_x, axis)
    sgn = tf.sign(sum_wx_over_max_absw_x)
    lswe = max_log_absw_x + tf.math.log(sgn * sum_wx_over_max_absw_x)
    if return_sign:
      return lswe, sgn
    return lswe


def reduce_log_harmonic_mean_exp(input_tensor,
                                 axis=None,
                                 keepdims=False,
                                 experimental_named_axis=None,
                                 name=None):
  """Computes `log(1 / mean(1 / exp(input_tensor)))`.

  Reduces `input_tensor` along the dimensions given in `axis`.  Unless
  `keepdims` is true, the rank of the tensor is reduced by 1 for each entry in
  `axis`. If `keepdims` is true, the reduced dimensions are retained with length
  1.

  If `axis` has no entries, all dimensions are reduced, and a tensor with a
  single element is returned.

  This function is more numerically stable than `log(1 / mean(1 - exp(input)))`.
  It avoids overflows caused by taking the exp of large inputs and underflows
  caused by taking the log of small inputs.

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims:  Boolean.  Whether to keep the axis as singleton dimensions.
      Default value: `False` (i.e., squeeze the reduced dimensions).
    experimental_named_axis: A `str or list of `str` axis names to additionally
      reduce over. Providing `None` will not reduce over any axes.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., `'reduce_log_harmonic_mean_exp'`).

  Returns:
    log_mean_exp: The reduced tensor.
  """
  with tf.name_scope(name or 'reduce_log_harmonic_mean_exp'):
    return -reduce_logmeanexp(-input_tensor, axis=axis, keepdims=keepdims,
                              experimental_named_axis=experimental_named_axis)


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

    # The following is similar to using the standard method
    # `tf.maximum(x, y) + tf.math.softplus(-abs(x - y))`
    # to compute `log_add_exp`. However, both `tf.maximum` and
    # `abs(x - y)` have discontinuities in their derivatives
    # along `x == y`.
    # This version ensures that the contribution of the discontinuities
    # to the derivative all cancel leaving a continuous result without
    # changing the domain in which the original was valid.
    larger = tf.maximum(x, y)
    return larger + tf.math.softplus((x - larger) + (y - larger))


def smootherstep(x, name=None):
  """Computes a sigmoid-like interpolation function on the unit-interval.

  Equivalent to:

  ```python
  x = tf.clip_by_value(x, clip_value_min=0., clip_value_max=1.)
  y = x**3. * (6. * x**2. - 15. * x + 10.)
  ```

  For more details see [Wikipedia][1].

  Args:
    x: `float` `Tensor`.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., `'smootherstep'`).

  Returns:
    smootherstep: `float` `Tensor` with the same shape and dtype as `x`,
      representing the value of the smootherstep function.

  #### References

  [1]: "Smoothstep." Wikipedia.
       https://en.wikipedia.org/wiki/Smoothstep#Variations
  """
  with tf.name_scope(name or 'smootherstep'):
    x = tf.clip_by_value(x, clip_value_min=0., clip_value_max=1.)
    # Note: Grappler will rewrite:
    #   x**2, x**3
    # as:
    #   x2 = tf.square(x)
    #   x3 = tf.square(x) * x
    # and common subexpression elimination (CSE) will produce:
    #   x2 = tf.square(x)
    #   x3 = x2 * x
    return x**3. * (6. * x**2. - 15. * x + 10.)


def log_sub_exp(x, y, return_sign=False, name=None):
  """Compute `log(exp(max(x, y)) - exp(min(x, y)))` in a numerically stable way.

  Use `return_sign=True` unless `x >= y`, since we can't represent a negative in
  log-space.

  Args:
    x: Float `Tensor` broadcastable with `y`.
    y: Float `Tensor` broadcastable with `x`.
    return_sign: Whether or not to return the second output value `sign`. If
      it is known that `x >= y`, this is unnecessary.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., `'log_sub_exp'`).

  Returns:
    logsubexp: Float `Tensor` of `log(exp(max(x, y)) - exp(min(x, y)))`.
    sign: Float `Tensor` +/-1 indicating the sign of `exp(x) - exp(y)`.
  """
  with tf.name_scope(name or 'log_sub_exp'):
    dtype = dtype_util.common_dtype([x, y], dtype_hint=tf.float32)
    x = tf.convert_to_tensor(x, dtype=dtype, name='x')
    y = tf.convert_to_tensor(y, dtype=dtype, name='y')
    larger = tf.maximum(x, y)
    smaller = tf.minimum(x, y)
    zero = dtype_util.as_numpy_dtype(dtype)(0)
    result = larger + log1mexp(tf.maximum(larger - smaller, zero))
    if return_sign:
      ones = tf.ones([], result.dtype)
      return result, tf.where(x < y, -ones, ones)
    return result


def log1mexp(x, name=None):
  """Compute `log(1 - exp(-|x|))` elementwise in a numerically stable way.

  Args:
    x: Float `Tensor`.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., `'log1mexp'`).

  Returns:
    log1mexp: Float `Tensor` of `log1mexp(x)`.

  #### References

  [1]: Machler, Martin. Accurately computing log(1 - exp(-|a|))
       https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
  """

  with tf.name_scope(name or 'log1mexp'):
    dtype = dtype_util.common_dtype([x], dtype_hint=tf.float32)
    x = tf.convert_to_tensor(x, dtype=dtype, name='x')
    x = tf.math.abs(x)
    return tf.where(
        # This switching point is recommended in [1].
        x < np.log(2), tf.math.log(-tf.math.expm1(-x)),
        tf.math.log1p(-tf.math.exp(-x)))


def sqrt1pm1(x):
  """Compute `sqrt(x + 1) - 1` elementwise in a numerically stable way.

  Args:
    x: Float `Tensor`.

  Returns:
    sqrt1pm1: Float `Tensor` of `sqrt1pm1(x)`.
  """
  # We follow Boost
  # https://www.boost.org/doc/libs/1_49_0/libs/math/doc/sf_and_dist/html/math_toolkit/special/powers/sqrt1pm1.html
  # and compute expm1(0.5 * log1p(x)).
  #
  # We can also derive an alternative formula by multiplying and
  # dividing by sqrt(x + 1) + 1:
  #   sqrt(x + 1) - 1 = (x + 1 - 1) / (sqrt(x + 1) + 1)
  #                   = x / (sqrt(x + 1) + 1)
  # The latter form is well-conditioned everywhere, and in particular
  # does not experience catastrophic cancellation when x ~ 0.  However,
  # without where-gating, it emits `nan` when x is `+inf`.
  return tf.math.expm1(0.5 * tf.math.log1p(x))


def _log_cosh_impl(x):
  """Body of numerically stable log_cosh."""
  # log(cosh(x)) = log(e^x + e^-x) - log(2).
  # For x > 0, we can rewrite this as x + log(1 + e^(-2 * x)) - log(2).
  # The second term will be small when x is large, so we don't get any large
  # cancellations.
  # Similarly for x < 0, we can rewrite the expression as -x + log(1 + e^(2 *
  # x)) - log(2)
  # This gives us abs(x) + softplus(-2 * abs(x)) - log(2)

  # For x close to zero, we can write the taylor series of softplus(
  # -2 * abs(x)) to see that we get;
  # log(2) - abs(x) + x**2 / 2. - x**4 / 12 + x**6 / 45. + O(x**8)
  # We can cancel out terms to get:
  # x ** 2 / 2.  * (1. - x ** 2 / 6) + x ** 6 / 45. + O(x**8)
  # For x < 45 * sixthroot(smallest normal), all higher level terms
  # disappear and we can use the above expression.
  numpy_dtype = dtype_util.as_numpy_dtype(x.dtype)
  abs_x = tf.math.abs(x)
  logcosh = abs_x + tf.math.softplus(-2 * abs_x) - np.log(2).astype(
      numpy_dtype)
  bound = 45. * np.power(np.finfo(numpy_dtype).tiny, 1 / 6.)
  return tf.where(
      abs_x <= bound,
      tf.math.exp(tf.math.log(abs_x) + tf.math.log1p(-tf.square(abs_x) / 6.)),
      logcosh)


def _log_cosh_jvp(primals, tangents):
  x, = primals
  dx, = tangents
  return _log_cosh_impl(x), tf.math.tanh(x) * dx


# The gradient of log(cosh(x)) is tanh(x)
@tfp_custom_gradient.custom_gradient(
    vjp_fwd=lambda x: (_log_cosh_impl(x), x),
    vjp_bwd=lambda x, dy: dy * tf.math.tanh(x),
    jvp_fn=_log_cosh_jvp)
def _log_cosh_custom_gradient(x):
  return _log_cosh_impl(x)


def log_cosh(x, name=None):
  """Compute `log(cosh(x))` in a numerically stable way.

  Args:
    x: Float `Tensor`.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., `'log_cosh'`).

  Returns:
    log_cosh: `log_cosh(x)`.
  """
  with tf.name_scope(name or 'log_cosh'):
    dtype = dtype_util.common_dtype([x], dtype_hint=tf.float32)
    x = tf.convert_to_tensor(x, dtype=dtype, name='x')
    return _log_cosh_custom_gradient(x)


def soft_sorting_matrix(x, temperature, name=None):
  """Computes a matrix representing a continuous relaxation of sorting.

  Given a vector `x`, there exists a permutation matrix `P_x`, when applied to
  `x` gives `x` sorted in decreasing order. Here, we compute a continuous
  relaxation of `P_x`, parameterized by `temperature`. This continuous
  relaxation satisfies the property that it is a unimodal row-stochastic matrix,
  meaning that all entries are non-negative, all rows sum to 1., and there is a
  unique maximum entry in each column. The unique maximum entry will correspond
  to the location of a `1` in the exact sorting permutation.

  Complexity: Given a vector `x` of size `N`, this operation will take `O(N**2)`
    time.

  This is also known as a Neural sort in [1].

  Args:
    x: `float` `Tensor`. Argument to compute the relaxed sorting matrix with
      respect to.  The relaxed permutation is computed with respect to the last
      axis.
    temperature: Positive `float` Tensor`. When `temperature` approaches zero,
      this will retrieve the exact permutation matrix corresponding to sorting
      from largest to smallest.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., `'soft_sorting_matrix'`).
  Returns:
    soft_sort: A unimodal row-stochastic matrix. Applying this matrix on x
      will in the limit of low temperature, sort it.

  #### References

  [1]: Aditya Grover, Eric Wang, Aaron Zweig, Stefano Ermon.
       Stochastic Optimization of Sorting Networks via Continuous Relaxations.
       https://arxiv.org/abs/1903.08850
  """
  with tf.name_scope(name or 'soft_sorting_matrix'):
    dtype = dtype_util.common_dtype([temperature, x], dtype_hint=tf.float32)
    temperature = tf.convert_to_tensor(
        temperature, name='temperature', dtype=dtype)
    x = tf.convert_to_tensor(x, name='x', dtype=dtype)
    n = tf.shape(x)[-1]
    y = x[..., tf.newaxis]
    pairwise_distances = tf.abs(y - tf.linalg.matrix_transpose(y))
    scaling = tf.cast(
        tf.range(n - 1, -(n - 1) - 1, delta=-2), dtype=dtype)
    p_logits = tf.linalg.matrix_transpose(
        tf.matmul(y, scaling[tf.newaxis, ...]) - tf.reduce_sum(
            pairwise_distances, axis=-1)[..., tf.newaxis])
    y = tf.nn.softmax(p_logits / temperature, axis=-1)
    return y
