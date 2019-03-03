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
"""Numerically stable variants of common mathematical expressions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def log1psquare(x, name=None):
  """Numerically stable calculation of `log(1 + x**2)` for small or large `|x|`.

  For sufficiently large `x` we use the following observation:

  ```none
  log(1 + x**2) =   2 log(|x|) + log(1 + 1 / x**2)
                --> 2 log(|x|)  as x --> inf
  ```

  Numerically, `log(1 + 1 / x**2)` is `0` when `1 / x**2` is small relative to
  machine epsilon.

  Args:
    x: Float `Tensor` input.
    name: Python string indicating the name of the TensorFlow operation.
      Default value: `'log1psquare'`.

  Returns:
    log1psq: Float `Tensor` representing `log(1. + x**2.)`.
  """
  with tf.compat.v1.name_scope(name, 'log1psquare', [x]):
    x = tf.convert_to_tensor(value=x, dtype_hint=tf.float32, name='x')
    dtype = x.dtype.as_numpy_dtype

    eps = np.finfo(dtype).eps.astype(np.float64)
    is_large = tf.abs(x) > (eps**-0.5).astype(dtype)

    # Mask out small x's so the gradient correctly propagates.
    abs_large_x = tf.where(is_large, tf.abs(x), tf.ones_like(x))
    return tf.where(is_large, 2. * tf.math.log(abs_large_x),
                    tf.math.log1p(tf.square(x)))


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
  with tf.compat.v1.name_scope(name, 'soft_threshold', [x, threshold]):
    x = tf.convert_to_tensor(value=x, name='x')
    threshold = tf.convert_to_tensor(
        value=threshold, dtype=x.dtype, name='threshold')
    return tf.sign(x) * tf.maximum(tf.abs(x) - threshold, 0.)


def clip_by_value_preserve_gradient(t, clip_value_min, clip_value_max,
                                    name=None):
  """Clips values to a specified min and max while leaving gradient unaltered.

  Like `tf.clip_by_value`, this function returns a tensor of the same type and
  shape as input `t` but with values clamped to be no smaller than to
  `clip_value_min` and no larger than `clip_value_max`. Unlike
  `tf.clip_by_value`, the gradient is unaffected by this op, i.e.,

  ```python
  tf.gradients(tfp.math.clip_by_value_preserve_gradient(x), x)[0]
  # ==> ones_like(x)
  ```

  Note: `clip_value_min` needs to be smaller or equal to `clip_value_max` for
  correct results.

  Args:
    t: A `Tensor`.
    clip_value_min: A scalar `Tensor`, or a `Tensor` with the same shape
      as `t`. The minimum value to clip by.
    clip_value_max: A scalar `Tensor`, or a `Tensor` with the same shape
      as `t`. The maximum value to clip by.
    name: A name for the operation (optional).
      Default value: `'clip_by_value_preserve_gradient'`.

  Returns:
    clipped_t: A clipped `Tensor`.
  """
  with tf.compat.v1.name_scope(name, 'clip_by_value_preserve_gradient',
                               [t, clip_value_min, clip_value_max]):
    t = tf.convert_to_tensor(value=t, name='t')
    clip_t = tf.clip_by_value(t, clip_value_min, clip_value_max)
    return t + tf.stop_gradient(clip_t - t)
