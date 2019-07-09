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
"""Positive-Semidefinite Kernel library utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util

__all__ = [
    'pad_shape_with_ones',
    'maybe_get_common_dtype',
    'sum_rightmost_ndims_preserving_shape',
]


def pad_shape_with_ones(x, ndims, start=-1):
  """Maybe add `ndims` ones to `x.shape` starting at `start`.

  If `ndims` is zero, this is a no-op; otherwise, we will create and return a
  new `Tensor` whose shape is that of `x` with `ndims` ones concatenated on the
  right side. If the shape of `x` is known statically, the shape of the return
  value will be as well.

  Args:
    x: The `Tensor` we'll return a reshaping of.
    ndims: Python `integer` number of ones to pad onto `x.shape`.
    start: Python `integer` specifying where to start padding with ones. Must
      be a negative integer. For instance, a value of `-1` means to pad at the
      end of the shape. Default value: `-1`.
  Returns:
    If `ndims` is zero, `x`; otherwise, a `Tensor` whose shape is that of `x`
    with `ndims` ones concatenated on the right side. If possible, returns a
    `Tensor` whose shape is known statically.
  Raises:
    ValueError: if `ndims` is not a Python `integer` greater than or equal to
    zero.
  """
  if not (isinstance(ndims, int) and ndims >= 0):
    raise ValueError(
        '`ndims` must be a Python `integer` greater than zero. Got: {}'
        .format(ndims))
  if not (isinstance(start, int) and start <= -1):
    raise ValueError(
        '`start` must be a Python `integer` less than zero. Got: {}'
        .format(start))
  if ndims == 0:
    return x
  x = tf.convert_to_tensor(value=x)
  original_shape = x.shape
  rank = tf.rank(input=x)
  first_shape = tf.shape(input=x)[:rank + start + 1]
  second_shape = tf.shape(input=x)[rank + start + 1:]
  new_shape = distribution_util.pad(
      first_shape, axis=0, back=True, value=1, count=ndims)
  new_shape = tf.concat([new_shape, second_shape], axis=0)
  x = tf.reshape(x, new_shape)
  if start == -1:
    x.set_shape(original_shape.concatenate([1] * ndims))
  elif original_shape.ndims is not None:
    x.set_shape(original_shape[
        :original_shape.ndims + start + 1].concatenate(
            [1] * ndims).concatenate(
                original_shape[original_shape.ndims + start + 1:]))
  return x


def sum_rightmost_ndims_preserving_shape(x, ndims):
  """Return `Tensor` with right-most ndims summed.

  Args:
    x: the `Tensor` whose right-most `ndims` dimensions to sum
    ndims: number of right-most dimensions to sum.

  Returns:
    A `Tensor` resulting from calling `reduce_sum` on the `ndims` right-most
    dimensions. If the shape of `x` is statically known, the result will also
    have statically known shape. Otherwise, the resulting shape will only be
    known at runtime.
  """
  x = tf.convert_to_tensor(value=x)
  if x.shape.ndims is not None:
    axes = tf.range(x.shape.ndims - ndims, x.shape.ndims)
  else:
    axes = tf.range(tf.rank(x) - ndims, tf.rank(x))
  return tf.reduce_sum(input_tensor=x, axis=axes)


@tf.custom_gradient
def sqrt_with_finite_grads(x, name=None):
  """A sqrt function whose gradient at zero is very large but finite.

  Args:
    x: a `Tensor` whose sqrt is to be computed.
    name: a Python `str` prefixed to all ops created by this function.
      Default `None` (i.e., "sqrt_with_finite_grads").

  Returns:
    sqrt: the square root of `x`, with an overridden gradient at zero
    grad: a gradient function, which is the same as sqrt's gradient everywhere
      except at zero, where it is given a large finite value, instead of `inf`.

  Raises:
    TypeError: if `tf.convert_to_tensor(x)` is not a `float` type.

  Often in kernel functions, we need to compute the L2 norm of the difference
  between two vectors, `x` and `y`: `sqrt(sum_i((x_i - y_i) ** 2))`. In the
  case where `x` and `y` are identical, e.g., on the diagonal of a kernel
  matrix, we get `NaN`s when we take gradients with respect to the inputs. To
  see, this consider the forward pass:

    ```
    [x_1 ... x_N]  -->  [x_1 ** 2 ... x_N ** 2]  -->
        (x_1 ** 2 + ... + x_N ** 2)  -->  sqrt((x_1 ** 2 + ... + x_N ** 2))
    ```

  When we backprop through this forward pass, the `sqrt` yields an `inf` because
  `grad_z(sqrt(z)) = 1 / (2 * sqrt(z))`. Continuing the backprop to the left, at
  the `x ** 2` term, we pick up a `2 * x`, and when `x` is zero, we get
  `0 * inf`, which is `NaN`.

  We'd like to avoid these `NaN`s, since they infect the rest of the connected
  computation graph. Practically, when two inputs to a kernel function are
  equal, we are in one of two scenarios:
    1. We are actually computing k(x, x), in which case norm(x - x) is
       identically zero, independent of x. In this case, we'd like the
       gradient to reflect this independence: it should be zero.
    2. We are computing k(x, y), and x just *happens* to have the same value
       as y. The gradient at such inputs is in fact ill-defined (there is a
       cusp in the sqrt((x - y) ** 2) surface along the line x = y). There are,
       however, an infinite number of sub-gradients, all of which are valid at
       all such inputs. By symmetry, there is exactly one which is "special":
       zero, and we elect to use that value here. In practice, having two
       identical inputs to a kernel matrix is probably a pathological
       situation to be avoided, but that is better resolved at a higher level
       than this.

  To avoid the infinite gradient at zero, we use tf.custom_gradient to redefine
  the gradient at zero. We assign it to be a very large value, specifically
  the sqrt of the max value of the floating point dtype of the input. We use
  the sqrt (as opposed to just using the max floating point value) to avoid
  potential overflow when combining this value with others downstream.
  """
  with tf1.name_scope(name, 'sqrt_with_finite_grads', [x]):
    x = tf.convert_to_tensor(value=x, name='x')
    if not x.dtype.is_floating:
      raise TypeError('Input `x` must be floating type.')
    def grad(grad_ys):
      large_float_like_x = np.sqrt(np.finfo(x.dtype.as_numpy_dtype()).max)
      safe_grads = tf.where(
          tf.equal(x, 0), large_float_like_x, 0.5 * tf.math.rsqrt(x))
      return grad_ys * safe_grads
    return tf.sqrt(x), grad


def maybe_get_common_dtype(arg_list):
  """Return common dtype of arg_list, or None.

  Args:
    arg_list: an iterable of items which are either `None` or have a `dtype`
      property.

  Returns:
    dtype: The common dtype of items in `arg_list`, or `None` if the list is
      empty or all items are `None`.
  """
  # Note that `all` defaults to `True` if `arg_list` is empty.
  if all(a is None for a in arg_list):
    return None
  return dtype_util.common_dtype(arg_list, tf.float32)
