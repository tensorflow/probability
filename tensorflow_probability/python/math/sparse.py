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
"""Functions for working with `tf.SparseTensor`."""


import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util

__all__ = [
    'dense_to_sparse',
]


def dense_to_sparse(x, ignore_value=None, name=None):
  """Converts dense `Tensor` to `SparseTensor`, dropping `ignore_value` cells.

  Args:
    x: A `Tensor`.
    ignore_value: Entries in `x` equal to this value will be
      absent from the return `SparseTensor`. If `None`, default value of
      `x` dtype will be used (e.g. '' for `str`, 0 for `int`).
    name: Python `str` prefix for ops created by this function.

  Returns:
    sparse_x: A `tf.SparseTensor` with the same shape as `x`.

  Raises:
    ValueError: when `x`'s rank is `None`.
  """
  # Copied (with modifications) from:
  # tensorflow/contrib/layers/python/ops/sparse_ops.py.
  with tf.name_scope(name or 'dense_to_sparse'):
    x = tf.convert_to_tensor(x, name='x')
    if ignore_value is None:
      if dtype_util.base_dtype(x.dtype) == tf.string:
        # Exception due to TF strings are converted to numpy objects by default.
        ignore_value = ''
      else:
        ignore_value = dtype_util.as_numpy_dtype(x.dtype)(0)
      ignore_value = tf.cast(ignore_value, x.dtype, name='ignore_value')
    indices = tf.where(tf.not_equal(x, ignore_value), name='indices')
    return tf.SparseTensor(
        indices=indices,
        values=tf.gather_nd(x, indices, name='values'),
        dense_shape=tf.shape(x, out_type=tf.int64, name='dense_shape'))
