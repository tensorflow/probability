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
"""Transpose bijector."""

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util

__all__ = [
    'Transpose',
]


class Transpose(bijector.AutoCompositeTensorBijector):
  """Compute `Y = g(X) = transpose_rightmost_dims(X, rightmost_perm)`.

  This bijector is semantically similar to `tf.transpose` except that it
  transposes only the rightmost "event" dimensions. That is, unlike
  `tf.transpose` the `perm` argument is itself a permutation of
  `tf.range(rightmost_transposed_ndims)` rather than `tf.range(tf.rank(x))`,
  i.e., users specify the (rightmost) dimensions to permute, not all dimensions.

  The actual (forward) transformation is:

  ```python
  def forward(x, perm):
    sample_batch_ndims = tf.rank(x) - tf.size(perm)
    perm = tf.concat([
        tf.range(sample_batch_ndims),
        sample_batch_ndims + perm,
    ], axis=0)
    return tf.transpose(x, perm)
  ```

  #### Examples

  ```python
  tfp.bijectors.Transpose(perm=[1, 0]).forward(
      [
        [[1, 2],
         [3, 4]],
        [[5, 6],
         [7, 8]],
      ])
  # ==>
  #  [
  #    [[1, 3],
  #     [2, 4]],
  #    [[5, 7],
  #     [6, 8]],
  #  ]

  # Using `rightmost_transposed_ndims=2` means this bijector has the same
  # semantics as `tf.matrix_transpose`.
  tfp.bijectors.Transpose(rightmost_transposed_ndims=2).inverse(
      [
        [[1, 3],
         [2, 4]],
        [[5, 7],
         [6, 8]],
      ])
  # ==>
  #  [
  #    [[1, 2],
  #     [3, 4]],
  #    [[5, 6],
  #     [7, 8]],
  #  ]
  ```

  """

  def __init__(self, perm=None, rightmost_transposed_ndims=None,
               validate_args=False, name='transpose'):
    """Instantiates the `Transpose` bijector.

    Args:
      perm: Positive `int32` vector-shaped `Tensor` representing permutation of
        rightmost dims (for forward transformation).  Note that the `0`th index
        represents the first of the rightmost dims and the largest value must be
        `rightmost_transposed_ndims - 1` and corresponds to `tf.rank(x) - 1`.
        Only one of `perm` and `rightmost_transposed_ndims` can (and must) be
        specified. The number of elements in a permutation must have a value
        that can be determined statically.
        Default value:
        `tf.range(start=rightmost_transposed_ndims, limit=-1, delta=-1)`.
      rightmost_transposed_ndims: Positive `int32` scalar-shaped `Tensor`
        representing the number of rightmost dimensions to permute.
        Only one of `perm` and `rightmost_transposed_ndims` can (and must) be
        specified. If `rightmost_transposed_ndims` is specified, the rightmost
        dims are reversed. This argument must have a value that can be
        determined statically.
        Default value: `tf.size(perm)`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.

    Raises:
      ValueError: if both or neither `perm` and `rightmost_transposed_ndims` are
        specified.
      NotImplementedError: if `rightmost_transposed_ndims` is not known prior to
        graph execution.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      # We need to determine `forward_min_event_ndims` statically, which
      # requires that we know `rightmost_transposed_ndims` statically.
      # So the corresponding assertions go here rather than in
      # `_parameter_control_dependencies`
      if (rightmost_transposed_ndims is None) == (perm is None):
        raise ValueError('Must specify exactly one of '
                         '`rightmost_transposed_ndims` and `perm`.')
      if rightmost_transposed_ndims is not None:
        rightmost_transposed_ndims = tensor_util.convert_nonref_to_tensor(
            rightmost_transposed_ndims,
            dtype_hint=np.int32, as_shape_tensor=True)
        if not dtype_util.is_integer(rightmost_transposed_ndims.dtype):
          raise TypeError('`rightmost_transposed_ndims` must be integer type.')
        rightmost_transposed_ndims_ = tf.get_static_value(
            rightmost_transposed_ndims)
        if rightmost_transposed_ndims_ is None:
          raise NotImplementedError('`rightmost_transposed_ndims` must be '
                                    'known prior to graph execution.')
        msg = '`rightmost_transposed_ndims` must be non-negative.'
        if rightmost_transposed_ndims_ < 0:
          raise ValueError(msg[:-1] + ', saw: {}.'.format(
              rightmost_transposed_ndims_))

      else:  # perm is not None:
        perm = tensor_util.convert_nonref_to_tensor(
            perm, dtype_hint=np.int32, name='perm', as_shape_tensor=True)
        rightmost_transposed_ndims_ = tf.get_static_value(ps.size(perm))

      # TODO(b/110828604): If bijector base class ever supports dynamic
      # `min_event_ndims`, then this class already works dynamically and the
      # following five lines can be removed. In this case, we may have to
      # deprecate doing work in the constructor to avoid graph cycles when
      # `validate_args==True`.
      if rightmost_transposed_ndims_ is None:
        raise NotImplementedError('`rightmost_transposed_ndims` must be '
                                  'known prior to graph execution.')
      else:
        rightmost_transposed_ndims_ = int(rightmost_transposed_ndims_)

      self._perm = perm
      self._rightmost_transposed_ndims = rightmost_transposed_ndims
      self._initial_rightmost_transposed_ndims = rightmost_transposed_ndims_
      super(Transpose, self).__init__(
          forward_min_event_ndims=rightmost_transposed_ndims_,
          is_constant_jacobian=True,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict(
        rightmost_transposed_ndims=(
            parameter_properties.ShapeParameterProperties(is_preferred=False)),
        perm=parameter_properties.ShapeParameterProperties())

  @property
  def perm(self):
    return self._perm

  @property
  def rightmost_transposed_ndims(self):
    return self._rightmost_transposed_ndims

  @property
  def _is_permutation(self):
    return True

  def _is_increasing(self):
    if self.forward_min_event_ndims == 0:
      return True
    raise NotImplementedError(
        '`_is_increasing` not supported unless Transpose is no-op.')

  def _get_perm(self):
    if self.perm is None:
      perm_start = (
          distribution_util.prefer_static_value(
              self.rightmost_transposed_ndims) - 1)
      return ps.range(start=perm_start, limit=-1, delta=-1, name='perm')
    return self.perm

  def _get_rightmost_transposed_ndims(self):
    if self.rightmost_transposed_ndims is None:
      return ps.size(self.perm)
    return self.rightmost_transposed_ndims

  def _forward(self, x):
    return self._transpose(x, self._get_perm())

  def _event_shape(self, shape, static_perm_to_shape):
    """Helper for _forward and _inverse_event_shape."""
    rightmost_ = tf.get_static_value(self._get_rightmost_transposed_ndims())
    if tensorshape_util.rank(shape) is None or rightmost_ is None:
      return tf.TensorShape(None)
    if tensorshape_util.rank(shape) < rightmost_:
      raise ValueError('Invalid shape: min event ndims={} but got {}'.format(
          rightmost_, shape))
    perm_ = tf.get_static_value(self._get_perm(), partial=True)
    if perm_ is None:
      return shape[:tensorshape_util.rank(shape) - rightmost_].concatenate(
          [None] * int(rightmost_))
    # We can use elimination to reidentify a single None dimension.
    if sum(p is None for p in perm_) == 1:
      present = np.argsort([-1 if p is None else p for p in perm_])
      for i, p in enumerate(present[1:]):  # The -1 sorts to position 0.
        if i != p:
          perm_ = [i if p is None else p for p in perm_]
          break
    return shape[:tensorshape_util.rank(shape) - rightmost_].concatenate(
        static_perm_to_shape(shape[tensorshape_util.rank(shape) - rightmost_:],
                             perm_))

  def _forward_event_shape(self, input_shape):
    def static_perm_to_shape(subshp, perm):
      return tf.TensorShape(
          [None if p is None else subshp[p] for p in perm])
    return self._event_shape(input_shape, static_perm_to_shape)

  def _forward_event_shape_tensor(self, input_shape):
    perm = self._make_perm(ps.size(input_shape), self._get_perm())
    return ps.gather(input_shape, perm)

  def _inverse(self, y):
    return self._transpose(y, ps.argsort(self._get_perm()))

  def _inverse_event_shape(self, output_shape):
    def static_perm_to_shape(subshp, perm):
      result = [None] * len(perm)
      for i, p in enumerate(perm):
        if p is not None:
          result[p] = subshp[i]
      return tf.TensorShape(result)
    return self._event_shape(output_shape, static_perm_to_shape)

  def _inverse_event_shape_tensor(self, output_shape):
    perm = self._make_perm(tf.size(output_shape), tf.argsort(self._get_perm()))
    return tf.gather(output_shape, perm)

  def _inverse_log_det_jacobian(self, y):
    return tf.constant(0, dtype=y.dtype)

  def _forward_log_det_jacobian(self, x):
    return tf.constant(0, dtype=x.dtype)

  def _make_perm(self, x_rank, perm):
    sample_batch_ndims = (
        distribution_util.prefer_static_value(x_rank) -
        distribution_util.prefer_static_value(
            self._get_rightmost_transposed_ndims()))
    dtype = perm.dtype
    perm = ps.concat([
        ps.range(ps.cast(sample_batch_ndims, dtype)),
        ps.cast(
            sample_batch_ndims + distribution_util.prefer_static_value(perm),
            dtype),
    ],
                     axis=0)
    return perm

  def _transpose(self, x, perm):
    perm = self._make_perm(ps.rank(x), perm)
    return tf.transpose(a=x, perm=perm)

  def _parameter_control_dependencies(self, is_init):
    assertions = []
    if is_init != tensor_util.is_ref(self.rightmost_transposed_ndims):
      if self.validate_args:
        assertions += _maybe_validate_rightmost_transposed_ndims(
            self._initial_rightmost_transposed_ndims,
            self._get_rightmost_transposed_ndims(), self.validate_args)

    if is_init != tensor_util.is_ref(self.perm):
      if self.validate_args:
        assertions += _maybe_validate_perm(
            self._initial_rightmost_transposed_ndims,
            self._get_perm(), self.validate_args)

    return assertions


def _maybe_validate_rightmost_transposed_ndims(
    initial_rightmost_transposed_ndims,
    rightmost_transposed_ndims, validate_args, name=None):
  """Checks that `rightmost_transposed_ndims` is valid."""
  with tf.name_scope(name or 'maybe_validate_rightmost_transposed_ndims'):
    assertions = []

    if tensorshape_util.rank(rightmost_transposed_ndims.shape) is not None:
      if tensorshape_util.rank(rightmost_transposed_ndims.shape) != 0:
        raise ValueError('`rightmost_transposed_ndims` must be a scalar, '
                         'saw rank: {}.'.format(
                             tensorshape_util.rank(
                                 rightmost_transposed_ndims.shape)))
    elif validate_args:
      assertions += [
          assert_util.assert_rank(rightmost_transposed_ndims, 0),
          assert_util.assert_equal(
              rightmost_transposed_ndims,
              initial_rightmost_transposed_ndims,
              message='`rightmost_transposed_ndims` must not change '
                      'from the value set when the `Transpose` '
                      'bijector was constructed.')]

    rightmost_transposed_ndims_ = tf.get_static_value(
        rightmost_transposed_ndims)
    msg = '`rightmost_transposed_ndims` must be non-negative.'
    if rightmost_transposed_ndims_ is not None:
      if rightmost_transposed_ndims_ < 0:
        raise ValueError(msg[:-1] + ', saw: {}.'.format(
            rightmost_transposed_ndims_))
    elif validate_args:
      assertions += [
          assert_util.assert_non_negative(
              rightmost_transposed_ndims, message=msg)
      ]

    return assertions


def _maybe_validate_perm(
    initial_rightmost_transposed_ndims,
    perm, validate_args, name=None):
  """Checks that `perm` is valid."""
  with tf.name_scope(name or 'maybe_validate_perm'):
    assertions = []
    if not dtype_util.is_integer(perm.dtype):
      raise TypeError('`perm` must be integer type')

    msg = '`perm` must be a vector.'
    if tensorshape_util.rank(perm.shape) is not None:
      if tensorshape_util.rank(perm.shape) != 1:
        raise ValueError(
            msg[:-1] +
            ', saw rank: {}.'.format(tensorshape_util.rank(perm.shape)))
    elif validate_args:
      assertions += [
          assert_util.assert_rank(perm, 1, message=msg),
          assert_util.assert_equal(
              tf.size(perm),
              initial_rightmost_transposed_ndims,
              message='The number of elements of `perm` must not '
                      'change from the value set when the `Transpose` '
                      'bijector was constructed.')]

    perm_ = tf.get_static_value(perm)
    msg = '`perm` must be a valid permutation vector.'
    if perm_ is not None:
      if not np.all(np.arange(np.size(perm_)) == np.sort(perm_)):
        raise ValueError(msg[:-1] + ', saw: {}.'.format(perm_))
    elif validate_args:
      assertions += [
          assert_util.assert_equal(
              ps.sort(perm), ps.range(ps.size(perm, out_type=perm.dtype)),
              message=msg)
      ]

    return assertions
