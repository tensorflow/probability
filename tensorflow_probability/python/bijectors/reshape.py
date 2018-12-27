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
"""Reshape bijectors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow.python.framework import tensor_util


__all__ = [
    'Reshape',
]


def _ndims_from_shape(shape):
  return tf.shape(shape)[0]


class Reshape(bijector.Bijector):
  """Reshapes the `event_shape` of a `Tensor`.

  The semantics generally follow that of `tf.reshape()`, with
  a few differences:

  * The user must provide both the input and output shape, so that
    the transformation can be inverted. If an input shape is not
    specified, the default assumes a vector-shaped input, i.e.,
    event_shape_in = (-1,).
  * The `Reshape` bijector automatically broadcasts over the leftmost
    dimensions of its input (`sample_shape` and `batch_shape`); only
    the rightmost `event_ndims_in` dimensions are reshaped. The
    number of dimensions to reshape is inferred from the provided
    `event_shape_in` (`event_ndims_in = len(event_shape_in)`).

  Example usage:

  ```python
  r = tfp.bijectors.Reshape(event_shape_out=[1, -1])

  r.forward([3., 4.])    # shape [2]
  # ==> [[3., 4.]]       # shape [1, 2]

  r.forward([[1., 2.], [3., 4.]])  # shape [2, 2]
  # ==> [[[1., 2.]],
  #      [[3., 4.]]]   # shape [2, 1, 2]

  r.inverse([[3., 4.]])  # shape [1,2]
  # ==> [3., 4.]         # shape [2]

  r.forward_log_det_jacobian(any_value)
  # ==> 0.

  r.inverse_log_det_jacobian(any_value)
  # ==> 0.
  ```

  Note: we had to make a tricky-to-describe policy decision, which we attempt to
  summarize here. At instantiation time and class method invocation time, we
  validate consistency of class-level and method-level arguments. Note that
  since the class-level arguments may be unspecified until graph execution time,
  we had the option of deciding between two validation policies. One was,
  roughly, "the earliest, most statically specified arguments take precedence".
  The other was "method-level arguments must be consistent with class-level
  arguments". The former policy is in a sense more optimistic about user intent,
  and would enable us, in at least one particular case [1], to perform
  additional inference about resulting shapes. We chose the latter policy, as it
  is simpler to implement and a bit easier to articulate.

  [1] The case in question is exemplified in the following snippet:

  ```python
  bijector = tfp.bijectors.Reshape(
    event_shape_out=tf.placeholder(dtype=tf.int32, shape=[1]),
    event_shape_in= tf.placeholder(dtype=tf.int32, shape=[3]),
    validate_args=True)

  bijector.forward_event_shape(tf.TensorShape([5, 2, 3, 7]))
  # Chosen policy    ==> (5, None)
  # Alternate policy ==> (5, 42)
  ```

  In the chosen policy, since we don't know what `event_shape_in/out` are at the
  time of the call to `forward_event_shape`, we simply fill in everything we
  *do* know, which is that the last three dims will be replaced with
  "something".

  In the alternate policy, we would assume that the intention must be to reshape
  `[5, 2, 3, 7]` such that the last three dims collapse to one, which is only
  possible if the resulting shape is `[5, 42]`.

  Note that the above is the *only* case in which we could do such inference; if
  the output shape has more than 1 dim, we can't infer anything. E.g., we would
  have

  ```python
  bijector = tfp.bijectors.Reshape(
    event_shape_out=tf.placeholder(dtype=tf.int32, shape=[2]),
    event_shape_in= tf.placeholder(dtype=tf.int32, shape=[3]),
    validate_args=True)

  bijector.forward_event_shape(tf.TensorShape([5, 2, 3, 7]))
  # Either policy ==> (5, None, None)
  ```

  """

  def __init__(self, event_shape_out, event_shape_in=(-1,),
               validate_args=False, name=None):
    """Creates a `Reshape` bijector.

    Args:
      event_shape_out: An `int`-like vector-shaped `Tensor`
        representing the event shape of the transformed output.
      event_shape_in: An optional `int`-like vector-shape `Tensor`
        representing the event shape of the input. This is required in
        order to define inverse operations; the default of (-1,)
        assumes a vector-shaped input.
      validate_args: Python `bool` indicating whether arguments should
        be checked for correctness.
      name: Python `str`, name given to ops managed by this object.

    Raises:
      TypeError: if either `event_shape_in` or `event_shape_out` has
        non-integer `dtype`.
      ValueError: if either of `event_shape_in` or `event_shape_out`
       has non-vector shape (`rank > 1`), or if their sizes do not
       match.
    """
    with tf.name_scope(
        name, 'reshape', values=[event_shape_out, event_shape_in]):

      event_shape_out = tf.convert_to_tensor(
          event_shape_out, name='event_shape_out', preferred_dtype=tf.int32)
      event_shape_in = tf.convert_to_tensor(
          event_shape_in, name='event_shape_in', preferred_dtype=tf.int32)

      forward_min_event_ndims = tensor_util.constant_value(
          tf.size(event_shape_in))
      if forward_min_event_ndims is None:
        raise NotImplementedError('Rank of `event_shape_in` currently must be '
                                  'statically known. Contact '
                                  '`tfprobability@tensorflow.org` if this is '
                                  'a problem for your use case.')

      inverse_min_event_ndims = tensor_util.constant_value(
          tf.size(event_shape_out))
      if inverse_min_event_ndims is None:
        raise NotImplementedError('Rank of `event_shape_out` currently must be '
                                  'statically known. Contact '
                                  '`tfprobability@tensorflow.org` if this is '
                                  'a problem for your use case.')

      assertions = []
      assertions.extend(self._maybe_check_valid_shape(
          event_shape_out, validate_args))
      assertions.extend(self._maybe_check_valid_shape(
          event_shape_in, validate_args))

      self._assertions = assertions
      self._event_shape_in = event_shape_in
      self._event_shape_out = event_shape_out

      super(Reshape, self).__init__(
          forward_min_event_ndims=int(forward_min_event_ndims),
          inverse_min_event_ndims=int(inverse_min_event_ndims),
          is_constant_jacobian=True,
          validate_args=validate_args,
          name=name or 'reshape')

  def _maybe_check_valid_shape(self, shape, validate_args):
    """Check that a shape Tensor is int-type and otherwise sane."""
    if not shape.dtype.is_integer:
      raise TypeError('{} dtype ({}) should be `int`-like.'.format(
          shape, shape.dtype.name))

    assertions = []

    ndims = tf.rank(shape)
    ndims_ = tf.contrib.util.constant_value(ndims)
    if ndims_ is not None and ndims_ > 1:
      raise ValueError('`{}` rank ({}) should be <= 1.'.format(
          shape, ndims_))
    elif validate_args:
      assertions.append(
          tf.assert_less_equal(
              ndims, 1, message='`{}` rank should be <= 1.'.format(shape)))

    # Note, we might be inclined to use tensor_util.constant_value_as_shape
    # here, but that method coerces negative values into `None`s, rendering the
    # checks we do below impossible.
    shape_tensor_ = tf.contrib.util.constant_value(shape)
    if shape_tensor_ is not None:
      es = np.int32(shape_tensor_)
      if sum(es == -1) > 1:
        raise ValueError(
            '`{}` must have at most one `-1` (given {})'
            .format(shape, es))
      if np.any(es < -1):
        raise ValueError(
            '`{}` elements must be either positive integers or `-1`'
            '(given {}).'
            .format(shape, es))
    elif validate_args:
      assertions.extend([
          tf.assert_less_equal(
              tf.reduce_sum(tf.cast(tf.equal(shape, -1), tf.int32)),
              1,
              message='`{}` elements must have at most one `-1`.'
              .format(shape)),
          tf.assert_greater_equal(
              shape,
              -1,
              message='`{}` elements must be either positive integers or `-1`.'
              .format(shape)),
      ])
    return assertions

  def _maybe_validate_event_shape(self, event_shape, reference_event_shape):
    """Add validation checks to graph if `self.validate_args` is `True`."""
    if not self.validate_args:
      return []
    # Similarly to the static case, we test for compatibility between
    # `event_shape` and `reference_event_shape`, where compatibility means the
    # shapes are equal in all positions except those in which
    # `reference_event_shape` is `-1` (there can be at most one of these).

    # Get a boolean mask of elements with explicitly known shape values (not
    # `-1` or `None`), and set the resulting shape explicitly since we know it
    # is rank-1 and `tf.boolean_mask` will check this as a precondition.
    mask = reference_event_shape >= 0
    mask.set_shape([reference_event_shape.shape.num_elements()])
    explicitly_known_event_shape_dims = tf.boolean_mask(event_shape, mask)
    explicitly_known_reference_event_shape_dims = tf.boolean_mask(
        reference_event_shape, mask)
    return [tf.assert_equal(
        explicitly_known_event_shape_dims,
        explicitly_known_reference_event_shape_dims,
        message='Input `event_shape` does not match `reference_event_shape`.')]

  def _forward(self, x):
    with tf.control_dependencies(self._assertions):
      shape_, shape = self._compute_shape(
          x, self._event_shape_in, self._event_shape_out)
      y = tf.reshape(x, shape)
      y.set_shape(shape_)
      return y

  def _inverse(self, y):
    with tf.control_dependencies(self._assertions):
      shape_, shape = self._compute_shape(
          y, self._event_shape_out, self._event_shape_in)
      x = tf.reshape(y, shape)
      x.set_shape(shape_)
      return x

  def _inverse_log_det_jacobian(self, y):
    with tf.control_dependencies(self._assertions):
      return tf.constant(0., dtype=y.dtype)

  def _forward_log_det_jacobian(self, x):
    with tf.control_dependencies(self._assertions):
      return tf.constant(0., dtype=x.dtype)

  def _forward_event_shape(self, input_shape):
    with tf.control_dependencies(self._assertions):
      return self._replace_event_shape_in_tensorshape(
          input_shape, self._event_shape_in, self._event_shape_out)

  def _inverse_event_shape(self, output_shape):
    with tf.control_dependencies(self._assertions):
      return self._replace_event_shape_in_tensorshape(
          output_shape, self._event_shape_out, self._event_shape_in)

  def _forward_event_shape_tensor(self, input_shape):
    with tf.control_dependencies(self._assertions):
      return self._replace_event_shape_in_shape_tensor(
          input_shape, self._event_shape_in, self._event_shape_out)

  def _inverse_event_shape_tensor(self, output_shape):
    with tf.control_dependencies(self._assertions):
      return self._replace_event_shape_in_shape_tensor(
          output_shape, self._event_shape_out, self._event_shape_in)

  def _compute_shape(
      self, x, event_shape_in, event_shape_out):
    """Compute the result we'd get from reshaping `x`'s event shape.

    Args:
      x: a `Tensor` whose event shape is to be transformed.
      event_shape_in: a `Tensor` representing the event shape of the `x`. We
        will attempt to validate that this event shape actually matches `x`,
        insofar as is possible at the time of the call, or with in-graph
        assertions if `self.validate_args` is `True`.
      event_shape_out: the event shape to supplant `event_shape_in` with in the
        shape of `x`. We will attempt to do this while preserving as much shape
        information as possible in the output.

    Returns:
      new_shape_: `TensorShape` resulting from swapping `x`'s event shape.
      new_shape: `Tensor` representing shape resulting from swapping `x`'s event
        shape.

    Raises:
      ValueError: if the rightmost dims of `x` are not compatible with
        `event_shape_in`. Note that `event_shape_in` might contain special
        values, like `-1`, in which case validation does not require exact
        equality. If the shapes are not statically known, and
        `self.validate_args` is `True`, then equivalent assertions are added to
        the graph as control dependencies of the returned `Tensor`.
    """
    # Attempt to compute the new shape as a static `TensorShape`.
    new_shape_ = self._replace_event_shape_in_tensorshape(
        x.shape, event_shape_in, event_shape_out)

    # Compute the new shape as a `Tensor`.
    new_shape = self._replace_event_shape_in_shape_tensor(
        tf.shape(x), event_shape_in, event_shape_out)

    return new_shape_, new_shape

  def _replace_event_shape_in_tensorshape(
      self, tensorshape_in, event_shape_in, event_shape_out):
    """Replaces the event shape dims of a `TensorShape`.

    Args:
      tensorshape_in: a `TensorShape` instance in which to attempt replacing
        event shape.
      event_shape_in: `Tensor` containing the event shape expected to be present
        in (rightmost dims of) `tensorshape_in`. Must be compatible with
        the rightmost dims of `tensorshape_in`.
      event_shape_out: `Tensor` containing the shape values with which to
        replace `event_shape_in` in `tensorshape_in`.

    Returns:
      tensorshape_out_: A `TensorShape` with the event shape replaced, if doing
        so is possible given the statically known shape data in
        `tensorshape_in` and `event_shape_in`. Else, `tf.TensorShape(None)`.

    Raises:
      ValueError: if we can determine the event shape portion of
        `tensorshape_in` as well as `event_shape_in` both statically, and they
        are not compatible. "Compatible" here means that they are identical on
        any dims that are not -1 in `event_shape_in`.
    """
    # Default to returning unknown shape
    tensorshape_out_ = tf.TensorShape(None)

    event_ndims_in_ = event_shape_in.shape.num_elements()
    if (event_ndims_in_ is not None and
        self._is_event_shape_fully_defined(tensorshape_in, event_ndims_in_)):
      ndims_ = tensorshape_in.ndims
      sample_and_batch_shape = tensorshape_in[:(ndims_ - event_ndims_in_)]
      event_shape_ = np.int32(tensorshape_in[ndims_ - event_ndims_in_:])

      # If both `event_shape_in` and the event shape dims of `tensorshape_in`
      # are statically known, we can statically validate the event shape.
      #
      # If `event_shape_in` is not statically known, we can only add runtime
      # validations to the graph (if enabled).
      event_shape_in_ = tf.contrib.util.constant_value(event_shape_in)
      if event_shape_in_ is not None:
        # Check that `event_shape_` and `event_shape_in` are compatible in
        # the sense that they have equal entries in any position that isn't a
        # `-1` in `event_shape_in`. Note that our validations at construction
        # time ensure there is at most one such entry in `event_shape_in`.
        event_shape_specified_ = event_shape_[event_shape_in_ >= 0]
        event_shape_in_specified_ = event_shape_in_[event_shape_in_ >= 0]
        if not all(event_shape_specified_ == event_shape_in_specified_):
          raise ValueError(
              'Input `event_shape` does not match `event_shape_in`. ' +
              '({} vs {}).'.format(event_shape_, event_shape_in_))
      else:
        with tf.control_dependencies(self._maybe_validate_event_shape(
            event_shape_, event_shape_in)):
          event_shape_out = tf.identity(event_shape_out)

      tensorshape_out_ = sample_and_batch_shape.concatenate(
          tensor_util.constant_value_as_shape(event_shape_out))

    return tensorshape_out_

  def _replace_event_shape_in_shape_tensor(
      self, shape_in, event_shape_in, event_shape_out):
    """Replaces the rightmost dims in a `Tensor` representing a shape.

    Args:
      shape_in: a rank-1 `Tensor` of integers
      event_shape_in: the event shape expected to be present in (rightmost dims
        of) `shape_in`.
      event_shape_out: the event shape with which to replace `event_shape_in` in
        `shape_in`

    Returns:
      shape_out: A rank-1 integer `Tensor` with the same contents as `shape_in`
        except for the event dims, which are replaced with `event_shape_out`.
    """
    # If possible, extract statically known `TensorShape` and transform that.
    tensorshape = tensor_util.constant_value_as_shape(shape_in)
    if tensorshape is not None and tensorshape.is_fully_defined():
      shape_out_ = self._replace_event_shape_in_tensorshape(
          tensorshape, event_shape_in, event_shape_out)
      if shape_out_.is_fully_defined():
        shape_out = tf.convert_to_tensor(
            shape_out_.as_list(), preferred_dtype=tf.int32)
        return shape_out

    # If not possible statically, use fully dynamic reshaping.
    rank = _ndims_from_shape(shape_in)
    event_ndims = _ndims_from_shape(event_shape_in)

    event_shape = shape_in[rank - event_ndims:]
    with tf.control_dependencies(self._maybe_validate_event_shape(
        event_shape, event_shape_in)):
      sample_and_batch_shape = shape_in[:(rank - event_ndims)]
      shape_out = tf.concat([sample_and_batch_shape, event_shape_out], axis=0)
      return shape_out

  def _is_event_shape_fully_defined(self, tensorshape, event_ndims):
    """Check if `tensorshape` has rightmost dims fully defined.

    Args:
      tensorshape: a `TensorShape`
      event_ndims: python integer number of event dimensions to check

    Returns:
      `True` if and only if the following hold:
        - `tensorshape.ndims` is statically known
        - rightmost `event_ndims` dims of `tensorshape` are fully defined.

    Raises:
      ValueError: if input ndims is smaller than `event_ndims`.
    """
    if tensorshape.ndims is None:
      return False
    if tensorshape.ndims < event_ndims:
      raise ValueError(
          ('Input has fewer dims (ndims={}) than given event shape '
           '(ndims={})').format(tensorshape.ndims, event_ndims))
    event_shape = tensorshape[tensorshape.ndims - event_ndims:]
    return event_shape.is_fully_defined()
