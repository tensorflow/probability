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
from tensorflow.python.framework import tensor_util  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'Reshape',
]


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
    with tf.compat.v1.name_scope(name, 'reshape',
                                 [event_shape_out, event_shape_in]):
      event_shape_out = tf.convert_to_tensor(
          value=event_shape_out, name='event_shape_out', dtype_hint=tf.int32)
      event_shape_in = tf.convert_to_tensor(
          value=event_shape_in, name='event_shape_in', dtype_hint=tf.int32)

      forward_min_event_ndims_ = event_shape_in.shape.num_elements()
      if forward_min_event_ndims_ is None:
        raise NotImplementedError(
            '`event_shape_in` `size` must be statically known. For dynamic '
            'support, please contact `tfprobability@tensorflow.org`.')

      inverse_min_event_ndims_ = event_shape_out.shape.num_elements()
      if inverse_min_event_ndims_ is None:
        raise NotImplementedError(
            '`event_shape_out` `size` must be statically known. For dynamic '
            'support, please contact `tfprobability@tensorflow.org`.')

      assertions = []
      assertions.extend(_maybe_check_valid_shape(
          event_shape_out, validate_args))
      assertions.extend(_maybe_check_valid_shape(
          event_shape_in, validate_args))

      if assertions:
        with tf.control_dependencies(assertions):
          event_shape_in = tf.identity(
              event_shape_in, name='validated_event_shape_in')
          event_shape_out = tf.identity(
              event_shape_out, name='validated_event_shape_out')

      self._event_shape_in = event_shape_in
      self._event_shape_out = event_shape_out

      super(Reshape, self).__init__(
          forward_min_event_ndims=forward_min_event_ndims_,
          inverse_min_event_ndims=inverse_min_event_ndims_,
          is_constant_jacobian=True,
          validate_args=validate_args,
          name=name or 'reshape')

  def _forward(self, x):
    output_shape, output_tensorshape = _replace_event_shape_in_shape_tensor(
        tf.shape(input=x),
        self._event_shape_in,
        self._event_shape_out,
        self.validate_args)
    y = tf.reshape(x, output_shape)
    y.set_shape(y.shape.merge_with(output_tensorshape))
    return y

  def _inverse(self, y):
    output_shape, output_tensorshape = _replace_event_shape_in_shape_tensor(
        tf.shape(input=y),
        self._event_shape_out,
        self._event_shape_in,
        self.validate_args)
    x = tf.reshape(y, output_shape)
    x.set_shape(x.shape.merge_with(output_tensorshape))
    return x

  def _inverse_log_det_jacobian(self, y):
    return tf.constant(0., dtype=y.dtype)

  def _forward_log_det_jacobian(self, x):
    return tf.constant(0., dtype=x.dtype)

  def _forward_event_shape(self, input_shape):
    return _replace_event_shape_in_tensorshape(
        input_shape,
        self._event_shape_in,
        self._event_shape_out)[0]

  def _inverse_event_shape(self, output_shape):
    return _replace_event_shape_in_tensorshape(
        output_shape,
        self._event_shape_out,
        self._event_shape_in)[0]

  def _forward_event_shape_tensor(self, input_shape):
    return _replace_event_shape_in_shape_tensor(
        input_shape,
        self._event_shape_in,
        self._event_shape_out,
        self.validate_args)[0]

  def _inverse_event_shape_tensor(self, output_shape):
    return _replace_event_shape_in_shape_tensor(
        output_shape,
        self._event_shape_out,
        self._event_shape_in,
        self.validate_args)[0]


def _replace_event_shape_in_shape_tensor(
    input_shape, event_shape_in, event_shape_out, validate_args):
  """Replaces the rightmost dims in a `Tensor` representing a shape.

  Args:
    input_shape: a rank-1 `Tensor` of integers
    event_shape_in: the event shape expected to be present in rightmost dims
      of `shape_in`.
    event_shape_out: the event shape with which to replace `event_shape_in` in
      the rightmost dims of `input_shape`.
    validate_args: Python `bool` indicating whether arguments should
      be checked for correctness.

  Returns:
    output_shape: A rank-1 integer `Tensor` with the same contents as
      `input_shape` except for the event dims, which are replaced with
      `event_shape_out`.
  """
  output_tensorshape, is_validated = _replace_event_shape_in_tensorshape(
      tensor_util.constant_value_as_shape(input_shape),
      event_shape_in,
      event_shape_out)

  # TODO(b/124240153): Remove map(tf.identity, deps) once tf.function
  # correctly supports control_dependencies.
  validation_dependencies = (
      map(tf.identity, (event_shape_in, event_shape_out))
      if validate_args else ())

  if (output_tensorshape.is_fully_defined() and
      (is_validated or not validate_args)):
    with tf.control_dependencies(validation_dependencies):
      output_shape = tf.convert_to_tensor(
          value=output_tensorshape, name='output_shape', dtype_hint=tf.int32)
    return output_shape, output_tensorshape

  with tf.control_dependencies(validation_dependencies):
    event_shape_in_ndims = (tf.size(input=event_shape_in)
                            if event_shape_in.shape.num_elements() is None
                            else event_shape_in.shape.num_elements())
    input_non_event_shape, input_event_shape = tf.split(
        input_shape, num_or_size_splits=[-1, event_shape_in_ndims])

  additional_assertions = []
  if is_validated:
    pass
  elif validate_args:
    # Check that `input_event_shape` and `event_shape_in` are compatible in the
    # sense that they have equal entries in any position that isn't a `-1` in
    # `event_shape_in`. Note that our validations at construction time ensure
    # there is at most one such entry in `event_shape_in`.
    mask = event_shape_in >= 0
    explicit_input_event_shape = tf.boolean_mask(
        tensor=input_event_shape, mask=mask)
    explicit_event_shape_in = tf.boolean_mask(
        tensor=event_shape_in, mask=mask)
    additional_assertions.append(
        tf.compat.v1.assert_equal(
            explicit_input_event_shape,
            explicit_event_shape_in,
            message='Input `event_shape` does not match `event_shape_in`.'))
    # We don't explicitly additionally verify
    # `tf.size(input_shape) > tf.size(event_shape_in)` since `tf.split`
    # already makes this assertion.

  with tf.control_dependencies(additional_assertions):
    output_shape = tf.concat([input_non_event_shape, event_shape_out], axis=0,
                             name='output_shape')

  return output_shape, output_tensorshape


def _replace_event_shape_in_tensorshape(
    input_tensorshape, event_shape_in, event_shape_out):
  """Replaces the event shape dims of a `TensorShape`.

  Args:
    input_tensorshape: a `TensorShape` instance in which to attempt replacing
      event shape.
    event_shape_in: `Tensor` shape representing the event shape expected to
      be present in (rightmost dims of) `tensorshape_in`. Must be compatible
      with the rightmost dims of `tensorshape_in`.
    event_shape_out: `Tensor` shape representing the new event shape, i.e.,
      the replacement of `event_shape_in`,

  Returns:
    output_tensorshape: `TensorShape` with the rightmost `event_shape_in`
      replaced by `event_shape_out`. Might be partially defined, i.e.,
      `TensorShape(None)`.
    is_validated: Python `bool` indicating static validation happened.

  Raises:
    ValueError: if we can determine the event shape portion of
      `tensorshape_in` as well as `event_shape_in` both statically, and they
      are not compatible. "Compatible" here means that they are identical on
      any dims that are not -1 in `event_shape_in`.
  """
  event_shape_in_ndims = event_shape_in.shape.num_elements()
  if input_tensorshape.ndims is None or event_shape_in_ndims is None:
    return tf.TensorShape(None), False  # Not is_validated.

  input_non_event_ndims = input_tensorshape.ndims - event_shape_in_ndims
  if input_non_event_ndims < 0:
    raise ValueError(
        'Input has fewer ndims ({}) than event shape ndims ({}).'.format(
            input_tensorshape.ndims, event_shape_in_ndims))

  input_non_event_tensorshape = input_tensorshape[:input_non_event_ndims]
  input_event_tensorshape = input_tensorshape[input_non_event_ndims:]

  # Check that `input_event_shape_` and `event_shape_in` are compatible in the
  # sense that they have equal entries in any position that isn't a `-1` in
  # `event_shape_in`. Note that our validations at construction time ensure
  # there is at most one such entry in `event_shape_in`.
  event_shape_in_ = tf.get_static_value(event_shape_in)
  is_validated = (input_event_tensorshape.is_fully_defined() and
                  event_shape_in_ is not None)
  if is_validated:
    input_event_shape_ = np.int32(input_event_tensorshape)
    mask = event_shape_in_ >= 0
    explicit_input_event_shape_ = input_event_shape_[mask]
    explicit_event_shape_in_ = event_shape_in_[mask]
    if not all(explicit_input_event_shape_ == explicit_event_shape_in_):
      raise ValueError(
          'Input `event_shape` does not match `event_shape_in`. '
          '({} vs {}).'.format(input_event_shape_, event_shape_in_))

  event_tensorshape_out = tensor_util.constant_value_as_shape(event_shape_out)
  if event_tensorshape_out.ndims is None:
    output_tensorshape = tf.TensorShape(None)
  else:
    output_tensorshape = input_non_event_tensorshape.concatenate(
        event_tensorshape_out)

  return output_tensorshape, is_validated


def _maybe_check_valid_shape(shape, validate_args):
  """Check that a shape Tensor is int-type and otherwise sane."""
  if not shape.dtype.is_integer:
    raise TypeError('{} dtype ({}) should be `int`-like.'.format(
        shape, shape.dtype.name))

  assertions = []

  message = '`{}` rank should be <= 1.'
  if shape.shape.ndims is not None:
    if shape.shape.ndims > 1:
      raise ValueError(message.format(shape))
  elif validate_args:
    assertions.append(tf.compat.v1.assert_less(
        tf.rank(shape), 2, message=message.format(shape)))

  shape_ = tf.get_static_value(shape)

  message = '`{}` elements must have at most one `-1`.'
  if shape_ is not None:
    if sum(shape_ == -1) > 1:
      raise ValueError(message.format(shape))
  elif validate_args:
    assertions.append(tf.compat.v1.assert_less(
        tf.reduce_sum(input_tensor=tf.cast(tf.equal(shape, -1), tf.int32)),
        2,
        message=message.format(shape)))

  message = '`{}` elements must be either positive integers or `-1`.'
  if shape_ is not None:
    if np.any(shape_ < -1):
      raise ValueError(message.format(shape))
  elif validate_args:
    assertions.append(tf.compat.v1.assert_greater(
        shape, -2, message=message.format(shape)))

  return assertions
