# Copyright 2020 The TensorFlow Probability Authors.
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
"""Utilities for testing numerical accuracy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

# Dependency imports
from absl import logging
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.math import gradient

__all__ = [
    'floating_tensor_to_f32',
    'floating_tensor_to_f64',
    'relerr',
    'relative_error_at',
    'wrong_bits',
    'rounding_error',
    'condition_number_one_input',
    'error_due_to_ill_conditioning',
    'excess_wrong_bits',
]


def floating_tensor_to_f32(x):
  """Cast x to float32 if a floating-point Tensor, else return it.

  This is meant to be used with `tf.nest.map_structure`, or other
  situations where it may not be obvious whether an object is a Tensor
  or not, or has floating dtype or not.

  Args:
    x: The object to be cast or left be.

  Returns:
    x: x, either cast or left be.

  """
  if tf.is_tensor(x) and dtype_util.is_floating(x.dtype):
    return tf.cast(x, dtype=tf.float32)
  else:
    return x


def floating_tensor_to_f64(x):
  """Cast x to float64 if a floating-point Tensor, else return it.

  This is meant to be used with `tf.nest.map_structure`, or other
  situations where it may not be obvious whether an object is a Tensor
  or not, or has floating dtype or not.

  Args:
    x: The object to be cast or left be.

  Returns:
    x: x, either cast or left be.

  """
  if tf.is_tensor(x) and dtype_util.is_floating(x.dtype):
    return tf.cast(x, dtype=tf.float64)
  else:
    return x


def relerr(result, truth):
  """Returns the relative error of `result` relative `truth`.

  The relative error is defined as

    |result - truth|
    ----------------
        |truth|

  The computation of that difference and ratio are done in 64-bit
  precision.

  Args:
    result: Tensor of values whose deviation to assess.
    truth: Tensor of values presumed correct.  Must broadcast with
      `result`.

  Returns:
    err: Float64 Tensor of elementwise relative error values.

  """
  result = tf.cast(result, dtype=tf.float64)
  truth = tf.cast(truth, dtype=tf.float64)
  err = tf.math.abs((result - truth) / truth)
  # If truth is 0 (in 64 bits!), the previous expression will give infinite
  # relative error for non-zero result (which is correct), and nan relative
  # error for zero result (which is incorrect).  This tf.where fixes that.
  return tf.where(result == truth, tf.constant(0., dtype=tf.float64), err)


def relative_error_at(f, *args):
  """Returns the relative error of `f` on `args` in float32.

  This function assumes that numerical error when computing `f` in float64 is
  negligible.  For this to work correctly, `f` needs to be _dtype-polymorphic_:
  the dtype in which computations internal to `f` are performed should match the
  dtype of the arguments of `f`.

  Note that we are looking for errors due to the implementation of
  `f`, not due to rounding of its inputs or outputs.  Therefore the
  arguments and the result are canonicalized to being representable
  exactly in 32-bit arithmetic.

  Args:
    f: Function whose accuracy to evaluate.  Must be dtype-polymorphic.
    *args: Arguments at which to test the accuracy of `f`.

  Returns:
    relerr: The relative error when computing `f(*args)` in float32.

  Raises:
    ValueError: If `f` is found not to be dtype-polymorphic.

  """
  args_32 = tf.nest.map_structure(floating_tensor_to_f32, args)
  logging.vlog(1, '32-bit arguments: %s', args_32)
  args_64 = tf.nest.map_structure(floating_tensor_to_f64, args_32)
  truth = f(*args_64)
  logging.vlog(1, 'Correct answer: %s', truth)
  if truth.dtype != tf.float64:
    raise ValueError('Evaluating on {} produced non-64-bit result {}'.format(
        args_64, truth))
  truth_32 = floating_tensor_to_f32(truth)
  logging.vlog(1, 'Correct answer representable in 32 bits: %s', truth_32)
  ans_32 = f(*args_32)
  logging.vlog(1, 'Answer computed in 32 bits: %s', ans_32)
  if ans_32.dtype != tf.float32:
    raise ValueError('Evaluating on {} produced non-32-bit result {}'.format(
        args_32, ans_32))
  return relerr(ans_32, truth_32)


def wrong_bits(rel_err):
  """Returns how many low-order bits `rel_err` corresponds to, in float32.

  In other words, if you see relative error `rel_err` on a
  (non-denomal) float-32 quantity, `wrong_bits(rel_err)` is the number
  of low-order bits that are wrong.

  Args:
    rel_err: Floating-point Tensor of relative error values.

  Returns:
    wrong: Tensor of elementwise corresponding wrong bits values, of the
      same dtype as `rel_err`.
  """
  log2 = tf.math.log(tf.constant(2.0, dtype=rel_err.dtype))
  # Negative wrong bits can only be an accident
  return tf.maximum(tf.math.log(tf.math.abs(rel_err)) / log2 + 24, 0)


def rounding_error(x, denormal_correction=True):
  """Compute the maximum absolute 32-bit rounding error possible in `x`.

  That is, we compute
    max_y |y - x|  among y where x = round_in_32_bits(y)

  All internal computations are carried out in float64 so this
  function is itself accurate.

  TensorFlow flushes denormals to zero, so there's a rounding error
  cliff at the smallest normal positive float: anything that rounded
  to 0 could have been as large as `tiny`; but anything that rounded
  to anything positive must have been normal.  The
  `denormal_correction` flag controls whether this is taken into
  account.

  Args:
    x: Tensor of `x` values to compute 32-bit rounding error for.  Is
      internally cast to float64 regardless of input dtype.
    denormal_correction: Python bool.  Denormal flushing is accounted
      for if `True`; otherwise rounding is assumed to affect only the
      bits that fall off the mantissa.

  Returns:
    err: float64 Tensor of elementwise maximum rounding errors in `x`.
  """
  resolution = tf.math.pow(tf.constant(2.0, dtype=tf.float64), -24)
  tiny32 = tf.constant(np.finfo(np.float32).tiny, dtype=tf.float64)
  # minute32 approximates (in 64 bits) the smallest positive 32-bit
  # float including denormals.  There is no way for 32-bit rounding
  # error to be less than this.
  minute32 = tiny32 * resolution
  x = tf.math.abs(tf.cast(x, dtype=tf.float64))
  if denormal_correction:
    return tf.where(x >= tiny32, x * resolution, tiny32)
  else:
    return tf.maximum(x * resolution, minute32)


def condition_number_one_input(result, argument, derivative):
  """Returns the condition number at one scalar argument.

  Namely, the error in the output induced by rounding the input to a 32-bit
  float, divided by the error in the output induced by rounding the output
  itself to a 32-bit float.

  Over most of the float-point range, this is just

    |x||f'(x)|
   ------------
      |f(x)|

  but some care needs to be taken when x or f(x) may round to 0.

  Computations internal to this function are done in float64 to assure
  their own accuracy.

  Caveat: If `f` uses `stop_gradient` or similar internally, condition
  numbers estimated by this function may be incorrect.

  Args:
    result: A Tensor of `f(x)` values, to be analyzed as though it had
      been subject to float32 round-off.
    argument: A Tensor of `x` values broadcast-compatible with
      `result`, to be analyzed as though it had been subject to
      float32 round-off.
    derivative: A Tensor of `f'(x)` values broadcast-compatible with
      `result`.  If `None`, assume `f(x)` does not depend on `x`
      (which corresponds to a condition number of 0).

  Returns:
    condition_number: A float64 Tensor of condition numbers,
      corresponding elementwise to the input Tensors.

  """
  if derivative is None:
    # The output doesn't depend on this input (up to stop_gradient
    # tricks), so this input forces no wrong bits.  This is also correct
    # if the input is an integer, because then it's notionally exact
    # and still forces no wrong bits.
    return 0.0
  # Do not correct for increased rounding error in the answer when the answer is
  # close to 0, because that amounts to demanding more precision near zero
  # outputs, and I don't think that demand is appropriate.
  derivative = tf.cast(derivative, dtype=tf.float64)
  return (rounding_error(argument) * tf.math.abs(derivative) /
          rounding_error(result, denormal_correction=False))


def _full_flatten(xs):
  def flatten(x):
    return tf.reshape(x, shape=[-1])
  return tf.concat(tf.nest.flatten(tf.nest.map_structure(flatten, xs)), axis=-1)


def inputwise_condition_numbers(f, *args):
  """Computes the condition numbers of `f(*args)` at each arg independently.

  The function `f(*args)` must produce a scalar result; computing
  batches of condition numbers or computing condition numbers of
  vector-valued functions is not yet supported.

  This function assumes that numerical error when computing `f` in
  float64 is negligible.  For this to work correctly, `f` needs to be
  _dtype-polymorphic_: the dtype in which computations internal to `f`
  are performed should match the dtype of the arguments of `f`.

  Args:
    f: Function whose accuracy to evaluate.  Must be differentiable
      and dtype-polymorphic.
    *args: Arguments at which to test the accuracy of `f`.

  Returns:
    condition_numbers: The condition number of `f` with respect to each input.
      The returned structure is parallel to `*args`.

  Raises:
    ValueError: If `f` is found not to be dtype-polymorphic.

  """
  # TODO(b/181967692): Compute multivariate condition numbers.
  # TODO(b/181967437): To support batch condition numbers, need batch gradients.
  # Then can infer the "event shape" of the arguments by subtracting
  # off the number of dimensions in f(*args).
  # To also support vector outputs, need to know the "event_ndims" in
  # the output f(*args), and need full Jacobians of f underneath.
  args_32 = tf.nest.map_structure(floating_tensor_to_f32, args)
  logging.vlog(1, '32-bit arguments: %s', args_32)
  args_64 = tf.nest.map_structure(floating_tensor_to_f64, args_32)
  truth, derivatives = gradient.value_and_gradient(f, args_64)
  logging.vlog(1, 'Correct answer: %s', truth)
  logging.vlog(1, 'Argument gradient: %s', derivatives)
  def check_numerics(x):
    if x is None:
      return None
    msg = 'Cannot check accuracy if ground truth or derivatives are not finite'
    return tf.debugging.check_numerics(x, message=msg)
  truth = check_numerics(truth)
  derivatives = tf.nest.map_structure(check_numerics, derivatives)
  if truth.dtype != tf.float64:
    raise ValueError('Evaluating on {} produced non-64-bit result {}'.format(
        args_64, truth))
  return tf.nest.map_structure(
      functools.partial(condition_number_one_input, truth),
      # For some reason, value_and_gradient casts the outer structure to list in
      # jax.  Is that an oversight?
      tuple(args_64), tuple(derivatives))


def error_due_to_ill_conditioning(f, *args):
  """Returns relative error to expect in `f(*args)` due to conditioning.

  This function assumes that `f` is differentiable, and that numerical
  error when computing `f` or its derivatives in float64 is
  negligible.  One necessary condition for this to work correctly is
  that `f` be _dtype-polymorphic_: the dtype in which computations
  internal to `f` (and its derivatives) are performed should match the
  dtype of the arguments of `f`.

  The current implementation of this function evaluates ill
  conditioning of `f` independently for each argument.  It would
  perhaps be more faithful to accepted practice to compute the
  multivariate condition number instead, which takes account of errors
  caused by coordinated rounding among the inputs.

  The function `f` must return a single scalar output; batching and
  vector outputs from `f` are not currently supported.

  Args:
    f: Function whose accuracy to evaluate.  Must be differentiable
      and dtype-polymorphic.
    *args: Arguments at which to assess the accuracy of `f`.

  Returns:
    relerr: A scalar float64 Tensor.  The relative error when
      computing `f(*args)` in float32 that should be expected due to
      ill conditioning of `f`.

  Raises:
    ValueError: If `f` is found not to be dtype-polymorphic.

  """
  condition_numbers = inputwise_condition_numbers(f, *args)
  logging.vlog(1, 'Inputwise condition numbers: %s', condition_numbers)
  rounding_errors = tf.nest.map_structure(
      lambda x, k: rounding_error(x) * k, args, condition_numbers)
  logging.vlog(1, 'Relative error due to rounding each argument: %s',
               rounding_errors)
  return tf.reduce_max(_full_flatten(rounding_errors), axis=-1)


def excess_wrong_bits(f, *args):
  """Returns excess inaccuracy of 32-bit `f(*args)`, relative to conditioning.

  If this is positive, that suggests the implementation of `f` is
  introducing unnecessary numerical error at the given arguments.

  This function assumes that `f` is differentiable, and that numerical
  error when computing `f` or its derivatives in float64 is
  negligible.  One necessary condition for this to work correctly is
  that `f` be _dtype-polymorphic_: the dtype in which computations
  internal to `f` (and its derivatives) are performed should match the
  dtype of the arguments of `f`.

  Args:
    f: Function whose accuracy to evaluate.  Must be differentiable
      and dtype-polymorphic.
    *args: Arguments at which to test the accuracy of `f`.

  Returns:
    wrong: The wrong bits when computing `f(*args)` in float32, in excess
      of what would be expected from `f` being ill-conditioned.
  """
  err = relative_error_at(f, *args)
  logging.vlog(1, 'Relative error: %s', err)
  conditioning_err = error_due_to_ill_conditioning(f, *args)
  logging.vlog(1, 'Relative error due to input rounding: %s', conditioning_err)
  wrong = wrong_bits(err)
  conditioning = wrong_bits(conditioning_err)
  logging.vlog(1, 'Wrong bits: %s', wrong)
  logging.vlog(1, 'Wrong bits due to input rounding: %s', conditioning)
  return wrong - conditioning
