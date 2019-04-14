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
"""Tests for internal.backend.numpy.internal.numpy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import logging

# Dependency imports

from absl.testing import parameterized

import hypothesis as hp
import hypothesis.extra.numpy as hnp
import hypothesis.strategies as hps
import numpy as np
import tensorflow as tf

from tensorflow_probability.python.internal import test_case
from tensorflow_probability.python.internal.backend import numpy as numpy_backend


ALLOW_NAN = False
ALLOW_INFINITY = False


def _getattr(obj, name):
  names = name.split('.')
  return functools.reduce(getattr, names, obj)


class TestCase(dict):
  """`dict` object containing test strategies for a single function."""

  def __init__(self, name, strategy_list):
    self.name = name
    super(TestCase, self).__init__(
        testcase_name='_' + name.replace('.', '_'),
        tensorflow_function=_getattr(tf, name),
        numpy_function=_getattr(numpy_backend, name),
        strategy_list=strategy_list)

  def __repr__(self):
    return 'TestCase(\'{}\', {})'.format(self.name, self['strategy_list'])


# Below we define several test strategies. Each describes the valid inputs for
# different TensorFlow and numpy functions. See hypothesis.readthedocs.io for
# mode detail.

floats = functools.partial(
    hps.floats, allow_nan=ALLOW_NAN, allow_infinity=ALLOW_INFINITY)


@hps.composite
def n_same_shape(
    draw, n, min_dims=1, max_dims=5, dtype=None, elements=None):
  if elements is None:
    elements = floats()
  if dtype is None:
    dtype = np.float
  shape = draw(hnp.array_shapes(min_dims, max_dims))
  array_strategy = hnp.arrays(dtype, shape, elements=elements)
  if n == 1:
    return draw(array_strategy)
  return tuple([draw(array_strategy) for _ in range(n)])


single_array = functools.partial(n_same_shape, 1)


@hps.composite
def array_and_axis(draw, strategy=None):
  x = draw(strategy or single_array())
  rank = len(x.shape)
  axis = draw(hps.integers(- rank, rank - 1))
  return x, axis


@hps.composite
def array_and_diagonal(draw):
  side = draw(hps.integers(1, 10))
  shape = draw(hnp.array_shapes(
      min_dims=2, min_side=side, max_side=side))
  array = draw(hnp.arrays(np.float, shape, elements=floats()))
  diag = draw(hnp.arrays(np.float, shape[:-1], elements=floats()))
  return array, diag


@hps.composite
def matmul_compatible_pair(
    draw, dtype=np.float, x_strategy=None, elements=None):
  elements = elements or floats()
  x_strategy = x_strategy or single_array(2, 5, dtype=dtype, elements=elements)
  x = draw(x_strategy)
  x_shape = x.shape
  y_shape = x_shape[:-2] + x_shape[-1:] + (draw(hps.integers(1, 10)),)
  y = draw(
      hnp.arrays(dtype, y_shape, elements=elements))
  return x, y


@hps.composite
def psd_matrix(draw, eps=1e-2):
  x = draw(single_array(min_dims=2, max_dims=2, elements=floats(-1e3, 1e3)))
  return x.dot(x.T) + eps * np.eye(x.shape[0])


@hps.composite
def nested(draw, strategies):
  return tf.nest.map_structure(draw, strategies)

# __Currently untested:__
# math.bincount
# math.polyval
# linalg.set_diag
# linalg.diag_part
# linalg.band_part
# broadcast_to
# math.accumulate_n
# linalg.triangular_solve
# broadcast_dynamic_shape
# broadcast_static_shape
# math.zeta

# TODO(jamieas): add tests for these fucntions.

# pylint: disable=no-value-for-parameter

NUMPY_TEST_CASES = [

    # ArgSpec(args=['a', 'b', 'transpose_a', 'transpose_b', 'adjoint_a',
    #               'adjoint_b', 'a_is_sparse', 'b_is_sparse', 'name'],
    #         varargs=None,
    #         keywords=None,
    #         defaults=(False, False, False, False, False, False, None))
    TestCase('linalg.matmul', [
        matmul_compatible_pair(
            elements=floats(
                min_value=-1e12,
                max_value=1e12,
                allow_nan=ALLOW_NAN,
                allow_infinity=ALLOW_INFINITY))
    ]),

    # ArgSpec(args=['a', 'name', 'conjugate'], varargs=None, keywords=None,
    #         defaults=('matrix_transpose', False))
    TestCase('linalg.matrix_transpose', [n_same_shape(n=1, min_dims=2)]),

    # ArgSpec(args=['a', 'x', 'name'], varargs=None, keywords=None,
    #         defaults=(None,))
    TestCase('math.polygamma', [
        n_same_shape(
            n=2,
            elements=floats(
                min_value=0.01,
                max_value=10.0,
                allow_nan=ALLOW_NAN,
                allow_infinity=ALLOW_INFINITY))
    ]),

    # ArgSpec(args=['arr', 'weights', 'minlength',
    #               'maxlength', 'dtype', 'name'],
    #         varargs=None,
    #         keywords=None,
    #         defaults=(None, None, None, tf.int32, None))
    TestCase('math.bincount', []),

    # ArgSpec(args=['chol', 'rhs', 'name'], varargs=None, keywords=None,
    #         defaults=(None,))
    TestCase('linalg.cholesky_solve', [
        matmul_compatible_pair(x_strategy=psd_matrix().map(np.linalg.cholesky))
    ]),

    # ArgSpec(args=['coeffs', 'x', 'name'], varargs=None, keywords=None,
    #         defaults=(None,))
    TestCase('math.polyval', []),

    # ArgSpec(args=['diagonal', 'name'], varargs=None, keywords=None,
    #         defaults=(None,))
    TestCase('linalg.diag', [n_same_shape(n=1, max_dims=1)]),

    # ArgSpec(args=['features', 'name'], varargs=None, keywords=None,
    #         defaults=(None,))
    TestCase('math.softsign', [n_same_shape(n=1)]),

    # ArgSpec(args=['input', 'axis', 'keepdims', 'dtype', 'name'], varargs=None,
    #         keywords=None, defaults=(None, None, tf.int64, None))
    TestCase('math.count_nonzero', [n_same_shape(n=1)]),

    # ArgSpec(args=['input', 'axis', 'output_type', 'name'], varargs=None,
    #         keywords=None, defaults=(None, tf.int64, None))
    TestCase('math.argmax', [array_and_axis()]),
    TestCase('math.argmin', [array_and_axis()]),

    # ArgSpec(args=['input', 'diagonal', 'name'], varargs=None, keywords=None,
    #         defaults=(None,))
    # TODO(jamieas): test this with `array_and_diagonal()` once `set_diag` is
    # correctly implemented.
    TestCase('linalg.set_diag', []),

    # ArgSpec(args=['input', 'name'], varargs=None, keywords=None,
    #         defaults=(None,))
    TestCase('math.angle', [
        n_same_shape(
            n=1,
            dtype=np.complex,
            elements=hps.complex_numbers(min_magnitude=1e-06))
    ]),
    TestCase(
        'math.imag',
        [n_same_shape(n=1, dtype=np.complex, elements=hps.complex_numbers())]),
    TestCase(
        'math.real',
        [n_same_shape(n=1, dtype=np.complex, elements=hps.complex_numbers())]),
    TestCase('linalg.cholesky', [psd_matrix()]),
    TestCase('linalg.diag_part', []),
    TestCase('identity', [n_same_shape(n=1)]),

    # ArgSpec(args=['input', 'num_lower', 'num_upper', 'name'], varargs=None,
    #         keywords=None, defaults=(None,))
    TestCase('linalg.band_part', []),

    # ArgSpec(args=['input', 'shape', 'name'], varargs=None, keywords=None,
    #         defaults=(None,))
    TestCase('broadcast_to', []),

    # ArgSpec(args=['input_tensor', 'axis', 'keepdims', 'name'], varargs=None,
    #         keywords=None, defaults=(None, False, None))
    TestCase('math.reduce_all',
             [array_and_axis(
                 single_array(dtype=np.bool, elements=hps.booleans()))]),
    TestCase('math.reduce_any',
             [array_and_axis(
                 single_array(dtype=np.bool, elements=hps.booleans()))]),
    TestCase('math.reduce_logsumexp', [array_and_axis(single_array())]),
    TestCase('math.reduce_max', [array_and_axis(single_array())]),
    TestCase('math.reduce_mean', [array_and_axis(single_array())]),
    TestCase('math.reduce_min', [array_and_axis(single_array())]),
    TestCase('math.reduce_prod', [array_and_axis(single_array())]),
    TestCase('math.reduce_std', [array_and_axis(single_array())]),
    TestCase('math.reduce_sum', [array_and_axis(single_array())]),
    TestCase('math.reduce_variance', [array_and_axis(single_array())]),

    # ArgSpec(args=['inputs', 'name'], varargs=None, keywords=None,
    #         defaults=(None,))
    TestCase('math.add_n', [nested((n_same_shape(n=5),))]),

    # ArgSpec(args=['inputs', 'shape', 'tensor_dtype', 'name'], varargs=None,
    #         keywords=None, defaults=(None, None, None))
    TestCase('math.accumulate_n', []),

    # ArgSpec(args=['logits', 'axis', 'name'], varargs=None, keywords=None,
    #         defaults=(None, None))
    TestCase('math.log_softmax', [
        n_same_shape(
            n=1,
            elements=floats(
                min_value=-1e6,
                max_value=1e6,
                allow_nan=False,
                allow_infinity=False))
    ]),
    TestCase('math.softmax', [
        n_same_shape(
            n=1,
            elements=floats(
                min_value=-1e6,
                max_value=1e6,
                allow_nan=False,
                allow_infinity=False))
    ]),

    # ArgSpec(args=['matrix', 'rhs', 'lower', 'adjoint', 'name'], varargs=None,
    # keywords=None, defaults=(True, False, None))
    TestCase('linalg.triangular_solve', []),

    # ArgSpec(args=['shape_x', 'shape_y'], varargs=None, keywords=None,
    #         defaults=None)
    TestCase('broadcast_dynamic_shape', []),
    TestCase('broadcast_static_shape', []),

    # ArgSpec(args=['value', 'dtype', 'dtype_hint', 'name'], varargs=None,
    #         keywords=None, defaults=(None, None, None))
    TestCase('convert_to_tensor', [n_same_shape(n=1)]),

    # ArgSpec(args=['x', 'axis', 'exclusive', 'reverse', 'name'], varargs=None,
    #         keywords=None, defaults=(0, False, False, None))
    TestCase('math.cumprod', [n_same_shape(n=1)]),
    TestCase('math.cumsum', [n_same_shape(n=1)]),

    # ArgSpec(args=['x', 'name'], varargs=None, keywords=None, defaults=(None,))
    TestCase('math.abs', [n_same_shape(n=1)]),
    TestCase('math.acos', [n_same_shape(n=1)]),
    TestCase('math.acosh', [n_same_shape(n=1)]),
    TestCase('math.asin', [n_same_shape(n=1)]),
    TestCase('math.asinh', [n_same_shape(n=1)]),
    TestCase('math.atan', [n_same_shape(n=1)]),
    TestCase('math.atanh', [n_same_shape(n=1)]),
    TestCase('math.bessel_i0', [n_same_shape(n=1)]),
    TestCase('math.bessel_i0e', [n_same_shape(n=1)]),
    TestCase('math.bessel_i1', [n_same_shape(n=1)]),
    TestCase('math.bessel_i1e', [n_same_shape(n=1)]),
    TestCase('math.ceil', [n_same_shape(n=1)]),
    TestCase(
        'math.conj',
        [n_same_shape(n=1, dtype=np.complex, elements=hps.complex_numbers())]),
    TestCase('math.cos', [n_same_shape(n=1)]),
    TestCase('math.cosh', [n_same_shape(n=1)]),
    TestCase('math.digamma', [
        n_same_shape(
            n=1,
            elements=floats(
                min_value=1e-3,
                max_value=1e8,
                allow_nan=ALLOW_NAN,
                allow_infinity=ALLOW_INFINITY)),
        n_same_shape(
            n=1,
            elements=floats(
                min_value=-1e8,
                max_value=-1e-3,
                allow_nan=ALLOW_NAN,
                allow_infinity=ALLOW_INFINITY))
    ]),
    TestCase('math.erf', [n_same_shape(n=1)]),
    TestCase('math.erfc', [n_same_shape(n=1)]),
    TestCase('math.exp', [n_same_shape(n=1)]),
    TestCase('math.expm1', [n_same_shape(n=1)]),
    TestCase('math.floor', [n_same_shape(n=1)]),
    TestCase('math.is_finite', [n_same_shape(n=1)]),
    TestCase('math.is_inf', [n_same_shape(n=1)]),
    TestCase('math.is_nan', [n_same_shape(n=1)]),
    TestCase('math.lgamma', [
        n_same_shape(
            n=1,
            elements=floats(
                min_value=1e-08,
                max_value=1e+32,
                allow_nan=False,
                allow_infinity=False))
    ]),
    TestCase('math.log', [
        n_same_shape(
            n=1,
            elements=floats(
                min_value=1e-08,
                max_value=1e+32,
                allow_nan=False,
                allow_infinity=False))
    ]),
    TestCase('math.log1p', [
        n_same_shape(
            n=1,
            elements=floats(
                min_value=-0.99999999,
                max_value=1e+32,
                allow_nan=False,
                allow_infinity=False))
    ]),
    TestCase('math.log_sigmoid', [
        n_same_shape(
            n=1,
            elements=floats(
                min_value=-100.0, allow_nan=False, allow_infinity=False))
    ]),
    TestCase('math.logical_not',
             [n_same_shape(n=1, dtype=np.bool, elements=hps.booleans())]),
    TestCase('math.negative', [n_same_shape(n=1)]),
    TestCase('math.reciprocal', [n_same_shape(n=1)]),
    TestCase('math.rint', [n_same_shape(n=1)]),
    TestCase('math.round', [n_same_shape(n=1)]),
    TestCase('math.rsqrt', [
        n_same_shape(
            n=1,
            elements=floats(
                min_value=1e-06, allow_nan=False, allow_infinity=False))
    ]),
    TestCase('math.sigmoid', [n_same_shape(n=1)]),
    TestCase('math.sign', [n_same_shape(n=1)]),
    TestCase('math.sin', [n_same_shape(n=1)]),
    TestCase('math.sinh', [n_same_shape(n=1)]),
    TestCase('math.softplus', [n_same_shape(n=1)]),
    TestCase('math.sqrt', [n_same_shape(n=1)]),
    TestCase('math.square', [n_same_shape(n=1)]),
    TestCase('math.tan', [n_same_shape(n=1)]),
    TestCase('math.tanh', [n_same_shape(n=1)]),

    # ArgSpec(args=['x', 'q', 'name'], varargs=None, keywords=None,
    #         defaults=(None,))
    TestCase('math.zeta', []),

    # ArgSpec(args=['x', 'y', 'name'], varargs=None, keywords=None,
    #         defaults=(None,))
    TestCase('math.add', [n_same_shape(n=2)]),
    TestCase('math.atan2', [n_same_shape(n=2)]),
    TestCase('math.divide', [n_same_shape(n=2)]),
    TestCase('math.divide_no_nan', [n_same_shape(n=2)]),
    TestCase('math.equal', [n_same_shape(n=2)]),
    TestCase('math.floordiv', [
        n_same_shape(
            n=2,
            elements=floats(
                min_value=1e-06, allow_nan=False, allow_infinity=False))
    ]),
    TestCase('math.greater', [n_same_shape(n=2)]),
    TestCase('math.greater_equal', [n_same_shape(n=2)]),
    TestCase('math.less', [n_same_shape(n=2)]),
    TestCase('math.less_equal', [n_same_shape(n=2)]),
    TestCase('math.logical_and',
             [n_same_shape(n=2, dtype=np.bool, elements=hps.booleans())]),
    TestCase('math.logical_or',
             [n_same_shape(n=2, dtype=np.bool, elements=hps.booleans())]),
    TestCase('math.logical_xor',
             [n_same_shape(n=2, dtype=np.bool, elements=hps.booleans())]),
    TestCase('math.maximum', [n_same_shape(n=2)]),
    TestCase('math.minimum', [n_same_shape(n=2)]),
    TestCase('math.multiply', [n_same_shape(n=2)]),
    TestCase('math.multiply_no_nan', [n_same_shape(n=2)]),
    TestCase('math.not_equal', [n_same_shape(n=2)]),
    TestCase('math.pow', [n_same_shape(n=2)]),
    TestCase('math.squared_difference', [n_same_shape(n=2)]),
    TestCase('math.subtract', [n_same_shape(n=2)]),
    TestCase('math.truediv', [n_same_shape(n=2)]),
    TestCase('math.xdivy', [n_same_shape(n=2)]),
    TestCase('math.xlogy', [n_same_shape(n=2)]),
]


def _maybe_convert_to_tensors(args):
  return tuple([tf.convert_to_tensor(value=arg)  # pylint: disable=g-complex-comprehension
                if isinstance(arg, np.ndarray)
                else arg
                for arg in args])


class NumpyTest(test_case.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(NUMPY_TEST_CASES)
  def testLogEmptyTestCases(
      self, tensorflow_function, numpy_function, strategy_list, atol=1e-6):
    if not strategy_list:
      logging.warning(
          'The test for %s contains no strategies.', numpy_function.__name__)

  @parameterized.named_parameters(NUMPY_TEST_CASES)
  def testConsistency(
      self, tensorflow_function, numpy_function, strategy_list, atol=1e-6):
    for strategy in strategy_list:
      @hp.settings(deadline=None,
                   max_examples=10,
                   derandomize=True)
      @hp.given(strategy)
      def check_consistency(tf_fn, np_fn, args):
        # If `args` is a single item, put it in a tuple
        if isinstance(args, np.ndarray) or tf.is_tensor(args):
          args = (args,)
        tensorflow_value = self.evaluate(
            tf_fn(*_maybe_convert_to_tensors(args)))
        numpy_value = np_fn(*args)
        self.assertAllCloseAccordingToType(
            tensorflow_value, numpy_value, atol=atol)
      check_consistency(tensorflow_function, numpy_function)


if __name__ == '__main__':
  tf.test.main()
