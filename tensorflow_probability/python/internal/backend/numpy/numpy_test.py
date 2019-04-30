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


def floats(min_value=-1e16,
           max_value=1e16,
           allow_nan=ALLOW_NAN,
           allow_infinity=ALLOW_INFINITY):
  return hps.floats(min_value, max_value, allow_nan, allow_infinity)


def complex_numbers(min_magnitude=0.,
                    max_magnitude=1e16,
                    allow_nan=ALLOW_NAN,
                    allow_infinity=ALLOW_INFINITY):
  return hps.complex_numbers(
      min_magnitude, max_magnitude, allow_nan, allow_infinity)


@hps.composite
def non_zero_floats(draw, *args, **kwargs):
  return draw(floats(*args, **kwargs).filter(lambda x: np.all(x != 0.)))

positive_floats = functools.partial(floats, min_value=1e-6)


@hps.composite
def n_same_shape(draw,
                 n,
                 min_dims=1,
                 max_dims=4,
                 min_side=1,
                 max_side=5,
                 dtype=None,
                 elements=None):
  if elements is None:
    elements = floats()
  if dtype is None:
    dtype = np.float
  shape = draw(hnp.array_shapes(min_dims, max_dims, min_side, max_side))

  if isinstance(elements, (list, tuple)):
    return tuple([draw(hnp.arrays(dtype, shape, elements=e)) for e in elements])
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
  x = draw(single_array(
      min_dims=2, max_dims=2, elements=floats(min_value=-1e3, max_value=1e3)))
  return x.dot(x.T) + eps * np.eye(x.shape[0])


# __Currently untested:__
# broadcast_dynamic_shape
# broadcast_static_shape
# broadcast_to
# linalg.band_part
# linalg.diag_part
# linalg.set_diag
# linalg.triangular_solve
# math.accumulate_n
# math.betainc
# math.bincount
# math.igamma
# math.igammac
# math.lbeta
# math.polyval
# math.zeta
# random.categorical
# random.gamma
# random.normal
# random.poisson
# random.set_seed
# random.uniform


# TODO(jamieas): add tests for these functions.

# pylint: disable=no-value-for-parameter

NUMPY_TEST_CASES = [

    # ArgSpec(args=['a', 'b', 'transpose_a', 'transpose_b', 'adjoint_a',
    #               'adjoint_b', 'a_is_sparse', 'b_is_sparse', 'name'],
    #         varargs=None,
    #         keywords=None,
    #         defaults=(False, False, False, False, False, False, None))
    TestCase('linalg.matmul', [matmul_compatible_pair()]),

    # ArgSpec(args=['a', 'name', 'conjugate'], varargs=None, keywords=None)
    TestCase('linalg.matrix_transpose', [single_array(min_dims=2)]),

    # ArgSpec(args=['a', 'x', 'name'], varargs=None, keywords=None,
    #         defaults=(None,))
    TestCase('math.polygamma', [
        hps.tuples(hps.integers(0, 10).map(float), positive_floats()),
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
    TestCase('linalg.diag', [single_array(max_dims=1)]),

    # ArgSpec(args=['features', 'name'], varargs=None, keywords=None,
    #         defaults=(None,))
    TestCase('math.softsign', [single_array()]),

    # ArgSpec(args=['input', 'axis', 'keepdims', 'dtype', 'name'], varargs=None,
    #         keywords=None, defaults=(None, None, tf.int64, None))
    TestCase('math.count_nonzero', [single_array()]),

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
    TestCase('math.angle',
             [single_array(dtype=np.complex, elements=complex_numbers())]),
    TestCase('math.imag',
             [single_array(dtype=np.complex, elements=complex_numbers())]),
    TestCase('math.real',
             [single_array(dtype=np.complex, elements=complex_numbers())]),
    TestCase('linalg.cholesky', [psd_matrix()]),
    TestCase('linalg.diag_part', []),
    TestCase('identity', [single_array()]),

    # ArgSpec(args=['input', 'num_lower', 'num_upper', 'name'], varargs=None,
    #         keywords=None, defaults=(None,))
    TestCase('linalg.band_part', []),

    # ArgSpec(args=['input', 'shape', 'name'], varargs=None, keywords=None,
    #         defaults=(None,))
    TestCase('broadcast_to', []),

    # ArgSpec(args=['input_tensor', 'axis', 'keepdims', 'name'], varargs=None,
    #         keywords=None, defaults=(None, False, None))
    TestCase(
        'math.reduce_all',
        [array_and_axis(single_array(dtype=np.bool, elements=hps.booleans()))]),
    TestCase(
        'math.reduce_any',
        [array_and_axis(single_array(dtype=np.bool, elements=hps.booleans()))]),
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
    TestCase('math.add_n', [hps.tuples(n_same_shape(n=5))]),

    # ArgSpec(args=['inputs', 'shape', 'tensor_dtype', 'name'], varargs=None,
    #         keywords=None, defaults=(None, None, None))
    TestCase('math.accumulate_n', []),

    # ArgSpec(args=['logits', 'axis', 'name'], varargs=None, keywords=None,
    #         defaults=(None, None))
    TestCase('math.log_softmax', [
        single_array(
            elements=floats(
                min_value=-1e6,
                max_value=1e6,
                allow_nan=False,
                allow_infinity=False))
    ]),
    TestCase('math.softmax', [
        single_array(
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
    TestCase('convert_to_tensor', [single_array()]),

    # ArgSpec(args=['x', 'axis', 'exclusive', 'reverse', 'name'], varargs=None,
    #         keywords=None, defaults=(0, False, False, None))
    TestCase('math.cumprod', [single_array()]),
    TestCase('math.cumsum', [single_array()]),

    # ArgSpec(args=['x', 'name'], varargs=None, keywords=None, defaults=(None,))
    TestCase('math.abs', [single_array()]),
    TestCase('math.acos', [single_array(elements=floats(-1., 1.))]),
    TestCase('math.acosh', [single_array(elements=positive_floats())]),
    TestCase('math.asin', [single_array(elements=floats(-1., 1.))]),
    TestCase('math.asinh', [single_array(elements=positive_floats())]),
    TestCase('math.atan', [single_array()]),
    TestCase('math.atanh', [single_array(elements=floats(-1., 1.))]),
    TestCase('math.bessel_i0', [single_array(elements=floats(-50., 50.))]),
    TestCase('math.bessel_i0e', [single_array(elements=floats(-50., 50.))]),
    TestCase('math.bessel_i1', [single_array(elements=floats(-50., 50.))]),
    TestCase('math.bessel_i1e', [single_array(elements=floats(-50., 50.))]),
    TestCase('math.ceil', [single_array()]),
    TestCase('math.conj',
             [single_array(dtype=np.complex, elements=complex_numbers())]),
    TestCase('math.cos', [single_array()]),
    TestCase('math.cosh', [single_array(elements=floats(-100., 100.))]),
    TestCase(
        'math.digamma',
        [single_array(elements=non_zero_floats(min_value=-1e4, max_value=1e4))
        ]),
    TestCase('math.erf', [single_array()]),
    TestCase('math.erfc', [single_array()]),
    TestCase('math.exp', [single_array(
        elements=floats(min_value=-1e3, max_value=1e3))]),
    TestCase('math.expm1', [single_array(
        elements=floats(min_value=-1e3, max_value=1e3))]),
    TestCase('math.floor', [single_array()]),
    TestCase('math.is_finite', [single_array()]),
    TestCase('math.is_inf', [single_array()]),
    TestCase('math.is_nan', [single_array()]),
    TestCase('math.lgamma', [single_array(elements=positive_floats())]),
    TestCase('math.log', [single_array(elements=positive_floats())]),
    TestCase('math.log1p',
             [single_array(elements=positive_floats().map(lambda x: x - 1.))]),
    TestCase('math.log_sigmoid',
             [single_array(elements=floats(min_value=-100.))]),
    TestCase('math.logical_not',
             [single_array(dtype=np.bool, elements=hps.booleans())]),
    TestCase('math.negative', [single_array()]),
    TestCase('math.reciprocal', [single_array()]),
    TestCase('math.rint', [single_array()]),
    TestCase('math.round', [single_array()]),
    TestCase('math.rsqrt', [single_array(elements=positive_floats())]),
    TestCase('math.sigmoid', [single_array()]),
    TestCase('math.sign', [single_array()]),
    TestCase('math.sin', [single_array()]),
    TestCase('math.sinh', [single_array(elements=floats(-100., 100.))]),
    TestCase('math.softplus', [single_array()]),
    TestCase('math.sqrt', [single_array(elements=positive_floats())]),
    TestCase('math.square', [single_array()]),
    TestCase('math.tan', [single_array()]),
    TestCase('math.tanh', [single_array()]),

    # ArgSpec(args=['x', 'q', 'name'], varargs=None, keywords=None,
    #         defaults=(None,))
    TestCase('math.zeta', []),

    # ArgSpec(args=['x', 'y', 'name'], varargs=None, keywords=None,
    #         defaults=(None,))
    TestCase('math.add', [n_same_shape(n=2)]),
    TestCase('math.atan2', [n_same_shape(n=2)]),
    TestCase('math.divide',
             [n_same_shape(n=2, elements=[floats(), non_zero_floats()])]),
    TestCase('math.divide_no_nan', [n_same_shape(n=2)]),
    TestCase('math.equal', [n_same_shape(n=2)]),
    TestCase('math.floordiv',
             [n_same_shape(n=2, elements=[floats(), non_zero_floats()])]),
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
    TestCase('math.pow',
             [n_same_shape(
                 n=2, elements=[floats(-1e3, 1e3), floats(-10., 10.)])]),
    TestCase('math.squared_difference', [n_same_shape(n=2)]),
    TestCase('math.subtract', [n_same_shape(n=2)]),
    TestCase('math.truediv',
             [n_same_shape(n=2, elements=[floats(), non_zero_floats()])]),
    TestCase('math.xdivy',
             [n_same_shape(n=2, elements=[floats(), non_zero_floats()])]),
    TestCase('math.xlogy',
             [n_same_shape(n=2, elements=[floats(), positive_floats()])]),
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
