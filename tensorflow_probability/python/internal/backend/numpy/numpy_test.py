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
# import logging

# Dependency imports

from absl.testing import parameterized

import hypothesis as hp
import hypothesis.extra.numpy as hnp
import hypothesis.strategies as hps
import numpy as np  # Rewritten by script to import jax.numpy
import numpy as onp  # pylint: disable=reimported
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import test_case
from tensorflow_probability.python.internal.backend import numpy as numpy_backend


ALLOW_NAN = False
ALLOW_INFINITY = False

MODE_JAX = False


def _add_jax_prng_key_as_seed():
  import jax.random as jaxrand  # pylint: disable=g-import-not-at-top
  return dict(seed=jaxrand.PRNGKey(123))


def _getattr(obj, name):
  names = name.split('.')
  return functools.reduce(getattr, names, obj)


class TestCase(dict):
  """`dict` object containing test strategies for a single function."""

  def __init__(self, name, strategy_list, **kwargs):
    self.name = name
    super(TestCase, self).__init__(
        testcase_name='_' + name.replace('.', '_'),
        tensorflow_function=_getattr(tf, name),
        numpy_function=_getattr(numpy_backend, name),
        strategy_list=strategy_list,
        **kwargs)

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


def shapes(min_dims=0, max_dims=4, min_side=1, max_side=5):
  strategy = hnp.array_shapes(max(1, min_dims), max_dims, min_side, max_side)
  if min_dims < 1:
    strategy = hps.one_of(hps.just(()), strategy)
  return strategy


@hps.composite
def n_same_shape(draw, n, shape=shapes(), dtype=None, elements=None,
                 as_tuple=True):
  if elements is None:
    elements = floats()
  if dtype is None:
    dtype = np.float64
  shape = draw(shape)

  ensure_array = lambda x: onp.array(x, dtype=dtype)
  if isinstance(elements, (list, tuple)):
    return tuple([
        draw(hnp.arrays(dtype, shape, elements=e).map(ensure_array))
        for e in elements
    ])
  array_strategy = hnp.arrays(dtype, shape, elements=elements).map(ensure_array)
  if n == 1 and not as_tuple:
    return draw(array_strategy)
  return draw(hps.tuples(*([array_strategy] * n)))


single_arrays = functools.partial(n_same_shape, n=1, as_tuple=False)


@hps.composite
def array_and_axis(draw, strategy=None):
  x = draw(strategy or single_arrays(shape=shapes(min_dims=1)))
  rank = len(x.shape)
  axis = draw(hps.integers(-rank, rank - 1))
  return x, axis


@hps.composite
def sliceable_and_slices(draw, strategy=None):
  x = draw(strategy or single_arrays(shape=shapes(min_dims=1)))
  starts = []
  sizes = []
  for dim in x.shape:
    starts.append(draw(hps.integers(0, dim - 1)))
    sizes.append(
        draw(hps.one_of(hps.just(-1), hps.integers(0, dim - starts[-1]))))
  return x, starts, sizes


@hps.composite
def one_hot_params(draw):
  indices = draw(single_arrays(dtype=np.int32, elements=hps.integers(0, 8)))
  depth = np.maximum(1, np.max(indices)).astype(np.int32)
  dtype = draw(hps.sampled_from((np.int32, np.float32, np.complex64)))
  on_value = draw(hps.sampled_from((None, 1, 2)))
  on_value = on_value if on_value is None else dtype(on_value)
  off_value = draw(hps.sampled_from((None, 3, 7)))
  off_value = off_value if off_value is None else dtype(off_value)
  rank = indices.ndim
  axis = draw(hps.one_of(hps.just(None), hps.integers(-1, rank - 1)))
  return indices, depth, on_value, off_value, axis, dtype


@hps.composite
def array_and_diagonal(draw):
  side = draw(hps.integers(1, 10))
  shape = draw(shapes(min_dims=2, min_side=side, max_side=side))
  array = draw(hnp.arrays(np.float64, shape, elements=floats()))
  diag = draw(hnp.arrays(np.float64, shape[:-1], elements=floats()))
  return array, diag


@hps.composite
def matmul_compatible_pairs(draw,
                            dtype=np.float64,
                            x_strategy=None,
                            elements=None):
  elements = elements or floats()
  x_strategy = x_strategy or single_arrays(
      shape=shapes(min_dims=2, max_dims=5), dtype=dtype, elements=elements)
  x = draw(x_strategy)
  x_shape = x.shape
  y_shape = x_shape[:-2] + x_shape[-1:] + (draw(hps.integers(1, 10)),)
  y = draw(
      hnp.arrays(dtype, y_shape, elements=elements))
  return x, y


@hps.composite
def psd_matrices(draw, eps=1e-2):
  x = draw(
      single_arrays(
          shape=shapes(min_dims=2),
          elements=floats(min_value=-1e3, max_value=1e3)))
  y = np.swapaxes(x, -1, -2)
  if x.shape[-1] < x.shape[-2]:  # Ensure resultant matrix not rank-deficient.
    x, y = y, x
  psd = np.matmul(x, y)
  return psd + eps * np.eye(psd.shape[-1])


@hps.composite
def nonsingular_matrices(draw):
  mat = draw(psd_matrices())  # pylint: disable=no-value-for-parameter
  signs = draw(
      hnp.arrays(
          mat.dtype,
          mat.shape[:-2] + (1, 1),
          elements=hps.sampled_from([-1., 1.])))
  return mat * signs


def gamma_params():
  def dict_to_params(d):
    return (d['shape'],  # sample shape
            d['params'][0].astype(d['dtype'].as_numpy_dtype),  # alpha
            (d['params'][1].astype(d['dtype'].as_numpy_dtype)  # beta (or None)
             if d['include_beta'] else None),
            d['dtype'])  # dtype
  return hps.fixed_dictionaries(
      dict(shape=shapes(),
           # hps.composite confuses pylint w/ the draw parameter...
           # pylint: disable=no-value-for-parameter
           params=n_same_shape(n=2, elements=positive_floats()),
           # pylint: enable=no-value-for-parameter
           include_beta=hps.booleans(),
           dtype=hps.sampled_from([tf.float32, tf.float64]))
      ).map(dict_to_params)  # dtype


# __Currently untested:__
# broadcast_dynamic_shape
# broadcast_static_shape
# broadcast_to
# math.accumulate_n
# math.betainc
# math.bincount
# math.igamma
# math.igammac
# math.lbeta
# math.polyval
# math.zeta
# random.categorical
# random.poisson
# random.set_seed


# TODO(jamieas): add tests for these functions.

# pylint: disable=no-value-for-parameter

NUMPY_TEST_CASES = [

    # ArgSpec(args=['a', 'b', 'transpose_a', 'transpose_b', 'adjoint_a',
    #               'adjoint_b', 'a_is_sparse', 'b_is_sparse', 'name'],
    #         varargs=None,
    #         keywords=None,
    #         defaults=(False, False, False, False, False, False, None))
    TestCase('linalg.matmul', [matmul_compatible_pairs()]),
    TestCase('linalg.det', [nonsingular_matrices()]),

    # ArgSpec(args=['a', 'name', 'conjugate'], varargs=None, keywords=None)
    TestCase('linalg.matrix_transpose',
             [single_arrays(shape=shapes(min_dims=2))]),

    # ArgSpec(args=['a', 'x', 'name'], varargs=None, keywords=None,
    #         defaults=(None,))
    TestCase(
        'math.polygamma', [
            hps.tuples(hps.integers(0, 10).map(float), positive_floats()),
        ],
        jax_disabled=True),

    # ArgSpec(args=['arr', 'weights', 'minlength',
    #               'maxlength', 'dtype', 'name'],
    #         varargs=None,
    #         keywords=None,
    #         defaults=(None, None, None, tf.int32, None))
    TestCase('math.bincount', []),

    # ArgSpec(args=['chol', 'rhs', 'name'], varargs=None, keywords=None,
    #         defaults=(None,))
    TestCase('linalg.cholesky_solve', [
        matmul_compatible_pairs(
            x_strategy=psd_matrices().map(np.linalg.cholesky))
    ]),

    # ArgSpec(args=['coeffs', 'x', 'name'], varargs=None, keywords=None,
    #         defaults=(None,))
    TestCase('math.polyval', []),

    # ArgSpec(args=['diagonal', 'name'], varargs=None, keywords=None,
    #         defaults=(None,))
    TestCase('linalg.diag', [single_arrays(shape=shapes(min_dims=1))]),

    # ArgSpec(args=['features', 'name'], varargs=None, keywords=None,
    #         defaults=(None,))
    TestCase('math.softsign', [single_arrays()]),

    # ArgSpec(args=['input', 'axis', 'keepdims', 'dtype', 'name'], varargs=None,
    #         keywords=None, defaults=(None, None, tf.int64, None))
    TestCase('math.count_nonzero', [single_arrays()]),

    # ArgSpec(args=['input', 'axis', 'output_type', 'name'], varargs=None,
    #         keywords=None, defaults=(None, tf.int64, None))
    TestCase('math.argmax', [array_and_axis()]),
    TestCase('math.argmin', [array_and_axis()]),

    # ArgSpec(args=['input', 'diagonal', 'name'], varargs=None, keywords=None,
    #         defaults=(None,))
    TestCase('linalg.set_diag', [array_and_diagonal()]),

    # ArgSpec(args=['input', 'name'], varargs=None, keywords=None,
    #         defaults=(None,))
    TestCase('math.angle',
             [single_arrays(dtype=np.complex64, elements=complex_numbers())]),
    TestCase('math.imag',
             [single_arrays(dtype=np.complex64, elements=complex_numbers())]),
    TestCase('math.real',
             [single_arrays(dtype=np.complex64, elements=complex_numbers())]),
    TestCase('linalg.cholesky', [psd_matrices()]),
    TestCase('linalg.diag_part', [single_arrays(shape=shapes(min_dims=2))]),
    TestCase('identity', [single_arrays()]),

    # ArgSpec(args=['input', 'num_lower', 'num_upper', 'name'], varargs=None,
    #         keywords=None, defaults=(None,))
    TestCase('linalg.band_part', [
        hps.tuples(
            single_arrays(shape=shapes(min_dims=2, min_side=3)),
            hps.integers(min_value=-1, max_value=3),
            hps.integers(min_value=-1, max_value=3))
    ]),

    # ArgSpec(args=['input', 'shape', 'name'], varargs=None, keywords=None,
    #         defaults=(None,))
    TestCase('broadcast_to', []),

    # ArgSpec(args=['input_tensor', 'axis', 'keepdims', 'name'], varargs=None,
    #         keywords=None, defaults=(None, False, None))
    TestCase('math.reduce_all', [
        array_and_axis(
            single_arrays(
                shape=shapes(min_dims=1),
                dtype=np.bool,
                elements=hps.booleans()))
    ]),
    TestCase('math.reduce_any', [
        array_and_axis(
            single_arrays(
                shape=shapes(min_dims=1),
                dtype=np.bool,
                elements=hps.booleans()))
    ]),
    TestCase('math.reduce_logsumexp', [array_and_axis()]),
    TestCase('math.reduce_max', [array_and_axis()]),
    TestCase('math.reduce_mean', [array_and_axis()]),
    TestCase('math.reduce_min', [array_and_axis()]),
    TestCase('math.reduce_prod', [array_and_axis()]),
    TestCase('math.reduce_std', [array_and_axis()]),
    TestCase('math.reduce_sum', [array_and_axis()]),
    TestCase('math.reduce_variance', [array_and_axis()]),

    # ArgSpec(args=['inputs', 'name'], varargs=None, keywords=None,
    #         defaults=(None,))
    TestCase(
        'math.add_n',
        [hps.integers(1, 5).flatmap(lambda n: hps.tuples(n_same_shape(n=n)))]),

    # ArgSpec(args=['inputs', 'shape', 'tensor_dtype', 'name'], varargs=None,
    #         keywords=None, defaults=(None, None, None))
    TestCase('math.accumulate_n', []),

    # ArgSpec(args=['logits', 'axis', 'name'], varargs=None, keywords=None,
    #         defaults=(None, None))
    TestCase('math.log_softmax', [
        single_arrays(
            shape=shapes(min_dims=1),
            elements=floats(
                min_value=-1e6,
                max_value=1e6,
                allow_nan=False,
                allow_infinity=False))
    ]),
    TestCase('math.softmax', [
        single_arrays(
            shape=shapes(min_dims=1),
            elements=floats(
                min_value=-1e6,
                max_value=1e6,
                allow_nan=False,
                allow_infinity=False))
    ]),

    # ArgSpec(args=['matrix', 'rhs', 'lower', 'adjoint', 'name'], varargs=None,
    # keywords=None, defaults=(True, False, None))
    TestCase('linalg.triangular_solve', [
        matmul_compatible_pairs(
            x_strategy=psd_matrices().map(np.linalg.cholesky))
    ]),

    # ArgSpec(args=['shape_x', 'shape_y'], varargs=None, keywords=None,
    #         defaults=None)
    TestCase('broadcast_dynamic_shape', []),
    TestCase('broadcast_static_shape', []),

    # ArgSpec(args=['value', 'dtype', 'dtype_hint', 'name'], varargs=None,
    #         keywords=None, defaults=(None, None, None))
    TestCase('convert_to_tensor', [single_arrays()]),

    # ArgSpec(args=['x', 'axis', 'exclusive', 'reverse', 'name'], varargs=None,
    #         keywords=None, defaults=(0, False, False, None))
    TestCase('math.cumprod', [
        hps.tuples(array_and_axis(), hps.booleans(),
                   hps.booleans()).map(lambda x: x[0] + (x[1], x[2]))
    ]),
    TestCase('math.cumsum', [
        hps.tuples(array_and_axis(), hps.booleans(),
                   hps.booleans()).map(lambda x: x[0] + (x[1], x[2]))
    ]),

    # args=['input', 'name']
    TestCase('linalg.slogdet', [nonsingular_matrices()]),
    # ArgSpec(args=['x', 'name'], varargs=None, keywords=None, defaults=(None,))
    TestCase('math.abs', [single_arrays()]),
    TestCase('math.acos', [single_arrays(elements=floats(-1., 1.))]),
    TestCase('math.acosh', [single_arrays(elements=positive_floats())]),
    TestCase('math.asin', [single_arrays(elements=floats(-1., 1.))]),
    TestCase('math.asinh', [single_arrays(elements=positive_floats())]),
    TestCase('math.atan', [single_arrays()]),
    TestCase('math.atanh', [single_arrays(elements=floats(-1., 1.))]),
    TestCase(
        'math.bessel_i0', [single_arrays(elements=floats(-50., 50.))],
        jax_disabled=True),
    TestCase(
        'math.bessel_i0e', [single_arrays(elements=floats(-50., 50.))],
        jax_disabled=True),
    TestCase(
        'math.bessel_i1', [single_arrays(elements=floats(-50., 50.))],
        jax_disabled=True),
    TestCase(
        'math.bessel_i1e', [single_arrays(elements=floats(-50., 50.))],
        jax_disabled=True),
    TestCase('math.ceil', [single_arrays()]),
    TestCase('math.conj',
             [single_arrays(dtype=np.complex64, elements=complex_numbers())]),
    TestCase('math.cos', [single_arrays()]),
    TestCase('math.cosh', [single_arrays(elements=floats(-100., 100.))]),
    TestCase('math.digamma',
             [single_arrays(elements=non_zero_floats(-1e4, 1e4))]),
    TestCase('math.erf', [single_arrays()]),
    TestCase('math.erfc', [single_arrays()]),
    TestCase('math.exp',
             [single_arrays(elements=floats(min_value=-1e3, max_value=1e3))]),
    TestCase('math.expm1',
             [single_arrays(elements=floats(min_value=-1e3, max_value=1e3))]),
    TestCase('math.floor', [single_arrays()]),
    TestCase('math.is_finite', [single_arrays()]),
    TestCase('math.is_inf', [single_arrays()]),
    TestCase('math.is_nan', [single_arrays()]),
    TestCase('math.lgamma', [single_arrays(elements=positive_floats())]),
    TestCase('math.log', [single_arrays(elements=positive_floats())]),
    TestCase('math.log1p',
             [single_arrays(elements=positive_floats().map(lambda x: x - 1.))]),
    TestCase('math.log_sigmoid',
             [single_arrays(elements=floats(min_value=-100.))]),
    TestCase('math.logical_not',
             [single_arrays(dtype=np.bool, elements=hps.booleans())]),
    TestCase('math.negative', [single_arrays()]),
    TestCase('math.reciprocal', [single_arrays()]),
    TestCase('math.rint', [single_arrays()]),
    TestCase('math.round', [single_arrays()]),
    TestCase('math.rsqrt', [single_arrays(elements=positive_floats())]),
    TestCase('math.sigmoid', [single_arrays()]),
    TestCase('math.sign', [single_arrays()]),
    TestCase('math.sin', [single_arrays()]),
    TestCase('math.sinh', [single_arrays(elements=floats(-100., 100.))]),
    TestCase('math.softplus', [single_arrays()]),
    TestCase('math.sqrt', [single_arrays(elements=positive_floats())]),
    TestCase('math.square', [single_arrays()]),
    TestCase('math.tan', [single_arrays()]),
    TestCase('math.tanh', [single_arrays()]),

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
    TestCase(
        'math.pow',
        [n_same_shape(n=2, elements=[floats(-1e3, 1e3),
                                     floats(-10., 10.)])]),
    TestCase('math.squared_difference', [n_same_shape(n=2)]),
    TestCase('math.subtract', [n_same_shape(n=2)]),
    TestCase('math.truediv',
             [n_same_shape(n=2, elements=[floats(), non_zero_floats()])]),
    TestCase('math.xdivy',
             [n_same_shape(n=2, elements=[floats(), non_zero_floats()])]),
    TestCase('math.xlogy',
             [n_same_shape(n=2, elements=[floats(), positive_floats()])]),
    TestCase(
        'random.gamma', [gamma_params()],
        jax_kwargs=_add_jax_prng_key_as_seed,
        assert_shape_only=True),
    TestCase(
        'random.normal', [hps.tuples(shapes())],
        jax_kwargs=_add_jax_prng_key_as_seed,
        assert_shape_only=True),
    TestCase(
        'random.uniform', [hps.tuples(shapes())],
        jax_kwargs=_add_jax_prng_key_as_seed,
        assert_shape_only=True),

    # Array ops.
    TestCase('one_hot', [one_hot_params()]),
    TestCase('slice', [sliceable_and_slices()]),
]


def _maybe_convert_to_tensors(args):
  # Ensures we go from JAX np -> original np -> tf.Tensor. (no-op for non-JAX.)
  convert = lambda a: tf.convert_to_tensor(onp.array(a))
  return tuple(
      convert(arg) if isinstance(arg, np.ndarray) else arg for arg in args)


class NumpyTest(test_case.TestCase, parameterized.TestCase):

  def test_convert_to_tensor(self):
    convert_to_tensor = numpy_backend.convert_to_tensor
    self.assertEqual(
        np.complex64,
        convert_to_tensor(np.complex64(1 + 2j), dtype_hint=tf.int32).dtype)
    self.assertEqual(
        np.complex64,
        convert_to_tensor(np.complex64(1 + 2j), dtype_hint=tf.float64).dtype)
    self.assertEqual(np.float64,
                     convert_to_tensor(1., dtype_hint=tf.int32).dtype)
    self.assertEqual(np.int32, convert_to_tensor(1, dtype_hint=tf.int32).dtype)
    self.assertEqual(np.int64, convert_to_tensor(1, dtype_hint=tf.int64).dtype)
    self.assertEqual(
        np.int32,
        convert_to_tensor(np.int32(False), dtype_hint=tf.bool).dtype)

  def evaluate(self, tensors):
    if tf.executing_eagerly():
      return self._eval_helper(tensors)
    else:
      sess = tf.compat.v1.get_default_session()
      if sess is None:
        with self.session() as sess:
          return sess.run(tensors)
      else:
        return sess.run(tensors)

  @parameterized.named_parameters(NUMPY_TEST_CASES)
  def testLogEmptyTestCases(self,
                            tensorflow_function,
                            numpy_function,
                            strategy_list,
                            jax_disabled=False,
                            **_):
    if jax_disabled and MODE_JAX:
      tf.compat.v1.logging.warning('The test for %s is disabled for JAX.',
                                   numpy_function.__name__)
    elif not strategy_list:
      tf.compat.v1.logging.warning(
          'The test for %s contains no strategies.', numpy_function.__name__)
    else:
      self.skipTest('Has coverage.')

  @parameterized.named_parameters(NUMPY_TEST_CASES)
  def testConsistency(self,
                      tensorflow_function,
                      numpy_function,
                      strategy_list,
                      atol=1e-6,
                      rtol=1e-6,
                      jax_disabled=False,
                      assert_shape_only=False,
                      jax_kwargs=lambda: {}):
    if jax_disabled and MODE_JAX:
      self.skipTest('Test is disabled for JAX')
    for strategy in strategy_list:
      @hp.settings(deadline=None,
                   max_examples=10,
                   database=None,
                   derandomize=True,
                   suppress_health_check=(hp.HealthCheck.too_slow,))
      @hp.given(strategy)
      def check_consistency(tf_fn, np_fn, args):
        # If `args` is a single item, put it in a tuple
        if isinstance(args, np.ndarray) or tf.is_tensor(args):
          args = (args,)
        tensorflow_value = self.evaluate(
            tf_fn(*_maybe_convert_to_tensors(args)))
        kwargs = jax_kwargs() if MODE_JAX else {}
        numpy_value = np_fn(*args, **kwargs)
        if assert_shape_only:

          def assert_same_shape(x, y):
            self.assertAllEqual(x.shape, y.shape)

          tf.nest.map_structure(assert_same_shape, tensorflow_value,
                                numpy_value)
        else:
          self.assertAllCloseAccordingToType(
              tensorflow_value, numpy_value, atol=atol, rtol=rtol)

      check_consistency(tensorflow_function, numpy_function)


if __name__ == '__main__':
  tf.test.main()
