# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for internal.backend.numpy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

# Dependency imports

from absl import logging
from absl.testing import parameterized

import hypothesis as hp
import hypothesis.extra.numpy as hnp
import hypothesis.strategies as hps
import numpy as np  # Rewritten by script to import jax.numpy
import numpy as onp  # pylint: disable=reimported
import six
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability.python.experimental.substrates.numpy as tfp

from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.internal.backend import numpy as nptf
from tensorflow_probability.python.internal.backend.numpy import functional_ops as np_pfor
from tensorflow.python.ops import parallel_for as tf_pfor  # pylint: disable=g-direct-tensorflow-import


ALLOW_NAN = False
ALLOW_INFINITY = False

JAX_MODE = False


class Kwargs(dict):
  """Sentinel to indicate a single item arg is actually a **kwargs."""
  # See usage with raw_ops.MatrixDiagPartV2.
  pass


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
        numpy_function=_getattr(
            nptf,
            name.replace('random.', 'random.stateless_'
                        ).replace('random.stateless_gamma', 'random.gamma')),
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


def integers(min_value=-2**30, max_value=2**30):
  return hps.integers(min_value, max_value)


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


def fft_shapes(fft_dim):
  return hps.tuples(
      shapes(max_dims=2),  # batch portion
      hps.lists(min_size=fft_dim, max_size=fft_dim,
                elements=hps.sampled_from([2, 4, 8, 16, 32]))).map(
                    lambda t: t[0] + tuple(t[1]))


@hps.composite
def n_same_shape(draw, n, shape=shapes(), dtype=None, elements=None,
                 as_tuple=True, batch_shape=(), unique=False):
  if elements is None:
    elements = floats()
  if dtype is None:
    dtype = np.float64
  shape = tuple(batch_shape) + draw(shape)

  ensure_array = lambda x: onp.array(x, dtype=dtype)
  if isinstance(elements, (list, tuple)):
    return tuple([
        draw(hnp.arrays(
            dtype, shape, unique=unique, elements=e).map(ensure_array))
        for e in elements
    ])
  array_strategy = hnp.arrays(
      dtype, shape, unique=unique, elements=elements).map(ensure_array)
  if n == 1 and not as_tuple:
    return draw(array_strategy)
  return draw(hps.tuples(*([array_strategy] * n)))


single_arrays = functools.partial(n_same_shape, n=1, as_tuple=False)


@hps.composite
def array_axis_tuples(draw, strategy=None, elements=None):
  x = draw(strategy or single_arrays(shape=shapes(min_dims=1),
                                     elements=elements))
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
  dtype = draw(hps.sampled_from((onp.int32, onp.float32, onp.complex64)))
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
  x_shape = tuple(map(int, x.shape))
  y_shape = x_shape[:-2] + x_shape[-1:] + (draw(hps.integers(1, 10)),)
  y = draw(hnp.arrays(dtype, y_shape, elements=elements))
  return x, y


@hps.composite
def pd_matrices(draw, eps=1.):
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
  mat = draw(pd_matrices())  # pylint: disable=no-value-for-parameter
  signs = draw(
      hnp.arrays(
          mat.dtype,
          tuple(int(dim) for dim in mat.shape[:-2]) + (1, 1),
          elements=hps.sampled_from([-1., 1.])))
  return mat * signs


@hps.composite
def batched_probabilities(draw, batch_shape, num_classes):
  probs = draw(single_arrays(
      batch_shape=batch_shape,
      shape=hps.just((num_classes,)),
      dtype=np.float32, elements=floats()))
  probs = onp.exp(probs - onp.max(
      probs, axis=-1, keepdims=True))
  return probs / probs.sum(keepdims=True, axis=-1)


def tensorshapes_to_tuples(tensorshapes):
  return tuple(tuple(tensorshape.as_list()) for tensorshape in tensorshapes)


@hps.composite
def normal_params(draw):
  shape = draw(shapes())
  arg_shapes = draw(
      tfp_hps.broadcasting_shapes(shape, 3).map(tensorshapes_to_tuples))
  include_arg = draw(hps.lists(hps.booleans(), min_size=2, max_size=2))
  dtype = draw(hps.sampled_from([np.float32, np.float64]))
  mean = (
      draw(single_arrays(shape=hps.just(arg_shapes[1]), dtype=dtype,
                         elements=floats()))
      if include_arg[0] else 0)
  stddev = (
      draw(single_arrays(shape=hps.just(arg_shapes[2]), dtype=dtype,
                         elements=positive_floats()))
      if include_arg[1] else 1)
  return (arg_shapes[0], mean, stddev, dtype)


@hps.composite
def uniform_params(draw):
  shape = draw(shapes())
  arg_shapes = draw(
      tfp_hps.broadcasting_shapes(shape, 3).map(tensorshapes_to_tuples))
  include_arg = draw(hps.lists(hps.booleans(), min_size=2, max_size=2))
  dtype = draw(hps.sampled_from([np.int32, np.int64, np.float32, np.float64]))
  elements = floats(), positive_floats()
  if dtype == np.int32 or dtype == np.int64:
    # TF RandomUniformInt only supports scalar min/max.
    arg_shapes = (arg_shapes[0], (), ())
    elements = integers(), integers(min_value=1)
  minval = (
      draw(single_arrays(shape=hps.just(arg_shapes[1]), dtype=dtype,
                         elements=elements[0]))
      if include_arg[0] else 0)
  maxval = minval + (
      draw(single_arrays(shape=hps.just(arg_shapes[2]), dtype=dtype,
                         elements=elements[1]))
      if include_arg[1] else dtype(10))
  return (arg_shapes[0], minval, maxval, dtype)


def gamma_params():
  def dict_to_params(d):
    return (d['shape'],  # sample shape
            d['params'][0].astype(d['dtype']),  # alpha
            (d['params'][1].astype(d['dtype'])  # beta (or None)
             if d['include_beta'] else None),
            d['dtype'])  # dtype
  return hps.fixed_dictionaries(
      dict(shape=shapes(),
           # hps.composite confuses pylint w/ the draw parameter...
           # pylint: disable=no-value-for-parameter
           params=n_same_shape(n=2, elements=positive_floats()),
           # pylint: enable=no-value-for-parameter
           include_beta=hps.booleans(),
           dtype=hps.sampled_from([np.float32, np.float64]))
      ).map(dict_to_params)  # dtype


@hps.composite
def bincount_params(draw):
  num_buckets = draw(hps.integers(2, 20))
  minlength = draw(hps.one_of(
      hps.just(None),
      hps.integers(num_buckets, num_buckets + 3),
  ))
  arr = draw(single_arrays(dtype=np.int32,
                           shape=hps.just(tuple()),
                           batch_shape=(num_buckets,),
                           elements=hps.integers(
                               0, num_buckets - 1)))
  weights = draw(hps.one_of(
      hps.just(None),
      single_arrays(dtype=np.int32,
                    shape=hps.just(tuple()),
                    batch_shape=(num_buckets,),
                    elements=hps.integers(0, 4))))
  return arr, weights, minlength


@hps.composite
def confusion_matrix_params(draw):
  num_labels = draw(hps.integers(1, 8))
  labels = draw(single_arrays(
      dtype=np.int32,
      shape=hps.just(tuple()),
      batch_shape=(num_labels,),
      elements=hps.integers(0, num_labels - 1)))
  predictions = draw(single_arrays(
      dtype=np.int32,
      shape=hps.just(tuple()),
      batch_shape=(num_labels,),
      elements=hps.integers(0, num_labels - 1)))
  num_classes = draw(hps.one_of(
      hps.just(None),
      hps.integers(num_labels, num_labels + 3)))
  weights = draw(hps.one_of(
      hps.just(None),
      single_arrays(dtype=np.int32,
                    shape=hps.just(tuple()),
                    batch_shape=(num_labels,),
                    elements=hps.integers(0, 4))))
  return labels, predictions, num_classes, weights


@hps.composite
def gather_params(draw):
  params_shape = shapes(min_dims=1)
  params = draw(single_arrays(shape=params_shape))
  rank = len(params.shape)
  # Restricting batch_dims to be positive for now
  # Batch dims can only be > 0 if rank > 1
  batch_dims = draw(hps.integers(0, max(0, rank - 2)))
  # Axis is constrained to be >= batch_dims
  axis = draw(hps.one_of(
      hps.integers(batch_dims, rank - 1),
      hps.integers(-rank + batch_dims, -1),
  ))
  elements = hps.integers(0, params.shape[axis] - 1)
  indices_shape = shapes(min_dims=batch_dims + 1)
  batch_shape = params.shape[:batch_dims]
  indices = draw(single_arrays(dtype=np.int32, elements=elements,
                               shape=indices_shape,
                               batch_shape=batch_shape))
  return params, indices, None, axis, batch_dims


@hps.composite
def gather_nd_params(draw):
  if JAX_MODE:
    # Restricting batch_dims to be positive for now
    batch_dims = draw(hps.integers(min_value=0, max_value=4))
  else:
    batch_dims = 0
  if batch_dims == 0:
    batch_shape = ()
  else:
    batch_shape = draw(shapes(min_dims=batch_dims, max_dims=batch_dims))

  params = draw(single_arrays(
      shape=hps.just(batch_shape + draw(shapes(min_dims=1)))
  ))
  params_shape = params.shape
  rank = len(params_shape)

  indices_shape = draw(hps.integers(min_value=1, max_value=rank - batch_dims))
  indices_batch_shape = draw(shapes())
  batches = []
  for idx in range(indices_shape):
    batches.append(
        draw(single_arrays(
            dtype=np.int32,
            elements=hps.integers(
                0, params.shape[batch_dims + idx] - 1
            ),
            batch_shape=batch_shape + indices_batch_shape,
            shape=hps.just((1,))
        ))
    )
  indices = np.concatenate(batches, -1)
  return params, indices, batch_dims, None


@hps.composite
def repeat_params(draw):
  input_array = draw(single_arrays())
  rank = input_array.ndim
  low, high = -rank, rank - 1
  low, high = min(low, high), max(low, high)
  axis = draw(hps.one_of(hps.just(None), hps.integers(low, high)))
  if draw(hps.booleans()):
    repeats = draw(hps.integers(1, 20))
    if draw(hps.booleans()):
      repeats = np.array([repeats])
    return input_array, repeats, axis
  if rank < 1:
    repeats_shape = draw(hps.one_of(hps.just(()), hps.just((1,))))
  else:
    repeats_shape = (input_array.shape[axis] if axis is not None
                     else np.size(input_array),)
  repeats = draw(hnp.arrays(dtype=np.int32, shape=repeats_shape,
                            elements=hps.integers(1, 20)))
  return input_array, repeats, axis


@hps.composite
def searchsorted_params(draw):
  sorted_array_shape = shapes(min_dims=1)
  sorted_array = draw(single_arrays(shape=sorted_array_shape))
  sorted_array = np.sort(sorted_array)
  num_values = hps.integers(1, 20)
  values = draw(single_arrays(
      shape=shapes(min_dims=1, max_dims=1, max_side=draw(num_values)),
      batch_shape=sorted_array.shape[:-1]))
  search_side = draw(hps.one_of(hps.just('left'), hps.just('right')))
  return sorted_array, values, search_side


@hps.composite
def segment_ids(draw, n):
  lengths = []
  rsum = 0
  while rsum < n:
    lengths.append(draw(hps.integers(1, n-rsum)))
    rsum += lengths[-1]
  return np.repeat(np.arange(len(lengths)), lengths)


@hps.composite
def segment_params(draw, shape=shapes(min_dims=1), dtype=None, elements=None,
                   batch_shape=(), unique=False):
  a = draw(single_arrays(shape=shape, dtype=dtype, elements=elements,
                         batch_shape=batch_shape, unique=unique))
  ids = draw(segment_ids(a.shape[0]))  # pylint: disable=no-value-for-parameter
  return (a, ids)


@hps.composite
def top_k_params(draw):
  array_shape = shapes(min_dims=1)
  # TODO(srvasude): The unique check can be removed once
  # https://github.com/google/jax/issues/2124 is resolved.
  array = draw(single_arrays(unique=True, shape=array_shape))
  k = draw(hps.integers(1, int(array.shape[-1])))
  return array, k


@hps.composite
def histogram_fixed_width_bins_params(draw):
  values = draw(single_arrays(
      dtype=np.float32,
      shape=shapes(min_dims=1),
      elements=hps.floats(min_value=-1e5, max_value=1e5)
  ))
  vmin, vmax = np.min(values), np.max(values)
  value_min = draw(hps.one_of(
      hps.just(vmin),
      hps.just(vmin - 3))).astype(np.float32)
  value_max = draw(hps.one_of(
      hps.just(vmax),
      hps.just(vmax + 3))).astype(np.float32)
  nbins = draw(hps.integers(2, 10))
  return values, [value_min, value_max], nbins


@hps.composite
def sparse_xent_params(draw):
  num_classes = draw(hps.integers(1, 6))
  batch_shape = draw(shapes(min_dims=1))
  labels = single_arrays(
      batch_shape=batch_shape,
      shape=hps.just(tuple()),
      dtype=np.int32,
      elements=hps.integers(0, num_classes - 1))
  logits = single_arrays(
      batch_shape=batch_shape,
      shape=hps.just((num_classes,)),
      elements=hps.floats(min_value=-1e5, max_value=1e5))
  return draw(
      hps.fixed_dictionaries(dict(
          labels=labels, logits=logits)).map(Kwargs))


@hps.composite
def xent_params(draw):
  num_classes = draw(hps.integers(1, 6))
  batch_shape = draw(shapes(min_dims=1))
  labels = batched_probabilities(  # pylint:disable=no-value-for-parameter
      batch_shape=batch_shape, num_classes=num_classes)
  logits = single_arrays(
      batch_shape=batch_shape,
      shape=hps.just((num_classes,)),
      elements=hps.floats(min_value=-1e5, max_value=1e5))
  return draw(
      hps.fixed_dictionaries(dict(
          labels=labels, logits=logits)).map(Kwargs))


def _svd_post_process(vals):
  # SVDs are not unique, so reconstruct input to test consistency (b/154538680).

  # create_uv = False
  if not isinstance(vals, tuple):
    return vals
  # create_uv = True
  s, u, v = (np.array(x) for x in vals)
  return np.matmul(
      u,
      s[..., None] *
      # Vectorized matrix transpose.
      np.swapaxes(v, -2, -1))


# __Currently untested:__
# broadcast_dynamic_shape
# broadcast_static_shape
# broadcast_to
# math.accumulate_n
# math.betainc
# math.igamma
# math.igammac
# math.lbeta
# math.polyval
# math.zeta
# random.poisson
# random.set_seed


# TODO(jamieas): add tests for these functions.

# pylint: disable=no-value-for-parameter

NUMPY_TEST_CASES = [

    TestCase('signal.fft',
             [single_arrays(shape=fft_shapes(fft_dim=1),
                            dtype=np.complex64,
                            elements=complex_numbers(max_magnitude=1e3))],
             atol=1e-4, rtol=1e-4),
    TestCase('signal.fft2d',
             [single_arrays(shape=fft_shapes(fft_dim=2),
                            dtype=np.complex64,
                            elements=complex_numbers(max_magnitude=1e3))],
             atol=1e-4, rtol=1e-4),
    TestCase('signal.fft3d',
             [single_arrays(shape=fft_shapes(fft_dim=3),
                            dtype=np.complex64,
                            elements=complex_numbers(max_magnitude=1e3))],
             atol=1e-3, rtol=1e-3),
    TestCase('signal.rfft',
             [single_arrays(shape=fft_shapes(fft_dim=1),
                            dtype=np.float32,
                            elements=floats(min_value=-1e3, max_value=1e3))],
             atol=1e-4, rtol=1e-4),
    TestCase('signal.rfft2d',
             [single_arrays(shape=fft_shapes(fft_dim=2),
                            dtype=np.float32,
                            elements=floats(min_value=-1e3, max_value=1e3))],
             atol=2e-4, rtol=2e-4),
    TestCase('signal.rfft3d',
             [single_arrays(shape=fft_shapes(fft_dim=3),
                            dtype=np.float32,
                            elements=floats(min_value=-1e3, max_value=1e3))],
             atol=1e-3, rtol=1e-3),
    TestCase('signal.ifft',
             [single_arrays(shape=fft_shapes(fft_dim=1),
                            dtype=np.complex64,
                            elements=complex_numbers(max_magnitude=1e3))],
             atol=1e-4, rtol=1e-4),
    TestCase('signal.ifft2d',
             [single_arrays(shape=fft_shapes(fft_dim=2),
                            dtype=np.complex64,
                            elements=complex_numbers(max_magnitude=1e3))],
             atol=1e-4, rtol=1e-4),
    TestCase('signal.ifft3d',
             [single_arrays(shape=fft_shapes(fft_dim=3),
                            dtype=np.complex64,
                            elements=complex_numbers(max_magnitude=1e3))],
             atol=1e-4, rtol=1e-4),
    TestCase('signal.irfft',
             [single_arrays(shape=fft_shapes(fft_dim=1),
                            dtype=np.complex64,
                            elements=complex_numbers(max_magnitude=1e3))],
             atol=3e-4, rtol=3e-4),
    TestCase('signal.irfft2d',
             [single_arrays(shape=fft_shapes(fft_dim=2),
                            dtype=np.complex64,
                            elements=complex_numbers(max_magnitude=5e2))],
             atol=2e-4, rtol=2e-4),
    TestCase('signal.irfft3d',
             [single_arrays(shape=fft_shapes(fft_dim=3),
                            dtype=np.complex64,
                            elements=complex_numbers(max_magnitude=1e3))],
             atol=4e-4, rtol=4e-4),

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
    TestCase('linalg.trace', [nonsingular_matrices()]),

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
    TestCase('math.bincount', [bincount_params()]),
    TestCase('math.confusion_matrix', [confusion_matrix_params()]),
    TestCase('math.top_k', [top_k_params()]),

    # ArgSpec(args=['chol', 'rhs', 'name'], varargs=None, keywords=None,
    #         defaults=(None,))
    TestCase('linalg.cholesky_solve', [
        matmul_compatible_pairs(
            x_strategy=pd_matrices().map(np.linalg.cholesky))
    ]),

    # ArgSpec(args=['tensor', 'full_matrices', 'compute_uv', 'name'],
    #         varargs=None,
    #         keywords=None,
    #         defaults=(False, True, None))
    TestCase('linalg.svd',
             [single_arrays(shape=shapes(min_dims=2))],
             post_processor=_svd_post_process),

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
    TestCase('math.argmax', [array_axis_tuples()]),
    TestCase('math.argmin', [array_axis_tuples()]),

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
    TestCase('linalg.cholesky', [pd_matrices()]),
    TestCase('linalg.lu', [nonsingular_matrices()]),
    TestCase('linalg.diag_part', [single_arrays(shape=shapes(min_dims=2))]),
    TestCase('raw_ops.MatrixDiagPartV2', [
        hps.fixed_dictionaries(dict(
            input=single_arrays(shape=shapes(min_dims=2, min_side=2)),
            k=hps.sampled_from([-1, 0, 1]),
            padding_value=hps.just(0.))).map(Kwargs)]),
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
        array_axis_tuples(
            single_arrays(
                shape=shapes(min_dims=1),
                dtype=np.bool,
                elements=hps.booleans()))
    ]),
    TestCase('math.reduce_any', [
        array_axis_tuples(
            single_arrays(
                shape=shapes(min_dims=1),
                dtype=np.bool,
                elements=hps.booleans()))
    ]),
    TestCase('math.reduce_logsumexp', [array_axis_tuples()]),
    TestCase('math.reduce_max', [array_axis_tuples()]),
    TestCase('math.reduce_mean', [array_axis_tuples()]),
    TestCase('math.reduce_min', [array_axis_tuples()]),
    TestCase('math.reduce_prod', [array_axis_tuples()]),
    TestCase('math.reduce_std',
             [array_axis_tuples(elements=floats(-1e6, 1e6))]),
    TestCase('math.reduce_sum', [array_axis_tuples()]),
    TestCase('math.reduce_variance',
             [array_axis_tuples(elements=floats(-1e6, 1e6))]),

    TestCase('math.segment_max', [segment_params()]),
    TestCase('math.segment_mean',
             [segment_params()],
             # need jax.numpy.bincount
             jax_disabled=True),
    TestCase('math.segment_min', [segment_params()]),
    TestCase('math.segment_prod', [segment_params()]),
    TestCase('math.segment_sum', [segment_params()]),

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
            x_strategy=pd_matrices().map(np.linalg.cholesky))
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
        hps.tuples(array_axis_tuples(), hps.booleans(),
                   hps.booleans()).map(lambda x: x[0] + (x[1], x[2]))
    ]),
    TestCase('math.cumsum', [
        hps.tuples(array_axis_tuples(), hps.booleans(),
                   hps.booleans()).map(lambda x: x[0] + (x[1], x[2]))
    ]),

]

NUMPY_TEST_CASES += [  # break the array for pylint to not timeout.

    # args=['input', 'name']
    TestCase('linalg.adjoint',
             [single_arrays(shape=shapes(min_dims=2),
                            dtype=np.complex64, elements=complex_numbers())]),
    TestCase('linalg.slogdet', [nonsingular_matrices()]),
    # ArgSpec(args=['x', 'name'], varargs=None, keywords=None, defaults=(None,))
    TestCase('complex', [n_same_shape(n=2, dtype=np.float32),
                         n_same_shape(n=2, dtype=np.float64)]),
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
        'math.bessel_i0e', [single_arrays(elements=floats(-50., 50.))]),
    TestCase(
        'math.bessel_i1', [single_arrays(elements=floats(-50., 50.))],
        jax_disabled=True),
    TestCase(
        'math.bessel_i1e', [single_arrays(elements=floats(-50., 50.))]),
    TestCase('math.ceil', [single_arrays()]),
    TestCase('math.conj',
             [single_arrays(dtype=np.complex64, elements=complex_numbers())]),
    TestCase('math.cos', [single_arrays()]),
    TestCase('math.cosh', [single_arrays(elements=floats(-100., 100.))]),
    TestCase('math.digamma',
             [single_arrays(elements=non_zero_floats(-1e4, 1e4))]),
    TestCase('math.erf', [single_arrays()]),
    TestCase('math.erfc', [single_arrays()]),
    TestCase('math.erfinv', [single_arrays(elements=floats(-1., 1.))]),
    TestCase('math.exp',  # TODO(b/147394924): max_value=1e3
             [single_arrays(elements=floats(min_value=-1e3, max_value=85))]),
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
    TestCase('math.ndtri', [single_arrays(elements=floats(0., 1.))]),
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
    TestCase('math.floormod',
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
    TestCase('math.xlog1py',
             [n_same_shape(n=2, elements=[floats(), positive_floats()])]),
    TestCase('nn.sparse_softmax_cross_entropy_with_logits',
             [sparse_xent_params()], rtol=1e-4, atol=1e-4),
    TestCase('nn.softmax_cross_entropy_with_logits',
             [xent_params()], rtol=1e-4, atol=1e-4),
    TestCase(
        'random.categorical', [
            hps.tuples(
                single_arrays(
                    shape=shapes(min_dims=2, max_dims=2),
                    elements=floats(min_value=-1e3, max_value=1e3)),
                hps.integers(0, 10))
        ],
        jax_kwargs=_add_jax_prng_key_as_seed,
        assert_shape_only=True),
    TestCase(
        'random.gamma', [gamma_params()],
        jax_kwargs=_add_jax_prng_key_as_seed,
        assert_shape_only=True),
    TestCase(
        'random.normal', [normal_params()],
        jax_kwargs=_add_jax_prng_key_as_seed,
        assert_shape_only=True),
    TestCase(
        'random.uniform', [uniform_params()],
        jax_kwargs=_add_jax_prng_key_as_seed,
        assert_shape_only=True),

    # Array ops.
    TestCase('gather', [gather_params()]),
    TestCase('gather_nd', [gather_nd_params()]),
    # TODO(leben): Fix bug in jax.numpy.repeat(array(0), 1, 0).
    TestCase('repeat', [repeat_params()],
             jax_disabled=True),
    TestCase('searchsorted', [searchsorted_params()]),
    TestCase('one_hot', [one_hot_params()]),
    TestCase('slice', [sliceable_and_slices()]),

    # Misc
    TestCase('histogram_fixed_width_bins',
             [histogram_fixed_width_bins_params()]),
]


def _maybe_convert_to_tensors(args):
  # Ensures we go from JAX np -> original np -> tf.Tensor. (no-op for non-JAX.)
  convert = lambda a: tf.convert_to_tensor(onp.array(a))
  return tf.nest.map_structure(
      lambda arg: convert(arg) if isinstance(arg, np.ndarray) else arg,
      args)


class NumpyTest(test_util.TestCase):

  def _base_test_convert_to_tensor(self, nmpy):
    convert_to_tensor = nptf.convert_to_tensor
    self.assertEqual(
        nmpy.complex64,
        convert_to_tensor(nmpy.complex64(1 + 2j), dtype_hint=tf.int32).dtype)
    self.assertEqual(
        nmpy.complex64,
        convert_to_tensor(nmpy.complex64(1 + 2j), dtype_hint=tf.float64).dtype)
    self.assertEqual(nmpy.float64,
                     convert_to_tensor(1., dtype_hint=tf.int32).dtype)
    self.assertEqual(
        nmpy.int32, convert_to_tensor(1, dtype_hint=tf.int32).dtype)
    self.assertEqual(nmpy.float32,
                     convert_to_tensor(1, dtype_hint=tf.float32).dtype)
    self.assertEqual(nmpy.complex64,
                     convert_to_tensor(1., dtype_hint=tf.complex64).dtype)
    self.assertEqual(
        nmpy.int64, convert_to_tensor(1, dtype_hint=tf.int64).dtype)
    self.assertEqual(
        nmpy.int32,
        convert_to_tensor(nmpy.int32(False), dtype_hint=tf.bool).dtype)

  def test_convert_to_tensor(self):
    self._base_test_convert_to_tensor(np)

  def test_convert_to_tensor_numpy_array(self):
    if not JAX_MODE:
      self.skipTest('Check non-device arrays in JAX.')
    self._base_test_convert_to_tensor(onp)

  def test_convert_to_tensor_scalar_default(self):
    convert_to_tensor = nptf.convert_to_tensor
    self.assertEqual(np.complex128, convert_to_tensor(1. + 2j).dtype)
    self.assertEqual(np.float32, convert_to_tensor(1.).dtype)
    self.assertEqual(np.int32, convert_to_tensor(1).dtype)

  def test_convert_to_tensor_dimension(self):
    convert_to_tensor = nptf.convert_to_tensor
    shape = tf1.Dimension(1)

    tensor_shape = convert_to_tensor(shape)
    self.assertNotIsInstance(tensor_shape, tf1.Dimension)

  def test_convert_to_tensor_tensorshape(self):
    convert_to_tensor = nptf.convert_to_tensor
    shape = tf.TensorShape((1, 2))

    tensor_shape = convert_to_tensor(shape)
    for dim in tensor_shape:
      self.assertNotIsInstance(dim, tf1.Dimension)

    shape = tf.TensorShape((1, 2, 3))[:2]
    tensor_shape = convert_to_tensor(shape)

    for dim in tensor_shape:
      self.assertNotIsInstance(dim, tf1.Dimension)

  def test_concat_infers_dtype(self):
    self.assertEqual(np.int32, nptf.concat([[1], []], 0).dtype)
    self.assertEqual(np.float32, nptf.concat([[], [1]], 0).dtype)
    self.assertEqual(np.float32, nptf.concat([np.array([1], np.float32),
                                              np.array([1], np.float64)],
                                             0).dtype)
    self.assertEqual(np.float64, nptf.concat([np.array([1], np.float64),
                                              np.array([1], np.float32)],
                                             0).dtype)
    self.assertEqual(np.float32, nptf.concat([[np.float32(1)], [np.float64(1)]],
                                             0).dtype)
    self.assertEqual(np.float32, nptf.concat([[np.float64(1)], [np.float32(1)]],
                                             0).dtype)
    # TODO(sharadmv): rewrite these tests when convert_to_tensor is fixed
    self.assertEqual(np.int32, nptf.concat([[np.int32(1)], [np.int64(1)]],
                                           0).dtype)
    self.assertEqual(np.int32, nptf.concat([[np.int64(1)], [np.int32(1)]],
                                           0).dtype)

  @test_util.numpy_disable_gradient_test
  def test_while_loop_gradients(self):

    def _fn(x):

      def _cond_fn(i, _):
        return i < 3.

      def _body_fn(i, val):
        return i + 1, val + 1.

      return nptf.while_loop(
          cond=_cond_fn, body=_body_fn, loop_vars=(0, x),
          maximum_iterations=5)[1]

    _, grad = tfp.math.value_and_gradient(_fn, 0.)
    self.assertIsNotNone(grad)

  def test_scan_no_initializer(self):
    elems = np.arange(5).astype(np.int32)
    self.assertAllEqual(
        self.evaluate(tf.scan(lambda x, y: x + y, elems)),
        nptf.scan(lambda x, y: x + y, elems))

  def test_scan_with_initializer(self):
    elems = np.arange(5).astype(np.int32)
    self.assertAllEqual(
        self.evaluate(tf.scan(lambda x, y: x + y, elems, initializer=7)),
        nptf.scan(lambda x, y: x + y, elems, initializer=7))

  def test_scan_with_struct(self):
    elems = np.arange(5).astype(np.int32)
    self.assertAllEqual(
        self.evaluate(tf.scan(
            lambda x, y: (x[0] + y, x[1] - y), elems, initializer=(7, 3))),
        nptf.scan(lambda x, y: (x[0] + y, x[1] - y), elems, initializer=(7, 3)))

  def test_scan_with_struct_elems(self):
    elems = (np.arange(5).astype(np.int32),
             np.arange(10).astype(np.int32).reshape(5, 2))
    init = (np.int32([7, 8]), np.int32([9, 1]))
    self.assertAllEqual(
        self.evaluate(tf.scan(
            lambda x, y: (x[0] + y[0], x[1] - y[1]), elems, initializer=init)),
        nptf.scan(
            lambda x, y: (x[0] + y[0], x[1] - y[1]), elems, initializer=init))

  def test_scan_with_struct_elems_reverse(self):
    elems = (np.arange(5).astype(np.int32),
             np.arange(10).astype(np.int32).reshape(5, 2))
    init = (np.int32([7, 8]), np.int32([9, 1]))
    self.assertAllEqual(
        self.evaluate(tf.scan(
            lambda x, y: (x[0] + y[0], x[1] - y[1]), elems, initializer=init,
            reverse=True)),
        nptf.scan(
            lambda x, y: (x[0] + y[0], x[1] - y[1]), elems, initializer=init,
            reverse=True))

  def test_pfor(self):
    self.assertAllEqual(
        self.evaluate(tf_pfor.pfor(lambda x: tf.ones([]), 7)),
        np_pfor.pfor(lambda x: nptf.ones([]), 7))

  def test_pfor_with_closure(self):
    val = np.arange(7.)[:, np.newaxis]
    tf_val = tf.constant(val)
    def tf_fn(x):
      return tf.gather(tf_val, x)**2
    def np_fn(x):
      return nptf.gather(val, x)**2
    self.assertAllEqual(
        self.evaluate(tf_pfor.pfor(tf_fn, 7)),
        np_pfor.pfor(np_fn, 7))

  def test_pfor_with_closure_multi_out(self):
    val = np.arange(7.)[:, np.newaxis]
    tf_val = tf.constant(val)
    def tf_fn(x):
      return tf.gather(tf_val, x)**2, tf.gather(tf_val, x)
    def np_fn(x):
      return nptf.gather(val, x)**2, nptf.gather(val, x)
    self.assertAllEqual(
        self.evaluate(tf_pfor.pfor(tf_fn, 7)),
        np_pfor.pfor(np_fn, 7))

  def evaluate(self, tensors):
    if tf.executing_eagerly():
      return self._eval_helper(tensors)
    else:
      sess = tf1.get_default_session()
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
    if jax_disabled and JAX_MODE:
      logging.warning('The test for %s is disabled for JAX.',
                      numpy_function.__name__)
    elif not strategy_list:
      logging.warning(
          'The test for %s contains no strategies.', numpy_function.__name__)
    else:
      self.skipTest('Has coverage.')

  @parameterized.named_parameters(NUMPY_TEST_CASES)
  def testConsistency(self,
                      tensorflow_function,
                      numpy_function,
                      strategy_list,
                      atol=1e-5,
                      rtol=1e-5,
                      jax_disabled=False,
                      assert_shape_only=False,
                      post_processor=None,
                      jax_kwargs=lambda: {}):
    if jax_disabled and JAX_MODE:
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
        kwargs = {}
        if isinstance(args, Kwargs):
          kwargs = args
          args = ()
        tensorflow_value = self.evaluate(
            tf_fn(*_maybe_convert_to_tensors(args),
                  **_maybe_convert_to_tensors(kwargs)))

        kwargs.update(jax_kwargs() if JAX_MODE else {})
        numpy_value = np_fn(*args, **kwargs)
        if post_processor is not None:
          numpy_value = post_processor(numpy_value)
          tensorflow_value = post_processor(tensorflow_value)
        if assert_shape_only:

          def assert_same_shape(x, y):
            self.assertAllEqual(x.shape, y.shape)

          tf.nest.map_structure(assert_same_shape, tensorflow_value,
                                numpy_value)
        else:
          for i, (tf_val, np_val) in enumerate(six.moves.zip_longest(
              tf.nest.flatten(tensorflow_value), tf.nest.flatten(numpy_value))):
            self.assertAllCloseAccordingToType(
                tf_val, np_val, atol=atol, rtol=rtol,
                msg='output {}'.format(i))

      check_consistency(tensorflow_function, numpy_function)


if __name__ == '__main__':
  tf.test.main()
