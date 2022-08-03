# Copyright 2019 The TensorFlow Probability Authors.
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
"""Auto-generate LinearOperator replacements."""

import importlib
import inspect
import re

# Dependency imports

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('module_name', '', 'TF linalg module to transform')
flags.DEFINE_list(
    'allowlist', '',
    'TF linalg module allowlist (other imports will be commented-out)')

MODULE_MAPPINGS = {
    'framework import dtypes': 'dtype as dtypes',
    'framework import errors': 'errors',
    'framework import ops': 'ops',
    'framework import common_shapes': 'ops as common_shapes',
    'framework import tensor_shape': 'tensor_shape',
    'framework import tensor_util': 'ops',
    'module import module': 'ops as module',
    'ops import array_ops': 'numpy_array as array_ops',
    'ops import check_ops': 'debugging as check_ops',
    'ops.signal import fft_ops': 'numpy_signal as fft_ops',
    'ops import control_flow_ops': 'control_flow as control_flow_ops',
    'ops import linalg_ops': 'linalg_impl as linalg_ops',
    'ops import math_ops': 'numpy_math as math_ops',
    'ops import nn': 'nn',
    'ops import sort_ops': 'misc as sort_ops',
    'ops import variables as variables_module': 'ops as variables_module',
    'ops.linalg import linalg_impl as linalg': 'linalg_impl as linalg'
}

COMMENT_OUT = [
    'from tensorflow.python.util import dispatch',
    'from tensorflow.python.util.tf_export',
    'from tensorflow.python.framework import tensor_util',
    '@tf_export',
    '@dispatch',
    '@linear_operator.make_composite_tensor',
    'self._check_input_dtype',
]

UTIL_IMPORTS = """
from tensorflow_probability.python.internal.backend.numpy import private
distribution_util = private.LazyLoader(
    "distribution_util", globals(),
    "tensorflow_probability.substrates.numpy.internal.distribution_util")
tensorshape_util = private.LazyLoader(
    "tensorshape_util", globals(),
    "tensorflow_probability.substrates.numpy.internal.tensorshape_util")
prefer_static = private.LazyLoader(
    "prefer_static", globals(),
    "tensorflow_probability.substrates.numpy.internal.prefer_static")
"""

LINOP_UTIL_SUFFIX = """

JAX_MODE = False
if JAX_MODE:

  def shape_tensor(shape, name=None):  # pylint: disable=unused-argument,function-redefined
    import numpy as onp
    try:
      return onp.array(tuple(int(x) for x in shape), dtype=np.int32)
    except:  # JAX raises raw Exception on __array__  # pylint: disable=bare-except
      pass
    return onp.array(int(shape), dtype=np.int32)
"""

DISABLED_LINTS = ('g-import-not-at-top', 'g-direct-tensorflow-import',
                  'g-bad-import-order', 'unused-import', 'line-too-long',
                  'reimported', 'g-bool-id-comparison',
                  'g-statement-before-imports', 'bad-continuation',
                  'useless-import-alias', 'property-with-parameters',
                  'trailing-whitespace', 'g-inconsistent-quotes')


def gen_module(module_name):
  """Rewrite for numpy the code loaded from the given linalg module."""
  module = importlib.import_module(
      'tensorflow.python.ops.linalg.{}'.format(module_name))
  code = inspect.getsource(module)
  for k, v in MODULE_MAPPINGS.items():
    code = code.replace(
        'from tensorflow.python.{}'.format(k),
        'from tensorflow_probability.python.internal.backend.numpy '
        'import {}'.format(v))
  for k in COMMENT_OUT:
    code = code.replace(k, '# {}'.format(k))
  code = code.replace(
      '.backend.numpy import tensor_shape',
      '.backend.numpy.gen import tensor_shape')
  code = code.replace(
      'from tensorflow.python.platform import tf_logging',
      'from absl import logging')
  code = code.replace(
      'from tensorflow.python.framework import '
      'composite_tensor',
      'from tensorflow_probability.python.internal.backend.numpy '
      'import composite_tensor')
  code = code.replace(
      'from tensorflow.python.ops import '
      'resource_variable_ops',
      'from tensorflow_probability.python.internal.backend.numpy '
      'import resource_variable_ops')
  code = code.replace(
      'from tensorflow.python.framework import tensor_spec',
      'from tensorflow_probability.python.internal.backend.numpy import '
      'tensor_spec')
  code = code.replace(
      'from tensorflow.python.framework import type_spec',
      'from tensorflow_probability.python.internal.backend.numpy '
      'import type_spec')
  code = code.replace(
      'from tensorflow.python.ops import variables',
      'from tensorflow_probability.python.internal.backend.numpy '
      'import variables')
  code = code.replace(
      'from tensorflow.python.trackable '
      'import data_structures',
      'from tensorflow_probability.python.internal.backend.numpy '
      'import data_structures')
  code = code.replace(
      'from tensorflow.python.training.tracking '
      'import data_structures',
      'from tensorflow_probability.python.internal.backend.numpy '
      'import data_structures')
  code = re.sub(
      r'from tensorflow\.python\.linalg import (\w+)',
      'from tensorflow_probability.python.internal.backend.numpy.gen import \\1 '
      'as \\1', code)
  code = code.replace(
      'from tensorflow.python.ops.linalg import ',
      '# from tensorflow.python.ops.linalg import ')
  for f in FLAGS.allowlist:
    code = code.replace(
        '# from tensorflow.python.ops.linalg '
        'import {}'.format(f),
        'from tensorflow.python.ops.linalg '
        'import {}'.format(f))
  code = code.replace(
      'tensorflow.python.ops.linalg import',
      'tensorflow_probability.python.internal.backend.numpy.gen import')
  code = code.replace(
      'tensorflow.python.util import',
      'tensorflow_probability.python.internal.backend.numpy import')
  code = code.replace('tensor_util.constant_value(', 'ops.get_static_value(')
  code = code.replace('tensor_util.is_tensor(', 'ops.is_tensor(')
  code = code.replace('tensor_util.is_tf_type(', 'ops.is_tensor(')
  code = code.replace(
      'from tensorflow.python.ops.distributions import '
      'util as distribution_util', UTIL_IMPORTS)
  code = code.replace(
      'control_flow_ops.with_dependencies',
      'distribution_util.with_dependencies')
  code = code.replace('.base_dtype', '')
  code = code.replace('.get_shape()', '.shape')
  code = re.sub(r'([_a-zA-Z0-9.\[\]]+\.shape)([^(_])',
                'tensor_shape.TensorShape(\\1)\\2', code)
  code = re.sub(r'([_a-zA-Z0-9.\[\]]+).is_floating',
                'np.issubdtype(\\1, np.floating)', code)
  code = re.sub(r'([_a-zA-Z0-9.\[\]]+).is_complex',
                'np.issubdtype(\\1, np.complexfloating)', code)
  code = re.sub(r'([_a-zA-Z0-9.\[\]]+).is_integer',
                'np.issubdtype(\\1, np.integer)', code)

  code = code.replace('array_ops.shape', 'prefer_static.shape')
  code = code.replace('array_ops.concat', 'prefer_static.concat')
  code = code.replace('array_ops.broadcast_dynamic_shape',
                      '_ops.broadcast_dynamic_shape')
  code = code.replace('array_ops.broadcast_static_shape',
                      '_ops.broadcast_static_shape')
  code = code.replace('array_ops.broadcast_to', '_ops.broadcast_to')
  code = code.replace('array_ops.matrix_diag', '_linalg.diag')
  code = code.replace('array_ops.matrix_band_part', '_linalg.band_part')
  code = code.replace('array_ops.matrix_diag_part', '_linalg.diag_part')
  code = code.replace('array_ops.matrix_set_diag', '_linalg.set_diag')
  code = code.replace('array_ops.matrix_transpose', '_linalg.matrix_transpose')
  code = code.replace('array_ops.newaxis', '_ops.newaxis')
  code = code.replace('linalg_ops.matrix_determinant', '_linalg.det')
  code = code.replace('linalg_ops.matrix_solve', '_linalg.solve')
  code = code.replace('linalg_ops.matrix_triangular_solve',
                      'linalg_ops.triangular_solve')
  code = code.replace('math_ops.cast', '_ops.cast')
  code = code.replace('math_ops.matmul', '_linalg.matmul')
  code = code.replace('math_ops.range', 'array_ops.range')
  code = code.replace('ops.convert_to_tensor_v2_with_dispatch(',
                      'ops.convert_to_tensor(')
  code = code.replace('ops.convert_to_tensor(dim_value)',
                      'np.array(dim_value, np.int32)')

  code = code.replace('self.dtype.real_dtype', 'dtypes.real_dtype(self.dtype)')
  code = code.replace('dtype.real_dtype', 'dtypes.real_dtype(dtype)')
  code = code.replace('.as_numpy_dtype', '')

  # Replace `x.set_shape(...)` with `tensorshape_util.set_shape(x, ...)`.
  code = re.sub(r' (\w*)\.set_shape\(',
                ' tensorshape_util.set_shape(\\1, ', code)

  # Replace in-place Python operators (e.g. `+=`) with implicit copying.
  code = re.sub(r'([_a-zA-Z0-9.\[\]]+)[ ]{0,1}(\+|\-|\*|\/)[\=][ ]{0,1}',
                '\\1 = \\1 \\2 ', code)

  for lint in DISABLED_LINTS:
    code = code.replace('pylint: enable={}'.format(lint),
                        'pylint: disable={}'.format(lint))

  print('# Copyright 2020 The TensorFlow Probability Authors. '
        'All Rights Reserved.')
  print('# ' + '@' * 78)
  print('# THIS FILE IS AUTO-GENERATED BY `gen_linear_operators.py`.')
  print('# DO NOT MODIFY DIRECTLY.')
  print('# ' + '@' * 78)
  for lint in DISABLED_LINTS:
    print('# pylint: disable={}'.format(lint))
  print()
  print(code)
  print('import numpy as np')
  print('from tensorflow_probability.python.internal.backend.numpy import '
        'linalg_impl as _linalg')
  print('from tensorflow_probability.python.internal.backend.numpy import '
        'ops as _ops')
  print('from tensorflow_probability.python.internal.backend.numpy.gen import '
        'tensor_shape')
  if module_name == 'linear_operator_util':
    print(LINOP_UTIL_SUFFIX)
  print(UTIL_IMPORTS)


def main(_):
  gen_module(FLAGS.module_name)


if __name__ == '__main__':
  app.run(main)
