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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import inspect
import re

# Dependency imports

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('module_name', '', 'TF linalg module to transform')
flags.DEFINE_list(
    'whitelist', '',
    'TF linalg module whitelist (other imports will be commented-out)')

MODULE_MAPPINGS = {
    'framework import dtypes': 'dtype as dtypes',
    'framework import errors': 'errors',
    'framework import ops': 'ops',
    'framework import tensor_shape': 'ops as tensor_shape',
    'module import module': 'ops as module',
    'ops import array_ops': 'numpy_array as array_ops',
    'ops import check_ops': 'debugging as check_ops',
    'ops.signal import fft_ops': 'numpy_signal as fft_ops',
    'ops import control_flow_ops': 'control_flow as control_flow_ops',
    'ops import linalg_ops': 'linalg_impl as linalg_ops',
    'ops import math_ops': 'numpy_math as math_ops',
    'ops import variables as variables_module': 'ops as variables_module',
    'ops.linalg import linalg_impl as linalg': 'linalg_impl as linalg'
}

COMMENT_OUT = [
    'from tensorflow.python.util import dispatch',
    'from tensorflow.python.util.tf_export',
    'from tensorflow.python.framework import tensor_util',
    '@tf_export',
    '@dispatch',
    'self._check_input_dtype',
]

DIST_UTIL_IMPORT = """
from tensorflow.python.util import lazy_loader
distribution_util = lazy_loader.LazyLoader(
    "distribution_util", globals(),
    "tensorflow_probability.python.internal._numpy.distribution_util")
"""


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
      'from tensorflow.python.platform import tf_logging',
      'from absl import logging')
  code = re.sub(
      r'from tensorflow\.python\.linalg import (\w+)',
      'from tensorflow_probability.python.internal.backend.numpy import \\1 '
      'as \\1', code)
  code = code.replace(
      'from tensorflow.python.ops.linalg import ',
      '# from tensorflow.python.ops.linalg import ')
  for f in FLAGS.whitelist:
    code = code.replace(
        '# from tensorflow.python.ops.linalg '
        'import {}'.format(f),
        'from tensorflow.python.ops.linalg '
        'import {}'.format(f))
  code = code.replace(
      'tensorflow.python.ops.linalg import ',
      'tensorflow_probability.python.internal.backend.numpy import ')

  code = code.replace('tensor_util.constant_value(', '(')
  code = code.replace('tensor_util.is_tensor(', 'ops.is_tensor(')
  code = code.replace(
      'from tensorflow.python.ops.distributions import '
      'util as distribution_util', DIST_UTIL_IMPORT)
  code = code.replace(
      'control_flow_ops.with_dependencies',
      'distribution_util.with_dependencies')
  code = code.replace('.base_dtype', '')
  code = code.replace('.get_shape()', '.shape')
  code = re.sub(r'([_a-zA-Z0-9.\[\]]+\.shape)([^(_])',
                '_ops.TensorShape(\\1)\\2', code)
  code = re.sub(r'([_a-zA-Z0-9.\[\]]+).is_complex',
                'np.issubdtype(\\1, np.complexfloating)', code)
  code = re.sub(r'([_a-zA-Z0-9.\[\]]+).is_integer',
                'np.issubdtype(\\1, np.integer)', code)

  code = code.replace('array_ops.broadcast_static_shape',
                      '_ops.broadcast_static_shape_as_tensorshape')
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

  code = code.replace('self.dtype.real_dtype', 'dtypes.real_dtype(self.dtype)')
  code = code.replace('dtype.real_dtype', 'dtypes.real_dtype(dtype)')
  code = code.replace('.as_numpy_dtype', '')

  print(code)
  print('import numpy as np')
  print('from tensorflow_probability.python.internal.backend.numpy import '
        'linalg_impl as _linalg')
  print('from tensorflow_probability.python.internal.backend.numpy import '
        'ops as _ops')
  print(DIST_UTIL_IMPORT)


def main(_):
  gen_module(FLAGS.module_name)


if __name__ == '__main__':
  app.run(main)
