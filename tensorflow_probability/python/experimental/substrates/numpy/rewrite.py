# Copyright 2019 The TensorFlow Probability Authors.
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
"""Rewrite script for TF->JAX."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
if not sys.path[0].endswith('.runfiles'):
  sys.path.pop(0)

# pylint: disable=g-import-not-at-top,g-bad-import-order
import collections

# Dependency imports
from absl import app
from absl import flags
# pylint: enable=g-import-not-at-top,g-bad-import-order

flags.DEFINE_boolean('numpy_to_jax', False,
                     'Whether or not to rewrite numpy imports to jax.numpy')

FLAGS = flags.FLAGS

TF_REPLACEMENTS = {
    'import tensorflow ':
        'from tensorflow_probability.python.internal.backend import numpy ',
    'import tensorflow.compat.v1':
        'from tensorflow_probability.python.internal.backend.numpy.compat '
        'import v1',
    'import tensorflow.compat.v2':
        'from tensorflow_probability.python.internal.backend.numpy.compat '
        'import v2',
    'import tensorflow_probability as tfp':
        'import tensorflow_probability as tfp; '
        'tfp = tfp.experimental.substrates.numpy',
}

DISABLED_BIJECTORS = ('masked_autoregressive', 'matveclu', 'real_nvp')
DISABLED_DISTS = ('joint_distribution', 'gaussian_process',
                  'internal.moving_stats', 'student_t_process',
                  'variational_gaussian_process', 'von_mises')
LIBS = ('bijectors', 'distributions', 'math', 'stats', 'util.seed_stream')
INTERNALS = ('assert_util', 'distribution_util', 'dtype_util',
             'hypothesis_testlib', 'prefer_static', 'special_math',
             'tensor_util', 'test_case', 'test_util')


def main(argv):

  replacements = collections.OrderedDict(TF_REPLACEMENTS)
  replacements.update({
      'from tensorflow_probability.python.bijectors.{}'.format(bijector):
      '# from tensorflow_probability.python.bijectors.{}'.format(bijector)
      for bijector in DISABLED_BIJECTORS
  })
  replacements.update({
      'from tensorflow_probability.python.distributions.{}'.format(dist):
      '# from tensorflow_probability.python.distributions.{}'.format(dist)
      for dist in DISABLED_DISTS
  })
  substrates_pkg = 'tensorflow_probability.python.experimental.substrates'
  replacements.update({
      'tensorflow_probability.python.{}'.format(lib):
      '{}.numpy.{}'.format(substrates_pkg, lib)
      for lib in LIBS
  })
  replacements.update({
      'tensorflow_probability.python import {}'.format(lib):
      '{}.numpy import {}'.format(substrates_pkg, lib)
      for lib in LIBS
  })
  replacements.update({
      'tensorflow_probability.python.internal.{}'.format(internal):
      '{}.numpy.internal.{}'.format(substrates_pkg, internal)
      for internal in INTERNALS
  })
  replacements.update({
      'tensorflow_probability.python.internal import {}'.format(internal):
      '{}.numpy.internal import {}'.format(substrates_pkg, internal)
      for internal in INTERNALS
  })
  replacements.update({
      'self._maybe_assert_dtype': '# self._maybe_assert_dtype',
      'SKIP_DTYPE_CHECKS = False': 'SKIP_DTYPE_CHECKS = True',
      '@test_util.run_all_in_graph_and_eager_modes': (
          '# @test_util.run_all_in_graph_and_eager_modes'),
  })

  contents = open(argv[1]).read()
  for find, replace in replacements.items():
    contents = contents.replace(find, replace)
  if FLAGS.numpy_to_jax:
    contents = contents.replace('substrates.numpy', 'substrates.jax')
    contents = contents.replace('substrates import numpy',
                                'substrates import jax')
    contents = contents.replace('backend.numpy', 'backend.jax')
    contents = contents.replace('backend import numpy', 'backend import jax')
    contents = contents.replace(
        'tf.test.main()',
        'from jax.config import config; '
        'config.update("jax_enable_x64", True); '
        'tf.test.main()')
    contents = contents.replace('def _call_jax', 'def __call__')
    contents = contents.replace('JAX_MODE = False', 'JAX_MODE = True')

  print(contents)


if __name__ == '__main__':
  app.run(main)
