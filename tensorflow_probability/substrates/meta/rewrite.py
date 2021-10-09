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

import collections

# Dependency imports
from absl import app
from absl import flags

flags.DEFINE_boolean('numpy_to_jax', False,
                     'Whether or not to rewrite numpy imports to jax.numpy')
flags.DEFINE_list('omit_deps', [], 'List of build deps being omitted.')

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
        'tfp = tfp.substrates.numpy',
    ('from tensorflow.python.framework '
     'import composite_tensor'):
        ('from tensorflow_probability.python.internal.backend.numpy '
         'import composite_tensor'),
    'from tensorflow.python.framework import tensor_shape':
        ('from tensorflow_probability.python.internal.backend.numpy.gen '
         'import tensor_shape'),
    'from tensorflow.python.framework import ops':
        ('from tensorflow_probability.python.internal.backend.numpy '
         'import ops'),
    'from tensorflow.python.framework import tensor_util':
        ('from tensorflow_probability.python.internal.backend.numpy '
         'import ops'),
    'from tensorflow.python.framework import tensor_spec':
        ('from tensorflow_probability.python.internal.backend.numpy '
         'import tensor_spec'),
    'from tensorflow.python.framework import type_spec':
        ('from tensorflow_probability.python.internal.backend.numpy '
         'import type_spec'),
    ('from tensorflow.python.ops import '
     'resource_variable_ops'):
        ('from tensorflow_probability.python.internal.backend.numpy '
         'import resource_variable_ops'),
    'from tensorflow.python.util import':
        'from tensorflow_probability.python.internal.backend.numpy import',
    'from tensorflow.python.util.all_util':
        'from tensorflow_probability.python.internal.backend.numpy.private',
    'from tensorflow.python.ops.linalg':
        'from tensorflow_probability.python.internal.backend.numpy.gen',
    'from tensorflow.python.ops import parallel_for':
        'from tensorflow_probability.python.internal.backend.numpy '
        'import functional_ops as parallel_for',
    'from tensorflow.python.ops import control_flow_ops':
        'from tensorflow_probability.python.internal.backend.numpy '
        'import control_flow as control_flow_ops',
    ('from tensorflow.python.saved_model '
     'import nested_structure_coder'):
        ('from tensorflow_probability.python.internal.backend.numpy '
         'import nested_structure_coder'),
    'from tensorflow.python.eager import context':
        'from tensorflow_probability.python.internal.backend.numpy '
        'import private',
    ('from tensorflow.python.client '
     'import pywrap_tf_session as c_api'):
        'pass',
    ('from tensorflow.python '
     'import pywrap_tensorflow as c_api'):
        'pass'
}

DISABLED_BY_PKG = {
    'experimental':
        ('auto_batching', 'composite_tensor', 'linalg',
         'marginalize', 'nn', 'sequential', 'substrates', 'vi'),
}
LIBS = ('bijectors', 'distributions', 'experimental', 'math', 'mcmc',
        'optimizer', 'random', 'staging', 'stats', 'util')
INTERNALS = ('assert_util', 'auto_composite_tensor',
             'batched_rejection_sampler',
             'batch_shape_lib', 'broadcast_util', 'cache_util',
             'callable_util', 'custom_gradient', 'distribution_util',
             'distribute_lib', 'distribute_test_lib',
             'dtype_util', 'hypothesis_testlib', 'implementation_selection',
             'monte_carlo', 'name_util', 'nest_util', 'numerics_testing',
             'parameter_properties', 'prefer_static', 'samplers', 'slicing',
             'special_math', 'structural_tuple', 'tensor_util',
             'tensorshape_util', 'test_combinations', 'test_util', 'unnest',
             'variadic_reduce', 'vectorization_util')
OPTIMIZERS = ('linesearch',)
LINESEARCH = ('internal',)
SAMPLERS = ('categorical', 'normal', 'poisson', 'uniform', 'shuffle')

PRIVATE_TF_PKGS = ('array_ops', 'control_flow_util', 'gradient_checker_v2',
                   'numpy_text', 'random_ops')


def main(argv):

  disabled_by_pkg = dict(DISABLED_BY_PKG)
  for dep in FLAGS.omit_deps:
    if '/python/' in dep:
      pkg = 'python.' + dep.split('/python/')[1].split(':')[0].replace('/', '.')
    elif '/google/' in dep:
      pkg = 'google.' + dep.split('/google/')[1].split(':')[0].replace('/', '.')
    lib = dep.split(':')[1]
    if pkg.endswith('.{}'.format(lib)):
      pkg = pkg.replace('.{}'.format(lib), '')
      disabled_by_pkg.setdefault(pkg, ())
      disabled_by_pkg[pkg] += (lib,)
    else:
      disabled_by_pkg.setdefault(pkg, ())
      disabled_by_pkg[pkg] += (lib,)

  replacements = collections.OrderedDict(TF_REPLACEMENTS)
  for pkg, disabled in disabled_by_pkg.items():
    replacements.update({
        'from tensorflow_probability.{}.{} '.format(pkg, item):
        '# from tensorflow_probability.{}.{} '.format(pkg, item)
        for item in disabled
    })
    replacements.update({
        'from tensorflow_probability.{} import {}'.format(pkg, item):
        '# from tensorflow_probability.{} import {}'.format(pkg, item)
        for item in disabled
    })
  replacements.update({
      'tensorflow_probability.python.{}'.format(lib):
      'tensorflow_probability.substrates.numpy.{}'.format(lib)
      for lib in LIBS
  })
  replacements.update({
      'tensorflow_probability.python import {}'.format(lib):
      'tensorflow_probability.substrates.numpy import {}'.format(lib)
      for lib in LIBS
  })
  replacements.update({
      'tensorflow_probability.google.{}'.format(lib):
      'tensorflow_probability.substrates.numpy.google.{}'.format(lib)
      for lib in LIBS
  })
  replacements.update({
      'tensorflow_probability.google import {}'.format(lib):
      'tensorflow_probability.substrates.numpy.google import {}'.format(lib)
      for lib in LIBS
  })
  replacements.update({
      'tensorflow_probability.python.internal.{}'.format(internal):
      'tensorflow_probability.substrates.numpy.internal.{}'.format(internal)
      for internal in INTERNALS
  })
  # pylint: disable=g-complex-comprehension
  replacements.update({
      'tensorflow_probability.python.internal import {}'.format(internal):
          'tensorflow_probability.substrates.numpy.internal import {}'.format(
              internal)
      for internal in INTERNALS
  })
  replacements.update({
      'tensorflow.python.ops import {}'.format(private):
      'tensorflow_probability.python.internal.backend.numpy import private'
      ' as {}'.format(private)
      for private in PRIVATE_TF_PKGS
  })
  replacements.update({
      'tensorflow.python.framework.ops import {}'.format(
          private):
      'tensorflow_probability.python.internal.backend.numpy import private'
      ' as {}'.format(private)
      for private in PRIVATE_TF_PKGS
  })
  # pylint: enable=g-complex-comprehension

  # TODO(bjp): Delete this block after TFP uses stateless samplers.
  replacements.update({
      'tf.random.{}'.format(sampler): 'tf.random.stateless_{}'.format(sampler)
      for sampler in SAMPLERS
  })
  replacements.update({
      'self._maybe_assert_dtype': '# self._maybe_assert_dtype',
      'SKIP_DTYPE_CHECKS = False': 'SKIP_DTYPE_CHECKS = True',
      '@test_util.test_all_tf_execution_regimes':
          '# @test_util.test_all_tf_execution_regimes',
      '@test_util.test_graph_and_eager_modes':
          '# @test_util.test_graph_and_eager_modes',
      '@test_util.test_graph_mode_only':
          '# @test_util.test_graph_mode_only',
      'TestCombinationsTest(test_util.TestCase)':
          'TestCombinationsDoNotTest(object)',
      '@six.add_metaclass(TensorMetaClass)':
          '# @six.add_metaclass(TensorMetaClass)',
  })

  filename = argv[1]
  contents = open(filename, encoding='utf-8').read()
  if '__init__.py' in filename:
    # Comment out items from __all__.
    for pkg, disabled in disabled_by_pkg.items():
      for item in disabled:
        def disable_all(name):
          replacements.update({
              '"{}"'.format(name): '# "{}"'.format(name),
              '\'{}\''.format(name): '# \'{}\''.format(name),
          })
        if 'from tensorflow_probability.{} import {}'.format(
            pkg, item) in contents:
          disable_all(item)
        for segment in contents.split(
            'from tensorflow_probability.{}.{} import '.format(
                pkg, item)):
          disable_all(segment.split('\n')[0])

  for find, replace in replacements.items():
    contents = contents.replace(find, replace)

  disabler = 'JAX_DISABLE' if FLAGS.numpy_to_jax else 'NUMPY_DISABLE'
  lines = contents.split('\n')
  for i, l in enumerate(lines):
    if disabler in l:
      lines[i] = '# {}'.format(l)
  contents = '\n'.join(lines)

  if not FLAGS.numpy_to_jax:
    contents = contents.replace('NUMPY_MODE = False', 'NUMPY_MODE = True')
  if FLAGS.numpy_to_jax:
    contents = contents.replace('tfp.substrates.numpy', 'tfp.substrates.jax')
    contents = contents.replace('substrates.numpy', 'substrates.jax')
    contents = contents.replace('backend.numpy', 'backend.jax')

    contents = contents.replace('def _call_jax', 'def __call__')
    contents = contents.replace('JAX_MODE = False', 'JAX_MODE = True')
    contents = contents.replace('SKIP_DTYPE_CHECKS = True',
                                'SKIP_DTYPE_CHECKS = False')

  substrate = 'jax' if FLAGS.numpy_to_jax else 'numpy'
  if '/python/' in filename:
    path = filename.split('/python/')[1]
  elif '/google/' in filename:
    path = 'google/' + filename.split('/google/')[1]
  footer = '\n'.join([
      '\n',
      '# ' + '@' * 78,
      '# This file is auto-generated by substrates/meta/rewrite.py',
      '# It will be surfaced by the build system as a symlink at:',
      '#   `tensorflow_probability/substrates/{}/{}`'.format(substrate, path),
      '# For more info, see substrate_runfiles_symlinks in build_defs.bzl',
      '# ' + '@' * 78,
      ])

  print(contents + footer, file=open(1, 'w', encoding='utf-8', closefd=False))


if __name__ == '__main__':
  app.run(main)
