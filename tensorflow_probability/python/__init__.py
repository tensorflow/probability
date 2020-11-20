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
"""Tools for probabilistic reasoning in TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import types

from tensorflow_probability.python.internal import all_util
from tensorflow_probability.python.internal import lazy_loader


# pylint: disable=g-import-not-at-top
def _validate_tf_environment(package):
  """Check TF version and (depending on package) warn about TensorFloat32.

  Args:
    package: Python `str` indicating which package is being imported. Used for
      package-dependent warning about TensorFloat32.

  Raises:
    ImportError: if either tensorflow is not importable or its version is
      inadequate.
  """
  try:
    import tensorflow.compat.v1 as tf
  except ImportError:
    # Print more informative error message, then reraise.
    print('\n\nFailed to import TensorFlow. Please note that TensorFlow is not '
          'installed by default when you install TensorFlow Probability. This '
          'is so that users can decide whether to install the GPU-enabled '
          'TensorFlow package. To use TensorFlow Probability, please install '
          'the most recent version of TensorFlow, by following instructions at '
          'https://tensorflow.org/install.\n\n')
    raise

  import distutils.version

  #
  # Update this whenever we need to depend on a newer TensorFlow release.
  #
  required_tensorflow_version = '2.4'
#   required_tensorflow_version = '1.15'  # Needed internally -- DisableOnExport

  if (distutils.version.LooseVersion(tf.__version__) <
      distutils.version.LooseVersion(required_tensorflow_version)):
    raise ImportError(
        'This version of TensorFlow Probability requires TensorFlow '
        'version >= {required}; Detected an installation of version {present}. '
        'Please upgrade TensorFlow to proceed.'.format(
            required=required_tensorflow_version,
            present=tf.__version__))

  if (package == 'mcmc' and
      tf.config.experimental.tensor_float_32_execution_enabled()):
    # Must import here, because symbols get pruned to __all__.
    import warnings
    warnings.warn(
        'TensorFloat-32 matmul/conv are enabled for NVIDIA Ampere+ GPUs. The '
        'resulting loss of precision may hinder MCMC convergence. To turn off, '
        'run `tf.config.experimental.enable_tensor_float_32_execution(False)`. '
        'For more detail, see https://github.com/tensorflow/community/pull/287.'
        )


# Declare these explicitly to appease pytype, which otherwise misses them,
# presumably due to lazy loading.
distributions: types.ModuleType
bijectors: types.ModuleType
debugging: types.ModuleType
distributions: types.ModuleType
experimental: types.ModuleType
glm: types.ModuleType
layers: types.ModuleType
math: types.ModuleType
mcmc: types.ModuleType
monte_carlo: types.ModuleType
optimizer: types.ModuleType
random: types.ModuleType
stats: types.ModuleType
sts: types.ModuleType
util: types.ModuleType
vi: types.ModuleType

_allowed_symbols = [
    'bijectors',
    'debugging',
    'distributions',
    'experimental',
    'glm',
    'layers',
    'math',
    'mcmc',
    'monte_carlo',
    'optimizer',
    'random',
    'stats',
    'sts',
    'util',
    'vi',
]

for pkg in _allowed_symbols:
  globals()[pkg] = lazy_loader.LazyLoader(
      pkg, globals(), 'tensorflow_probability.python.{}'.format(pkg),
      # These checks need to happen before lazy-loading, since the modules
      # themselves will try to import tensorflow, too.
      on_first_access=functools.partial(_validate_tf_environment, pkg))

all_util.remove_undocumented(__name__, _allowed_symbols)
