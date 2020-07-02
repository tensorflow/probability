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
"""Rewrite script for NP->JAX."""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function

# Dependency imports

from absl import app


def main(argv):
  contents = open(argv[1]).read()
  contents = contents.replace('._numpy', '._jax')
  contents = contents.replace(
      'tensorflow_probability.python.internal.backend.numpy',
      'tensorflow_probability.python.internal.backend.jax')
  contents = contents.replace(
      'from tensorflow_probability.python.internal.backend import numpy',
      'from tensorflow_probability.python.internal.backend import jax')
  contents = contents.replace(
      ('import tensorflow_probability.python.experimental.substrates.numpy' +
       ' as tfp'),
      'import tensorflow_probability.python.experimental.substrates.jax as tfp')
  contents = contents.replace('scipy.linalg', 'jax.scipy.linalg')
  contents = contents.replace('scipy.special', 'jax.scipy.special')
  contents = contents.replace(
      'tf.test.main()',
      'from jax.config import config; config.update("jax_enable_x64", True); '
      'tf.test.main()')
  contents = contents.replace('\nimport numpy as np',
                              '\nimport numpy as onp\nimport jax.numpy as np')
  contents = contents.replace('np.bool', 'onp.bool')
  contents = contents.replace('np.dtype', 'onp.dtype')
  contents = contents.replace('np.generic', 'onp.generic')

  contents = contents.replace('np.broadcast', 'onp.broadcast')
  # so as to fixup np.broadcast_arrays or np.broadcast_to
  contents = contents.replace('onp.broadcast_arrays', 'np.broadcast_arrays')
  contents = contents.replace('onp.broadcast_to', 'np.broadcast_to')

  contents = contents.replace('np.ndindex', 'onp.ndindex')

  contents = contents.replace('JAX_MODE = False', 'JAX_MODE = True')
  contents = contents.replace('NumpyTest', 'JaxTest')

  print(contents)


if __name__ == '__main__':
  app.run(main)
