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

# Dependency imports

from absl import app
from absl import flags

flags.DEFINE_bool('rewrite_numpy_import', True,
                  'If False, we skip swapping numpy for jax.numpy.')

FLAGS = flags.FLAGS


def main(argv):
  contents = open(argv[1]).read()
  contents = contents.replace(
      'tensorflow_probability.python.internal.backend.numpy',
      'tensorflow_probability.python.internal.backend.jax')
  contents = contents.replace(
      'from tensorflow_probability.python.internal.backend import numpy',
      'from tensorflow_probability.python.internal.backend import jax')
  contents = contents.replace(
      'import tensorflow_probability.substrates.numpy as tfp',
      'import tensorflow_probability.substrates.jax as tfp')
  # To fix lazy imports in `LinearOperator`.
  contents = contents.replace(
      'tensorflow_probability.substrates.numpy',
      'tensorflow_probability.substrates.jax')
  contents = contents.replace('scipy.linalg', 'jax.scipy.linalg')
  contents = contents.replace('scipy.special', 'jax.scipy.special')
  if FLAGS.rewrite_numpy_import:
    contents = contents.replace('\nimport numpy as np',
                                '\nimport numpy as onp; import jax.numpy as np')
    contents = contents.replace('\nimport numpy as tnp',
                                '\nimport jax.numpy as tnp')
  else:
    contents = contents.replace('\nimport numpy as np',
                                '\nimport numpy as np; onp = np')
  contents = contents.replace('np.bool_', 'onp.bool_')
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
