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
"""TFP for JAX."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# pylint: disable=g-statement-before-imports,g-import-not-at-top,g-bad-import-order


# Ensure JAX is importable. This needs to happen first, since the imports below
# will try to import JAX, too.
def _ensure_jax_install():  # pylint: disable=g-statement-before-imports
  """Attempt to import JAX.

  Raises:
    ImportError: if JAX is not importable.
  """
  try:
    import jax
    del jax
  except ImportError:
    print('\n\nFailed to import JAX. '
          'pip install jax jaxlib\n\n')
    raise

_ensure_jax_install()
del _ensure_jax_install  # Cleanup symbol to avoid polluting namespace.

from tensorflow_probability.python.version import __version__
# from tensorflow_probability.substrates.jax.google import staging  # DisableOnExport  # pylint:disable=line-too-long
from tensorflow_probability.substrates.jax import bijectors
from tensorflow_probability.substrates.jax import distributions
from tensorflow_probability.substrates.jax import experimental
from tensorflow_probability.substrates.jax import internal
from tensorflow_probability.substrates.jax import math
from tensorflow_probability.substrates.jax import mcmc
from tensorflow_probability.substrates.jax import optimizer
from tensorflow_probability.substrates.jax import random
from tensorflow_probability.substrates.jax import stats
from tensorflow_probability.substrates.jax import util

from tensorflow_probability.python.internal.backend import jax as tf2jax
