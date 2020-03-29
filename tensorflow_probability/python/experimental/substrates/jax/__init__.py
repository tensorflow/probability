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
# will try to import jax, too.
def _ensure_jax_install():  # pylint: disable=g-statement-before-imports
  """Attempt to import jax.

  Raises:
    ImportError: if jax is not importable.
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

from tensorflow_probability.python.bijectors import _jax as bijectors
from tensorflow_probability.python.distributions import _jax as distributions
from tensorflow_probability.python.experimental import _jax as experimental
from tensorflow_probability.python.internal import _jax as internal
from tensorflow_probability.python.math import _jax as math
from tensorflow_probability.python.mcmc import _jax as mcmc
from tensorflow_probability.python.stats import _jax as stats
from tensorflow_probability.python.util import _jax as util
from tensorflow_probability.python.optimizer import _jax as optimizer

from tensorflow_probability.python.internal.backend import jax as tf2jax
