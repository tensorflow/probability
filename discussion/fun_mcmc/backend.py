# Copyright 2020 The TensorFlow Probability Authors.
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
"""FunMCMC backend.

This module is used to provide simultaneous compatibility between TensorFlow and
JAX.
"""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function

import types

import tensorflow.compat.v2 as real_tf
import tensorflow_probability as tfp_tf
from discussion.fun_mcmc import tf_on_jax
from discussion.fun_mcmc import util_jax
from discussion.fun_mcmc import util_tf
from tensorflow_probability.python.experimental.substrates import jax as tfp_jax

__all__ = [
    'get_backend',
    'JAX',
    'MANUAL_TRANSFORMS',
    'set_backend',
    'TENSORFLOW',
    'tf',
    'tfp',
    'util',
]


def _manual_value_and_ldj(fn, args):
  value, (extra, ldj) = fn(args)
  return value, (extra, ldj), ldj


def _manual_inverse_fn(fn):
  return fn.inverse


MANUAL_TRANSFORMS = types.ModuleType(
    'manual_transforms_backend', """Manual transforms backend.

In this backend you specify the transport map inverse and log determinant of the
jacobian manually. Each transform must have a field called `inverse` which
contains the inverse transformation and the `extra` return is a 2-tuple, where
the first element is arbitrary and the the last element is the log determinant
of the jacobian of the transformation.

Here's an example of a complete specification of an invertible function that
works with this backend:

```python
def scale_by_two(x):
  # Return x unchanged for illustrative purposes.
  return 2 * x, (x, np.log(2))

def scale_by_half(x):
  return x / 2, (x, -np.log(2))

scale_by_two.inverse = scale_by_half
scale_by_half.inverse = scale_by_two

y, y_extra, y_ldj = value_and_ldj(scale_by_2, 3.)
assert y == 6
assert y_extra == 3
assert y_ldj == np.log(2)

inv_scale_by_2 = inverse_fn(scale_by_2)
assert inv_scale_by_2 == scale_by_half

x, x_extra, x_ldj = value_and_ldj(inv_scale_by_2, 4.)
assert x == 2
assert x_extra == 4
assert x_ldj == -np.log(2)
```

""")
MANUAL_TRANSFORMS.util = types.ModuleType('utils', 'Manual transforms utils.')
MANUAL_TRANSFORMS.util.value_and_ldj = _manual_value_and_ldj
MANUAL_TRANSFORMS.util.inverse_fn = _manual_inverse_fn

TENSORFLOW = types.ModuleType('tensorflow_backend', 'The TensorFlow backend.')
TENSORFLOW.tf = real_tf
TENSORFLOW.util = util_tf
TENSORFLOW.tfp = tfp_tf

JAX = types.ModuleType('jax_backend', 'The JAX backend.')
JAX.tf = tf_on_jax.tf
JAX.util = util_jax
JAX.tfp = tfp_jax

_BACKEND = (TENSORFLOW, MANUAL_TRANSFORMS)


def get_backend():
  """Returns the list of backends used by FunMCMC."""
  return _BACKEND


def set_backend(*backend):
  """Sets the backend used by FunMCMC.

  A backend is a module with three submodules:

  - `tf` providing a subset of TensorFlow API
  - `tfp` providing a subset of TensorFlow Probability API
  - `util` providing certain utilities

  The above are deliberately vague, and are unstable in terms of API. You can,
  however, experiment with new backends by looking at the existing
  implementations for inspiration.

  If multiple backends are provided, then later backends override the earlier
  ones.

  Args:
    *backend: One or more backends.

  Raises:
    ValueError: If no backends are provided.
  """
  global _BACKEND
  if not backend:
    raise ValueError('Need at least one backend.')
  _BACKEND = backend


class _Sentinel(object):
  pass


_SENTINEL = _Sentinel()


class _Dispatcher(object):
  """Dispacher for a top-level backend module."""

  def __init__(self, module):
    self._module = module

  def __getattr__(self, attr):
    ret = _SENTINEL
    for backend in get_backend():
      mod = getattr(backend, self._module, None)
      if mod is None:
        continue
      ret = getattr(mod, attr, ret)
    if ret is _SENTINEL:
      raise NameError('Could not resolve {}.{}'.format(self._module, attr))
    return ret


tf = _Dispatcher('tf')
tfp = _Dispatcher('tfp')
util = _Dispatcher('util')
