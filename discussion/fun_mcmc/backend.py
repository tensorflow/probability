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

import importlib
import types

import tensorflow.compat.v2 as real_tf
import tensorflow_probability as tfp_tf
# Enable the rewriting system.
from discussion.fun_mcmc import rewrite  # pylint: disable=unused-import
from discussion.fun_mcmc import tf_on_jax
from discussion.fun_mcmc import util_jax
from discussion.fun_mcmc import util_tf
from tensorflow_probability.python.internal import prefer_static as prefer_static_tf
from tensorflow_probability.substrates import jax as tfp_jax

__all__ = [
    'BACKEND_NAME',
    'get_backend',
    'JAX',
    'MANUAL_TRANSFORMS',
    'prefer_static',
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
TENSORFLOW.prefer_static = prefer_static_tf

JAX = types.ModuleType('jax_backend', 'The JAX backend.')
JAX.tf = tf_on_jax.tf
JAX.util = util_jax
JAX.tfp = tfp_jax
JAX.prefer_static = tfp_jax.internal.prefer_static

_BACKEND = (TENSORFLOW, MANUAL_TRANSFORMS)
BACKEND_NAME = 'dynamic'


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


prefer_static = _Dispatcher('prefer_static')
tf = _Dispatcher('tf')
tfp = _Dispatcher('tfp')
util = _Dispatcher('util')


def multi_backend_test(globals_dict,
                       relative_module_name,
                       backends=('jax', 'tf'),
                       test_case=None):
  """Multi-backend test decorator.

  The end goal of this decorator is that the decorated test case is removed, and
  replaced with a set of new test cases that have been rewritten to use one or
  more backends. E.g., a test case named `Test` will by default be rewritten to
  `Test_tf` and `Test_jax` which use the TensorFlow and JAX backends,
  respectively.

  The decorator works by using the dynamic rewrite system to rewrite imports of
  the module the test is defined in, and inserting the approriately renamed test
  cases into the `globals()` dictionary of the original module. A side-effect of
  this is that the global code inside the module is run `1 + len(backends)`
  times, so avoid doing anything expensive there. This does mean that the
  original module needs to be in a runnable state, i.e., when it uses symbols
  from `backend`, those must be actually present in the literal `backend`
  module.

  A subtle point about what this decorator does in the rewritten modules: the
  rewrite system replaces the backend modules from which this decorator is
  imported. The versions of this decorator in the backend-specific backend
  modules act as a passthrough to avoid infinite rewriting loops.

  Args:
    globals_dict: Python dictionary of strings to symbols. Set this to the value
      of `globals()`.
    relative_module_name: Python string. The module name of the module where the
      decorated test resides relative to `fun_mcmc`. You must not use `__name__`
      for this as that is set to the defective value of `"__main__"` when the
      tests are run via absl's test runner, which is sufficiently abnormal that
      the rewrite system does not work on it.
    backends: Python iterable of strings. Which backends to test with.
    test_case: The actual test case to decorate.

  Returns:
    None, to delete the original test case.
  """
  if test_case is None:
    return lambda test_case: multi_backend_test(  # pylint: disable=g-long-lambda
        globals_dict=globals_dict,
        relative_module_name=relative_module_name,
        test_case=test_case)

  if relative_module_name == '__main__':
    raise ValueError(
        'module_name should be written out manually, not by passing __name__.')

  # This assumes `backend` is a top-level submodule of `fun_mcmc`.
  # If we move it, we'd change the `-1` to equal the (negative) nesting level.
  root_name_comps = __name__.split('.')[:-1]
  relative_module_name_comps = relative_module_name.split('.')

  new_test_case_names = []
  for backend in backends:
    new_module_name_comps = (
        root_name_comps + ['dynamic', 'backend_{}'.format(backend)] +
        relative_module_name_comps)
    # Rewrite the module.
    new_module = importlib.import_module('.'.join(new_module_name_comps))

    # Subclass the test case so that we can rename it (absl uses the class name
    # in its UI).
    base_new_test = getattr(new_module, test_case.__name__)
    new_test = type('{}_{}'.format(test_case.__name__, backend),
                    (base_new_test,), {})
    new_test_case_names.append(new_test.__name__)
    globals_dict[new_test.__name__] = new_test

  # We deliberately return None to delete the original test case from the
  # original module.
