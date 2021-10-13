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
"""Experimental Numpy backend."""

import contextlib
import importlib
import logging
import types

import numpy as np


from tensorflow_probability.python.internal.backend.numpy import _utils as utils


__all__ = [
    'parameterized_truncated_normal',
    'prevent_gradient',
    'remove_undocumented',
]


JAX_MODE = False


if JAX_MODE:
  from jax import random  # pylint: disable=g-import-not-at-top


def _parameterized_truncated_normal(shape, means=0., stddevs=1.,
                                    minvals=-2, maxvals=2., dtype=np.float32,
                                    seed=None, name=None):  # pylint: disable=unused-argument
  """Implementation of ops.random_ops.parameterized_truncated_normal."""

  # NOTE: The docstring for parameterized_truncated_normal is wrong.  The actual
  # requirements are:
  #  - Parameter `shape` must represent a shape with rank at least 1.
  #  - Parameters `means`, `stddevs`, `minvals`, `maxvals` can be rank 0 or
  #    rank 1.  If rank 1, then they must have shape `[shape[0]]`.

  def _right_expand(x, shape):
    if (np.ndim(x) == 0) or (np.ndim(x) >= len(shape)):
      return x
    # Add size-1 dimensions on the right to `x` so its rank is `len(shape)`.
    return np.reshape(x, np.shape(x) + (1,) * (len(shape) - 1))

  if JAX_MODE:
    min_z = (minvals - means) / stddevs
    max_z = (maxvals - means) / stddevs

    min_z = _right_expand(min_z, shape)
    max_z = _right_expand(max_z, shape)
    means = _right_expand(means, shape)
    stddevs = _right_expand(stddevs, shape)

    z = random.truncated_normal(
        seed, lower=min_z, upper=max_z, shape=shape, dtype=dtype)
    return z * stddevs + means

  raise NotImplementedError


def remove_undocumented(module_name, allowed_exception_list=None,  # pylint: disable=unused-argument
                        doc_string_modules=None):  # pylint: disable=unused-argument
  pass


def GraphOrParentsInXlaContext(*args, **kwargs):  # pylint: disable=invalid-name
  del args, kwargs
  return True


def numpy_text(tensor, is_repr=False):
  if is_repr:
    return repr(tensor)
  return str(tensor)


@contextlib.contextmanager
def eager_mode():
  yield


class LazyLoader(types.ModuleType):
  """Reimplementation of TF's LazyLoader."""

  def __init__(self, local_name, parent_module_globals, name, warning=None):
    self._local_name = local_name
    self._parent_module_globals = parent_module_globals
    self._warning = warning

    super(LazyLoader, self).__init__(name)

  def _load(self):
    """Load the module and insert it into the parent's globals."""
    # Import the target module and insert it into the parent's namespace
    module = importlib.import_module(self.__name__)
    self._parent_module_globals[self._local_name] = module

    # Emit a warning if one was specified
    if self._warning:
      logging.warning(self._warning)
      # Make sure to only warn once.
      self._warning = None

    # Update this object's dict so that if someone keeps a reference to the
    #   LazyLoader, lookups are efficient (__getattr__ is only called on lookups
    #   that fail).
    self.__dict__.update(module.__dict__)

    return module

  def __getattr__(self, item):
    module = self._load()
    return getattr(module, item)

  def __dir__(self):
    module = self._load()
    return dir(module)


def max_error(*args, **kwargs):
  del args, kwargs
  raise NotImplementedError

parameterized_truncated_normal = utils.copy_docstring(
    'random_ops.parameterized_truncated_normal',
    _parameterized_truncated_normal)


def _prevent_gradient(input, message='', name=None):  # pylint: disable=unused-argument,redefined-builtin
  raise NotImplementedError

prevent_gradient = utils.copy_docstring(
    'array_ops.prevent_gradient',
    _prevent_gradient)
