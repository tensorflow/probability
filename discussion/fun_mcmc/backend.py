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
from discussion.fun_mcmc import tf_on_jax
from discussion.fun_mcmc import util_jax
from discussion.fun_mcmc import util_tf

__all__ = [
    'get_backend',
    'JAX',
    'set_backend',
    'TENSORFLOW',
    'tf',
    'util',
]

TENSORFLOW = types.ModuleType('tensorflow_backend', 'The TensorFlow backend.')
TENSORFLOW.tf = real_tf
TENSORFLOW.util = util_tf

JAX = types.ModuleType('jax_backend', 'The JAX backend.')
JAX.tf = tf_on_jax.tf
JAX.util = util_jax

_BACKEND = (TENSORFLOW,)


def get_backend():
  """Returns the list of backends used by FunMCMC."""
  return _BACKEND


def set_backend(*backend):
  """Sets the backend used by FunMCMC.

  A backend is a module with two submodules:

  - `tf` providing a subset of TensorFlow API
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


class _TfDispatcher(object):
  """Dispacher for the `tf` module."""

  def __getattr__(self, attr):
    ret = _SENTINEL
    for backend in get_backend():
      ret = getattr(backend.tf, attr, ret)
    if ret is _SENTINEL:
      raise NameError('Could not resolve tf.{}'.format(attr))
    return ret


class _UtilDispatcher(object):
  """Dispacher for the `util` module."""

  def __getattr__(self, attr):
    ret = _SENTINEL
    for backend in get_backend():
      ret = getattr(backend.util, attr, ret)
    if ret is _SENTINEL:
      raise NameError('Could not resolve util.{}'.format(attr))
    return ret


tf = _TfDispatcher()
util = _UtilDispatcher()
