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

__all__ = [
    'get_backend',
    'JAX',
    'set_backend',
    'TENSORFLOW',
    'tf',
    'util',
]

TENSORFLOW = 'tensorflow'
JAX = 'jax'
_BACKEND = TENSORFLOW


def get_backend():
  """Returns the current backend used by FunMCMC."""
  return _BACKEND


def set_backend(backend):
  """Sets the backend used by FunMCMC."""
  global _BACKEND
  _BACKEND = backend


class _TfDispatcher(object):
  """Dispacher for the `tf` module."""

  def __getattr__(self, attr):
    # pylint: disable=g-import-not-at-top
    if get_backend() == TENSORFLOW:
      import tensorflow.compat.v2 as tf_
    elif get_backend() == JAX:
      from discussion.fun_mcmc import tf_on_jax
      tf_ = tf_on_jax.tf
    else:
      raise ValueError('Unknown backend "{}"'.format(get_backend()))
    return getattr(tf_, attr)


class _UtilDispatcher(object):
  """Dispacher for the `util` module."""

  def __getattr__(self, attr):
    # pylint: disable=g-import-not-at-top
    if get_backend() == TENSORFLOW:
      from discussion.fun_mcmc import util_tf as util_
    elif get_backend() == JAX:
      from discussion.fun_mcmc import util_jax as util_
    else:
      raise ValueError('Unknown backend "{}"'.format(get_backend()))
    return getattr(util_, attr)


tf = _TfDispatcher()
util = _UtilDispatcher()
