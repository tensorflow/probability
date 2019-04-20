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
"""Numpy implementations of TensorFlow functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

# Dependency imports
import numpy as np

import tensorflow as tf

from tensorflow_probability.python.internal.backend.numpy.internal import utils
from tensorflow_probability.python.internal.backend.numpy.math import softmax as _softmax


__all__ = [
    'categorical',
    'gamma',
    'normal',
    'poisson',
    'uniform',
    # 'all_candidate_sampler',
    # 'experimental',
    # 'fixed_unigram_candidate_sampler',
    # 'learned_unigram_candidate_sampler',
    # 'log_uniform_candidate_sampler',
    # 'set_seed',
    # 'shuffle',
    # 'stateless_categorical',
    # 'stateless_normal',
    # 'stateless_truncated_normal',
    # 'stateless_uniform',
    # 'truncated_normal',
    # 'uniform_candidate_sampler',
]


def _shape(args, size):
  try:
    size = tuple(size)
  except TypeError:
    size = (size,)
  if not args:
    return size
  if len(args) == 1:
    return size + np.array(args[0]).shape
  return size + functools.reduce(np.broadcast, args).shape


def _categorical(logits, num_samples, dtype=None, seed=None, name=None):  # pylint: disable=unused-argument
  rng = np.random if seed is None else np.random.RandomState(seed & 0xffffffff)
  dtype = utils.numpy_dtype(dtype or np.int64)
  if not hasattr(logits, 'shape'):
    logits = np.array(logits, np.float32)
  n = logits.shape[-1]
  return rng.choice(n, p=_softmax(logits), size=num_samples).astype(dtype)


def _gamma(shape, alpha, beta=None, dtype=tf.float32, seed=None,
           name=None):  # pylint: disable=unused-argument
  rng = np.random if seed is None else np.random.RandomState(seed & 0xffffffff)
  dtype = utils.common_dtype([alpha, beta], preferred_dtype=dtype)
  scale = 1. if beta is None else (1. / beta)
  shape = _shape([alpha, scale], shape)
  return rng.gamma(shape=alpha, scale=scale, size=shape).astype(dtype)


def _normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None,
            name=None):  # pylint: disable=unused-argument
  rng = np.random if seed is None else np.random.RandomState(seed & 0xffffffff)
  dtype = utils.common_dtype([mean, stddev], preferred_dtype=dtype)
  shape = _shape([mean, stddev], shape)
  return rng.normal(loc=mean, scale=stddev, size=shape).astype(dtype)


def _poisson(shape, lam, dtype=tf.float32, seed=None,
             name=None):  # pylint: disable=unused-argument
  rng = np.random if seed is None else np.random.RandomState(seed & 0xffffffff)
  dtype = utils.common_dtype([lam], preferred_dtype=dtype)
  shape = _shape([lam], shape)
  return rng.poisson(lam=lam, size=shape).astype(dtype)


def _uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None,
             name=None):  # pylint: disable=unused-argument
  rng = np.random if seed is None else np.random.RandomState(seed & 0xffffffff)
  dtype = utils.common_dtype([minval, maxval], preferred_dtype=dtype)
  maxval = 1 if maxval is None else maxval
  shape = _shape([minval, maxval], shape)
  return rng.uniform(low=minval, high=maxval, size=shape).astype(dtype)


# --- Begin Public Functions --------------------------------------------------


categorical = utils.copy_docstring(
    tf.random.categorical,
    _categorical)

gamma = utils.copy_docstring(
    tf.random.gamma,
    _gamma)

normal = utils.copy_docstring(
    tf.random.normal,
    _normal)

poisson = utils.copy_docstring(
    tf.random.poisson,
    _poisson)

uniform = utils.copy_docstring(
    tf.random.uniform,
    _uniform)
