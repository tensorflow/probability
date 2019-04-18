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

# Dependency imports
import numpy as np

import tensorflow as tf

from tensorflow_probability.python.internal.backend.numpy.internal import utils


__all__ = [
    'normal',
    'uniform',
    # 'all_candidate_sampler',
    # 'categorical',
    # 'experimental',
    # 'fixed_unigram_candidate_sampler',
    # 'gamma',
    # 'learned_unigram_candidate_sampler',
    # 'log_uniform_candidate_sampler',
    # 'poisson',
    # 'set_seed',
    # 'shuffle',
    # 'stateless_categorical',
    # 'stateless_normal',
    # 'stateless_truncated_normal',
    # 'stateless_uniform',
    # 'truncated_normal',
    # 'uniform_candidate_sampler',
]


def _normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None,
            name=None):  # pylint: disable=unused-argument
  rng = np.random if seed is None else np.random.RandomState(seed)
  dtype = utils.common_dtype([mean, stddev], preferred_dtype=dtype)
  return rng.normal(loc=mean, scale=stddev, size=shape).astype(dtype)


def _uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None,
             name=None):  # pylint: disable=unused-argument
  rng = np.random if seed is None else np.random.RandomState(seed)
  dtype = utils.common_dtype([minval, maxval], preferred_dtype=dtype)
  maxval = 1 if maxval is None else maxval
  return rng.uniform(low=minval, high=maxval, size=shape).astype(dtype)


# --- Begin Public Functions --------------------------------------------------


normal = utils.copy_docstring(
    tf.random.normal,
    _normal)

uniform = utils.copy_docstring(
    tf.random.uniform,
    _uniform)
