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

import contextlib

# Dependency imports
import numpy as np

import tensorflow as tf

__all__ = [
    'Benchmark',
    'TestCase',
]


# --- Begin Public Functions --------------------------------------------------


class Benchmark(tf.test.Benchmark):
  pass


class TestCase(tf.test.TestCase):
  """Wrapper of `tf.test.TestCase`."""

  def evaluate(self, x):
    return x

  def _GetNdArray(self, a):
    if isinstance(a, (np.generic, np.ndarray)):
      return a
    return np.array(a)

  @contextlib.contextmanager
  def assertRaisesOpError(self, msg):
    # Numpy backend doesn't raise OpErrors.
    try:
      yield
      self.fail('No exception raised. Expected exception similar to '
                'tf.errors.OpError with message: %s' % msg)
    except Exception:  # pylint: disable=broad-except
      pass

  def assertEqual(self, first, second, msg=None):
    if isinstance(first, list) and isinstance(second, tuple):
      first = tuple(first)
    if isinstance(first, tuple) and isinstance(second, list):
      second = tuple(second)

    return super(TestCase, self).assertEqual(first, second, msg)

main = tf.test.main
