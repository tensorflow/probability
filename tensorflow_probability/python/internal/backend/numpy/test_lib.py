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
import re

# Dependency imports
from absl import logging
import numpy as onp  # Avoid JAX rewrite.  # pylint: disable=reimported

# TODO(b/151669121): Remove dependency of test_case on TF
import tensorflow.compat.v2 as tf

__all__ = [
    'is_gpu_available',
    'Benchmark',
    'TestCase',
]


# --- Begin Public Functions --------------------------------------------------


is_gpu_available = lambda: False


class Benchmark(tf.test.Benchmark):
  pass


class TestCase(tf.test.TestCase):
  """Wrapper of `tf.test.TestCase`."""

  def evaluate(self, x):
    return tf.nest.map_structure(onp.array, x)

  def _GetNdArray(self, a):
    return onp.array(a)

  @contextlib.contextmanager
  def assertRaisesOpError(self, msg):
    # Numpy backend doesn't raise OpErrors.
    try:
      yield
      self.fail('No exception raised. Expected exception similar to '
                'tf.errors.OpError with message: %s' % msg)
    except Exception as e:  # pylint: disable=broad-except
      err_str = str(e)
      if re.search(msg, err_str):
        return
      logging.error('Expected exception to match `%s`!', msg)
      raise

  def assertEqual(self, first, second, msg=None):
    if isinstance(first, list) and isinstance(second, tuple):
      first = tuple(first)
    if isinstance(first, tuple) and isinstance(second, list):
      second = tuple(second)

    return super(TestCase, self).assertEqual(first, second, msg)

  def assertShapeEqual(self, first, second, msg=None):
    self.assertTupleEqual(first.shape, second.shape, msg=msg)


main = tf.test.main
