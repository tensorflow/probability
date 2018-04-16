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
"""Tests for linear algebra."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow as tf

from tensorflow_probability.python.math import pinv as pinv
from tensorflow.python.framework import test_util


class _PinvTest(object):

  def expected_pinv(self, a, rcond):
    """Calls `np.linalg.pinv` but corrects its broken batch semantics."""
    if a.ndim < 3:
      return np.linalg.pinv(a, rcond)
    if rcond is None:
      rcond = 10. * max(a.shape[-2], a.shape[-1]) * np.finfo(a.dtype).eps
    s = np.concatenate([a.shape[:-2], [a.shape[-1], a.shape[-2]]])
    a_pinv = np.zeros(s, dtype=a.dtype)
    for i in np.ndindex(a.shape[:(a.ndim - 2)]):
      a_pinv[i] = np.linalg.pinv(
          a[i],
          rcond=rcond if isinstance(rcond, float) else rcond[i])
    return a_pinv

  @test_util.run_in_graph_and_eager_modes()
  def test_symmetric(self):
    a_ = self.dtype([[1., .4, .5],
                     [.4, .2, .25],
                     [.5, .25, .35]])
    a_ = np.stack([a_ + 1., a_], axis=0)  # Batch of matrices.
    a = tf.placeholder_with_default(
        input=a_,
        shape=a_.shape if self.use_static_shape else None)
    if self.use_default_rcond:
      rcond = None
    else:
      rcond = self.dtype([0., 0.01])  # Smallest 1 component is forced to zero.
    expected_a_pinv_ = self.expected_pinv(a_, rcond)
    a_pinv = pinv(a, rcond, validate_args=True)
    a_pinv_ = self.evaluate(a_pinv)
    self.assertAllClose(expected_a_pinv_, a_pinv_,
                        atol=1e-5, rtol=1e-5)
    if not self.use_static_shape:
      return
    self.assertAllEqual(expected_a_pinv_.shape, a_pinv.shape)

  @test_util.run_in_graph_and_eager_modes()
  def test_nonsquare(self):
    a_ = self.dtype([[1., .4, .5, 1.],
                     [.4, .2, .25, 2.],
                     [.5, .25, .35, 3.]])
    a_ = np.stack([a_ + 0.5, a_], axis=0)  # Batch of matrices.
    a = tf.placeholder_with_default(
        input=a_,
        shape=a_.shape if self.use_static_shape else None)
    if self.use_default_rcond:
      rcond = None
    else:
      # Smallest 2 components are forced to zero.
      rcond = self.dtype([0., 0.25])
    expected_a_pinv_ = self.expected_pinv(a_, rcond)
    a_pinv = pinv(a, rcond, validate_args=True)
    a_pinv_ = self.evaluate(a_pinv)
    self.assertAllClose(expected_a_pinv_, a_pinv_,
                        atol=1e-5, rtol=1e-4)
    if not self.use_static_shape:
      return
    self.assertAllEqual(expected_a_pinv_.shape, a_pinv.shape)


class PinvTestDynamic32DefaultRcond(tf.test.TestCase, _PinvTest):
  dtype = np.float32
  use_static_shape = False
  use_default_rcond = True


class PinvTestStatic64DefaultRcond(tf.test.TestCase, _PinvTest):
  dtype = np.float64
  use_static_shape = True
  use_default_rcond = True


class PinvTestDynamic32CustomtRcond(tf.test.TestCase, _PinvTest):
  dtype = np.float32
  use_static_shape = False
  use_default_rcond = False


class PinvTestStatic64CustomRcond(tf.test.TestCase, _PinvTest):
  dtype = np.float64
  use_static_shape = True
  use_default_rcond = False


if __name__ == '__main__':
  tf.test.main()
