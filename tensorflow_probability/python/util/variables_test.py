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
"""Tests for utility functions related to managing `tf.Variable`s."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

# Dependency imports
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp


def test_fn(x):
  x = tf.convert_to_tensor(value=x, name='x')
  dtype = x.dtype.as_numpy_dtype
  s = x.shape.as_list()
  z = tf.compat.v1.get_variable(
      name='z',
      dtype=dtype,
      initializer=np.arange(np.prod(s)).reshape(s).astype(dtype))
  y = tf.compat.v1.get_variable(
      name='y',
      dtype=dtype,
      initializer=np.arange(np.prod(s)).reshape(s).astype(dtype)**2)
  return x + y + z


class _WrapCallableTest(object):

  def testDefaultArgsWorkCorrectly(self):
    with self.cached_session():
      x = tf.constant(self.dtype([0.1, 0.2]))
      wrapped_fn, vars_args = tfp.util.externalize_variables_as_args(
          test_fn, [x])

      tf.compat.v1.get_variable_scope().reuse_variables()

      result = wrapped_fn(self.dtype(2), [3, 4, 5], 0.5)

      y_actual = tf.compat.v1.get_variable('y', dtype=self.dtype)
      z_actual = tf.compat.v1.get_variable('z', dtype=self.dtype)

      tf.compat.v1.global_variables_initializer().run()
      result_ = self.evaluate(result)

      self.assertEqual(self.dtype, result_.dtype)
      self.assertAllEqual([5.5, 6.5, 7.5], result_)
      self.assertAllEqual([y_actual, z_actual], vars_args)

  def testNonDefaultArgsWorkCorrectly(self):
    with self.cached_session():
      x = tf.constant(self.dtype([0.1, 0.2]))

      _ = test_fn(self.dtype([0., 0.]))   # Needed to create vars.
      tf.compat.v1.get_variable_scope().reuse_variables()

      y_actual = tf.compat.v1.get_variable('y', dtype=self.dtype)

      wrapped_fn, vars_args = tfp.util.externalize_variables_as_args(
          test_fn, [x], possible_ancestor_vars=[y_actual])

      result = wrapped_fn(self.dtype([2, 3]), 0.5)  # x, y

      tf.compat.v1.global_variables_initializer().run()
      result_ = self.evaluate(result)

      self.assertEqual(self.dtype, result_.dtype)
      self.assertAllEqual([2.5, 4.5], result_)
      self.assertAllEqual([y_actual], vars_args)

  def testWarnings(self):
    with self.cached_session():
      x = tf.constant(self.dtype([0.1, 0.2]))
      wrapped_fn, _ = tfp.util.externalize_variables_as_args(
          test_fn, [x], possible_ancestor_vars=[])
      tf.compat.v1.get_variable_scope().reuse_variables()
      with warnings.catch_warnings(record=True) as w:
        wrapped_fn(self.dtype(2))
      w = sorted(w, key=lambda w: str(w.message))
      self.assertEqual(2, len(w))
      self.assertRegexpMatches(
          str(w[0].message),
          r"Variable .* 'y:0' .* not found in bypass dict.")
      self.assertRegexpMatches(
          str(w[1].message),
          r"Variable .* 'z:0' .* not found in bypass dict.")

  def testExceptions(self):
    with self.cached_session():
      x = tf.constant(self.dtype([0.1, 0.2]))
      wrapped_fn, _ = tfp.util.externalize_variables_as_args(
          test_fn,
          [x],
          possible_ancestor_vars=[],
          assert_variable_override=True)
      tf.compat.v1.get_variable_scope().reuse_variables()
      with self.assertRaisesRegexp(ValueError, 'not found'):
        wrapped_fn(self.dtype(2))


class WrapCallableTest16(tf.test.TestCase, _WrapCallableTest):
  dtype = np.float16


class WrapCallableTest32(tf.test.TestCase, _WrapCallableTest):
  dtype = np.float32


class WrapCallableTest64(tf.test.TestCase, _WrapCallableTest):
  dtype = np.float64


if __name__ == '__main__':
  tf.test.main()
