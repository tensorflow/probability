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
# Lint as: python3
"""Tests for tensorflow_probability.experimental.lazybones.deferred using Numpy."""

from absl.testing import absltest

import numpy as np
import tensorflow_probability as tfp

lb = tfp.experimental.lazybones


class DeferredNumpyTest(absltest.TestCase):

  def test_numpy(self):
    a = lb.DeferredInput()
    b = a.T + 8
    c = b.sum()

    a.value = np.array([[1.], [2]])

    # Properties are also Deferred object.
    self.assertTrue((b.shape == (1, 2)).eval())

    with lb.DeferredScope():
      b.value = np.array([4., 5, 6.])
      self.assertEqual(c.eval(), 15)

    self.assertEqual(c.eval(), 19)

    # Shape change is allowed.
    a.value = np.ones((1, 3))
    self.assertEqual(b.shape.eval(), (3, 1))
    self.assertEqual(c.eval(), 27)

  def test_repr(self):
    a = lb.DeferredInput()
    b = a.T + 8
    c = b.sum()
    self.assertEqual('<DeferredInput "input">', repr(a))
    self.assertEqual('<Deferred "__add__">', repr(b))
    self.assertEqual('<Deferred "sum">', repr(c))

    a.value = np.array([[1.], [2]])
    self.assertEqual(
        '<DeferredInput "input" value=array([[1.], [2.]])>', repr(a))
    self.assertEqual('<Deferred "__add__">', repr(b))
    self.assertEqual('<Deferred "sum">', repr(c))

    # Properties are also Deferred object.
    self.assertTrue((b.shape == (1, 2)).eval())
    self.assertEqual(
        '<DeferredInput "input" value=array([[1.], [2.]])>', repr(a))
    self.assertEqual(
        '<Deferred "__add__" value=array([[ 9., 10.]])>', repr(b))
    self.assertEqual('<Deferred "sum">', repr(c))

    with lb.DeferredScope():
      b.value = np.array([4., 5, 6.])
      self.assertEqual(c.eval(), 15)
      self.assertEqual(
          '<DeferredInput "input" value=array([[1.], [2.]])>', repr(a))
      self.assertEqual(
          '<Deferred "__add__" value=array([4., 5., 6.])>', repr(b))
      self.assertEqual('<Deferred "sum" value=15.0>', repr(c))

    self.assertEqual(c.eval(), 19)
    self.assertEqual(
        '<DeferredInput "input" value=array([[1.], [2.]])>', repr(a))
    self.assertEqual(
        '<Deferred "__add__" value=array([[ 9., 10.]])>', repr(b))
    self.assertEqual('<Deferred "sum" value=19.0>', repr(c))

    # Shape change is allowed.
    a.value = np.ones((1, 3))
    self.assertEqual(
        '<DeferredInput "input" value=array([[1., 1., 1.]])>', repr(a))
    self.assertEqual('<Deferred "__add__">', repr(b))
    self.assertEqual('<Deferred "sum">', repr(c))

    self.assertEqual(b.shape.eval(), (3, 1))
    self.assertEqual(
        '<DeferredInput "input" value=array([[1., 1., 1.]])>', repr(a))
    self.assertEqual(
        '<Deferred "__add__" value=array([[9.], [9.], [9.]])>', repr(b))
    self.assertEqual('<Deferred "sum">', repr(c))

    self.assertEqual(c.eval(), 27)
    self.assertEqual(
        '<DeferredInput "input" value=array([[1., 1., 1.]])>', repr(a))
    self.assertEqual(
        '<Deferred "__add__" value=array([[9.], [9.], [9.]])>', repr(b))
    self.assertEqual('<Deferred "sum" value=27.0>', repr(c))

  def test_str(self):
    a = lb.DeferredInput()
    b = a.T + 8
    c = b.sum()

    with self.assertRaisesRegex(
        ValueError, r'Must assign value to .* object named .*\.'):
      str(a)
    with self.assertRaisesRegex(
        ValueError, r'Must assign value to .* object named .*\.'):
      str(b)
    with self.assertRaisesRegex(
        ValueError, r'Must assign value to .* object named .*\.'):
      str(c)

    a.value = np.array([[1.], [2]])
    self.assertEqual('[[1.]\n [2.]]', str(a))
    self.assertEqual('[[ 9. 10.]]', str(b))
    self.assertEqual('19.0', str(c))

    with lb.DeferredScope():
      b.value = np.array([4., 5, 6.])
      self.assertEqual('[[1.]\n [2.]]', str(a))
      self.assertEqual('[4. 5. 6.]', str(b))
      self.assertEqual('15.0', str(c))

    self.assertEqual('[[1.]\n [2.]]', str(a))
    self.assertEqual('[[ 9. 10.]]', str(b))
    self.assertEqual('19.0', str(c))

    a.value = np.ones((1, 3))
    self.assertEqual('[[1. 1. 1.]]', str(a))
    self.assertEqual('[[9.]\n [9.]\n [9.]]', str(b))
    self.assertEqual('27.0', str(c))

  def test_repr_no_recursion(self):
    a = lb.Deferred(lambda x: x, lb.DeferredInput())
    self.assertEqual('<Deferred "<lambda>">', repr(a))
    with self.assertRaisesRegex(
        ValueError, r'Must assign value to .* object named .*\.'):
      str(a)
    a.value = lb.DeferredInput()
    self.assertEqual(
        '<Deferred "<lambda>" value=<DeferredInput "input">>', repr(a))
    a.value.value = 1.23
    self.assertEqual('<DeferredInput "input">', str(a))

  def test_for_loop(self):
    x = [x_ for x_ in lb.DeferredInput([3, 2, 1], _static_iter_len=3)]
    self.assertEqual([3, 2, 1], [x[0].eval(), x[1].eval(), x[2].eval()])


if __name__ == '__main__':
  absltest.main()
