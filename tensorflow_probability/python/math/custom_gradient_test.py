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
"""Tests for Custom Gradient."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp


class CustomGradientTest(tf.test.TestCase):

  def test_works_correctly(self):
    with self.cached_session() as sess:
      f = lambda x: x**2 / 2
      g = lambda x: (x - 1)**3 / 3
      x_ = np.linspace(-100, 100, int(1e4)) + [0.]

      x = tf.constant(x_)
      fx = tfp.math.custom_gradient(f(x), g(x), x)
      gx = tf.gradients(fx, x)[0]
      [fx_, gx_] = sess.run([fx, gx])

      self.assertAllClose(f(x_), fx_)
      self.assertAllClose(g(x_), gx_)

  def test_works_correctly_both_f_g_zero(self):
    with self.cached_session() as sess:
      f = lambda x: x**2 / 2
      g = lambda x: x**3 / 3
      x_ = np.linspace(-100, 100, int(1e4)) + [0.]

      x = tf.constant(x_)
      fx = tfp.math.custom_gradient(f(x), g(x), x)
      gx = tf.gradients(fx, x)[0]
      [fx_, gx_] = sess.run([fx, gx])

      self.assertAllClose(f(x_), fx_)
      self.assertAllClose(g(x_), gx_)

  def test_works_correctly_vector_of_vars(self):
    with self.cached_session() as sess:
      x = tf.get_variable(
          name='x',
          shape=[],
          dtype=tf.float32,
          initializer=tf.constant_initializer(2))
      y = tf.get_variable(
          name='y',
          shape=[],
          dtype=tf.float32,
          initializer=tf.constant_initializer(3))
      sess.run([tf.global_variables_initializer()])

      f = lambda z: z[0] * z[1]
      g = lambda z: z[0]**2 * z[1]**2 / 2

      z = tf.stack([x, y])
      fz = tfp.math.custom_gradient(f(z), g(z), z)
      gz = tf.gradients(fz, tf.trainable_variables())
      [z_, fz_, gx_, gy_] = sess.run([z, fz, gz[0], gz[1]])

      self.assertEqual(f(z_), fz_)
      self.assertEqual(g(z_), gx_)
      self.assertEqual(g(z_), gy_)

  def test_works_correctly_side_vars(self):
    with self.cached_session() as sess:
      x_ = np.float32(2.1)  # Adding extra tenth to force imprecision.
      y_ = np.float32(3.1)
      x = tf.get_variable(
          name='x',
          shape=[],
          dtype=tf.float32,
          initializer=tf.constant_initializer(x_))
      y = tf.get_variable(
          name='y',
          shape=[],
          dtype=tf.float32,
          initializer=tf.constant_initializer(y_))
      sess.run([tf.global_variables_initializer()])

      f = lambda x: x * y
      g = lambda z: tf.square(x) * y

      fx = tfp.math.custom_gradient(f(x), g(x), x)
      gx = tf.gradients(fx, tf.trainable_variables())
      [x_, fx_, gx_] = sess.run([x, fx, gx[0]])
      gy_ = gx[1]

      self.assertEqual(x_ * y_, fx_)
      self.assertEqual(np.square(x_) * y_, gx_)
      self.assertEqual(None, gy_)

  def test_works_correctly_fx_gx_manually_stopped(self):
    with self.cached_session() as sess:
      x_ = np.float32(2.1)  # Adding extra tenth to force imprecision.
      y_ = np.float32(3.1)
      x = tf.get_variable(
          name='x',
          shape=[],
          dtype=tf.float32,
          initializer=tf.constant_initializer(x_))
      y = tf.get_variable(
          name='y',
          shape=[],
          dtype=tf.float32,
          initializer=tf.constant_initializer(y_))
      sess.run([tf.global_variables_initializer()])

      stop = tf.stop_gradient  # For readability.

      # Basically we need to stop the `x` portion of `f`. And when we supply the
      # arg to `custom_gradient` we need to stop the complement, i.e., the `y`
      # part.
      f = lambda x: stop(x) * y
      g = lambda x: stop(tf.square(x)) * y
      fx = tfp.math.custom_gradient(f(x), g(x), x + stop(y),
                                    fx_gx_manually_stopped=True)

      gx = tf.gradients(fx, tf.trainable_variables())
      [x_, fx_, gx_, gy_] = sess.run([x, fx, gx[0], gx[1]])

      self.assertEqual(x_ * y_, fx_)
      self.assertEqual(np.square(x_) * y_, gx_)
      self.assertEqual(x_, gy_)


if __name__ == '__main__':
  tf.test.main()
