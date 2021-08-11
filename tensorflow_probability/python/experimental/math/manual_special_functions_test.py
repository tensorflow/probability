# Copyright 2021 The TensorFlow Probability Authors.
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
"""Tests for manual_special_functions."""

import functools

from absl import flags
from absl.testing import parameterized
import mock
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.experimental.math import manual_special_functions
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import test_util

flags.DEFINE_enum('compute_on', 'cpu', ['cpu', 'tpu'],
                  'On which device to run the computation.')
flags.DEFINE_bool('skip_builtin_on_tpu', True,
                  'Whether to skip builtin ops when running on TPU.')

FLAGS = flags.FLAGS
NUMPY_MODE = False
JAX_MODE = False


def skip_on_tpu():
  return FLAGS.compute_on == 'tpu' and FLAGS.skip_builtin_on_tpu


class ManualSpecialFunctionsTest(test_util.TestCase):

  dtype = tf.float32

  def setUp(self):
    super().setUp()

    if FLAGS.compute_on == 'tpu' and not JAX_MODE:
      resolver = tf.distribute.cluster_resolver.TPUClusterResolver('')
      tf.config.experimental_connect_to_cluster(resolver)
      tf.tpu.experimental.initialize_tpu_system(resolver)
      self.strategy = tf.distribute.experimental.TPUStrategy(resolver)
    else:
      self.strategy = None

  def run_fn(self, fn, args):
    if self.strategy:
      return tf.nest.map_structure(
          lambda x: x.values[0],
          self.evaluate(
              self.strategy.run(tf.function(fn, autograph=False), args)))
    else:
      return self.evaluate(fn(*args))

  @parameterized.named_parameters([
      dict(
          testcase_name='_exp_pade_4_4',
          x=np.concatenate(
              [np.linspace(-5., 5., 50), [-np.float('inf'),
                                          np.float('inf')]],
              axis=0),
          old_fn=tf.math.exp,
          new_fn=tfp.experimental.math.exp_pade_4_4,
      ),
      dict(
          testcase_name='_log_pade_4_4_newton',
          x=np.linspace(0., 5., 50),
          old_fn=tf.math.log,
          new_fn=tfp.experimental.math.log_pade_4_4_newton,
      ),
      dict(
          testcase_name='_expm1_pade_4_4',
          x=np.linspace(-5., 5., 50),
          old_fn=tf.math.expm1,
          new_fn=tfp.experimental.math.expm1_pade_4_4,
      ),
      dict(
          testcase_name='_log1p_pade_4_4',
          x=np.linspace(0., 2., 50),
          old_fn=tf.math.log1p,
          new_fn=tfp.experimental.math.log1p_pade_4_4,
      ),
      dict(
          testcase_name='_reduce_logsumexp',
          x=np.linspace(0., 10., 320).reshape([4, 10, 8]),
          old_fn=functools.partial(tf.math.reduce_logsumexp, axis=1),
          new_fn=functools.partial(
              tfp.experimental.math.reduce_logsumexp, axis=1),
      ),
      dict(
          testcase_name='_reduce_logsumexp_keepdims',
          x=np.linspace(0., 10., 320).reshape([4, 10, 8]),
          old_fn=functools.partial(
              tf.math.reduce_logsumexp, axis=1, keepdims=True),
          new_fn=functools.partial(
              tfp.experimental.math.reduce_logsumexp, axis=1, keepdims=True),
      ),
      dict(
          testcase_name='_softplus',
          x=np.linspace(-10., 10., 50),
          old_fn=tf.nn.softplus,
          new_fn=tfp.experimental.math.softplus,
      ),
  ])
  def test_correspondence(self,
                          x,
                          new_fn,
                          old_fn,
                          value_rtol=1e-4,
                          grad_rtol=1e-4):
    if FLAGS.compute_on == 'tpu':
      self.skipTest('The manual functions are more precise than TPU ones.')
    x = x.astype(dtype_util.as_numpy_dtype(self.dtype))

    y1 = self.run_fn(old_fn, (x,))
    y2 = self.run_fn(new_fn, (x,))

    self.assertAllClose(y1, y2, rtol=value_rtol)

    if NUMPY_MODE:
      return

    num_outputs = np.prod(y1.shape).astype(np.int32)

    dy = tf.reshape(tf.range(1, 1 + num_outputs, dtype=y1.dtype), y1.shape)

    y1, grads1 = self.run_fn(
        lambda x, dy: tfp.math.value_and_gradient(  # pylint: disable=g-long-lambda
            old_fn, x, output_gradients=dy), (x, dy))
    y2, grads2 = self.run_fn(
        lambda x, dy: tfp.math.value_and_gradient(  # pylint: disable=g-long-lambda
            new_fn, x, output_gradients=dy), (x, dy))

    self.assertAllClose(y1, y2, rtol=value_rtol)
    self.assertAllClose(grads1, grads2, rtol=grad_rtol)

  @parameterized.named_parameters([
      dict(
          testcase_name='_exp',
          x=10**np.linspace(-5., 0.5, 50),
          fn=tf.math.exp,
          true_fwd_fn=np.exp,
          true_bwd_fn=lambda x, dy: dy * np.exp(x),
          value_rtol=1e-6,
          grad_rtol=1e-6,
          skip_fn=skip_on_tpu,
      ),
      dict(
          testcase_name='_log_far',
          x=10**np.concatenate(
              [-np.linspace(3., 21., 50),
               np.linspace(3., 21., 50)]),
          fn=tf.math.log,
          true_fwd_fn=np.log,
          true_bwd_fn=lambda x, dy: dy / x,
          value_rtol=1e-6,
          grad_rtol=1e-6,
          skip_fn=skip_on_tpu,
      ),
      dict(
          testcase_name='_log_near',
          x=10**np.linspace(-3., 3., 50),
          fn=tf.math.log,
          true_fwd_fn=np.log,
          true_bwd_fn=lambda x, dy: dy / x,
          value_rtol=1e-5,
          grad_rtol=1e-5,
          skip_fn=skip_on_tpu,
      ),
      dict(
          testcase_name='_expm1',
          x=10**np.linspace(-5., 0.5, 50),
          fn=tf.math.expm1,
          true_fwd_fn=np.expm1,
          true_bwd_fn=lambda x, dy: dy * np.exp(x),
          value_rtol=1e-6,
          grad_rtol=1e-6,
          skip_fn=skip_on_tpu,
      ),
      dict(
          testcase_name='_log1p',
          x=10.**np.linspace(-4., 4., 50),
          fn=tf.math.log1p,
          true_fwd_fn=np.log1p,
          true_bwd_fn=lambda x, dy: dy / (1. + x),
          value_rtol=1e-6,
          grad_rtol=1e-6,
          skip_fn=skip_on_tpu,
      ),
      dict(
          testcase_name='_exp_pade_4_4',
          x=10**np.linspace(-5., 0.5, 50),
          fn=tfp.experimental.math.exp_pade_4_4,
          true_fwd_fn=np.exp,
          true_bwd_fn=lambda x, dy: dy * np.exp(x),
          value_rtol=1e-6,
          grad_rtol=1e-6,
      ),
      dict(
          testcase_name='_log_pade_4_4_newton_far',
          x=10**np.concatenate(
              [-np.linspace(3., 21., 50),
               np.linspace(3., 21., 50)]),
          fn=tfp.experimental.math.log_pade_4_4_newton,
          true_fwd_fn=np.log,
          true_bwd_fn=lambda x, dy: dy / x,
          value_rtol=1e-6,
          grad_rtol=1e-6,
      ),
      dict(
          testcase_name='_log_pade_4_4_newton_near',
          x=10**np.linspace(-3., 3., 50),
          fn=tfp.experimental.math.log_pade_4_4_newton,
          true_fwd_fn=np.log,
          true_bwd_fn=lambda x, dy: dy / x,
          value_rtol=1e-5,
          grad_rtol=1e-5,
      ),
      dict(
          testcase_name='_expm1_pade_4_4',
          x=10**np.linspace(-5., 0.5, 50),
          fn=tfp.experimental.math.expm1_pade_4_4,
          true_fwd_fn=np.expm1,
          true_bwd_fn=lambda x, dy: dy * np.exp(x),
          value_rtol=1e-6,
          grad_rtol=1e-6,
      ),
      dict(
          testcase_name='_log1p_pade_4_4',
          x=10.**np.linspace(-4., 4., 50),
          fn=tfp.experimental.math.log1p_pade_4_4,
          true_fwd_fn=np.log1p,
          true_bwd_fn=lambda x, dy: dy / (1. + x),
          value_rtol=1e-6,
          grad_rtol=1e-6,
      ),
  ])
  def test_accuracy(self,
                    x,
                    fn,
                    true_fwd_fn,
                    true_bwd_fn,
                    value_rtol=1e-4,
                    grad_rtol=1e-4,
                    skip_fn=lambda: False):
    if skip_fn():
      self.skipTest('This test is expected to fail in this configuration.')
    dtype = dtype_util.as_numpy_dtype(self.dtype)

    x_gt = np.array(x)
    x = x_gt.astype(dtype)

    y = self.run_fn(fn, (x,))
    y_gt = true_fwd_fn(x_gt).astype(dtype)

    self.assertAllClose(y_gt, y, rtol=value_rtol)

    if NUMPY_MODE:
      return

    num_outputs = np.prod(y.shape).astype(np.int32)

    dy_gt = np.arange(1, 1 + num_outputs).astype(np.float64).reshape(y.shape)
    dy = dy_gt.astype(dtype)

    y, grads = self.run_fn(
        lambda x, dy: tfp.math.value_and_gradient(fn, x, output_gradients=dy),
        (x, dy))
    grads_gt = true_bwd_fn(x_gt, dy_gt).astype(dtype)

    self.assertAllClose(y_gt, y, rtol=value_rtol)
    self.assertAllClose(grads_gt, grads, rtol=grad_rtol)

  @mock.patch.object(manual_special_functions, 'softplus')
  @mock.patch.object(manual_special_functions, 'reduce_logsumexp')
  @mock.patch.object(manual_special_functions, 'log1p_pade_4_4')
  @mock.patch.object(manual_special_functions, 'expm1_pade_4_4')
  @mock.patch.object(manual_special_functions, 'log_pade_4_4_newton')
  @mock.patch.object(manual_special_functions, 'exp_pade_4_4')
  def test_patching(self, exp, log, expm1, log1p, logsumexp, softplus):
    exp_calls = 0
    expm1_calls = 0
    log_calls = 0
    log1p_calls = 0
    logsumexp_calls = 0
    softplus_calls = 0
    with tfp.experimental.math.patch_manual_special_functions():
      tf.exp(0.)
      exp_calls += 1
      self.assertEqual(exp_calls, exp.call_count)

      tf.math.exp(0.)
      exp_calls += 1
      self.assertEqual(exp_calls, exp.call_count)

      tf.math.log(0.)
      log_calls += 1
      self.assertEqual(log_calls, log.call_count)

      tf.math.expm1(0.)
      expm1_calls += 1
      self.assertEqual(expm1_calls, expm1.call_count)

      tf.math.log1p(0.)
      log1p_calls += 1
      self.assertEqual(log1p_calls, log1p.call_count)

      tf.math.reduce_logsumexp(0.)
      logsumexp_calls += 1
      self.assertEqual(logsumexp_calls, logsumexp.call_count)

      tf.reduce_logsumexp(0.)
      logsumexp_calls += 1
      self.assertEqual(logsumexp_calls, logsumexp.call_count)

      tf.math.softplus(0.)
      softplus_calls += 1
      self.assertEqual(softplus_calls, softplus.call_count)

      tf.nn.softplus(0.)
      softplus_calls += 1
      self.assertEqual(softplus_calls, softplus.call_count)

  @mock.patch.object(manual_special_functions, 'softplus')
  @mock.patch.object(manual_special_functions, 'reduce_logsumexp')
  @mock.patch.object(manual_special_functions, 'log1p_pade_4_4')
  @mock.patch.object(manual_special_functions, 'expm1_pade_4_4')
  @mock.patch.object(manual_special_functions, 'log_pade_4_4_newton')
  @mock.patch.object(manual_special_functions, 'exp_pade_4_4')
  def test_patching_jax(self, exp, log, expm1, log1p, logsumexp, softplus):
    if not JAX_MODE or NUMPY_MODE:
      self.skipTest('This test is JAX-only')

    import jax  # pylint: disable=g-import-not-at-top

    exp_calls = 0
    expm1_calls = 0
    log_calls = 0
    log1p_calls = 0
    logsumexp_calls = 0
    softplus_calls = 0
    with tfp.experimental.math.patch_manual_special_functions():
      jax.numpy.exp(0.)
      exp_calls += 1
      self.assertEqual(exp_calls, exp.call_count)

      jax.numpy.log(0.)
      log_calls += 1
      self.assertEqual(log_calls, log.call_count)

      jax.numpy.expm1(0.)
      expm1_calls += 1
      self.assertEqual(expm1_calls, expm1.call_count)

      jax.numpy.log1p(0.)
      log1p_calls += 1
      self.assertEqual(log1p_calls, log1p.call_count)

      jax.scipy.special.logsumexp(0.)
      logsumexp_calls += 1
      self.assertEqual(logsumexp_calls, logsumexp.call_count)

      jax.nn.softplus(0.)
      softplus_calls += 1
      self.assertEqual(softplus_calls, softplus.call_count)


if __name__ == '__main__':
  tf.test.main()
