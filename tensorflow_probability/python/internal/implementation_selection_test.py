# Copyright 2020 The TensorFlow Probability Authors.
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
"""Tests for TFP-internal random samplers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import implementation_selection
from tensorflow_probability.python.internal import test_util


@tf.function(autograph=False)
def cumsum(x):

  def _xla_friendly(x):
    return tf.math.cumsum(x)

  def _xla_hostile(x):
    return tf.while_loop(
        cond=lambda x, _: tf.size(x) > 0,
        body=lambda x, cx: (x[:-1],  # pylint: disable=g-long-lambda
                            cx + tf.pad(x, [[tf.size(cx) - tf.size(x), 0]])),
        loop_vars=[x, tf.zeros_like(x)],
        shape_invariants=[tf.TensorShape([None]), x.shape])[1]

  return implementation_selection.implementation_selecting(
      'cumsum', default_fn=_xla_friendly, cpu_fn=_xla_hostile)(x=x)


@test_util.test_graph_and_eager_modes
class ImplSelectTest(test_util.TestCase):

  @test_util.jax_disable_test_missing_functionality('grappler impl selector')
  def testExampleCPU(self):
    arg = tf.constant([1, 2, 3])
    oracle = tf.math.cumsum(arg)
    with tf.device('CPU'):
      ans, runtime = cumsum(arg)
      # TODO(b/153684592): Drop +0, move `self.evaluate` to the line above.
      ans, runtime = self.evaluate([ans + 0, runtime + 0])
    self.assertAllEqual(oracle, ans)
    self.assertEqual(implementation_selection._RUNTIME_CPU, runtime)

  def testExampleXLA(self):
    self.skip_if_no_xla()
    arg = tf.constant([1, 2, 3])
    oracle = tf.math.cumsum(arg)
    with tf.device('CPU'):
      ans, runtime = self.evaluate(
          tf.function(cumsum, experimental_compile=True)(arg))
    self.assertAllEqual(oracle, ans)
    self.assertEqual(implementation_selection._RUNTIME_DEFAULT, runtime)

  def testExampleGPU(self):
    self.skip_if_no_xla()
    if not tf.test.is_gpu_available():
      self.skipTest('no GPU')
    arg = tf.constant([1, 2, 3])
    oracle = tf.math.cumsum(arg)
    with tf.device('GPU'):
      ans1, runtime1 = self.evaluate(
          tf.function(cumsum, experimental_compile=False)(arg))
      ans2, runtime2 = self.evaluate(
          tf.function(cumsum, experimental_compile=True)(arg))
    self.assertAllEqual(oracle, ans1)
    self.assertAllEqual(oracle, ans2)
    self.assertEqual(implementation_selection._RUNTIME_DEFAULT, runtime1)
    self.assertEqual(implementation_selection._RUNTIME_DEFAULT, runtime2)


if __name__ == '__main__':
  tf.test.main()
