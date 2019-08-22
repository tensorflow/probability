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
"""Tests for utilities for testing distributions and/or bijectors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl import flags
from absl.testing import flagsaver
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import seed_stream
from tensorflow_probability.python.internal import test_util as tfp_test_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

FLAGS = flags.FLAGS


JAX_MODE = False


def _maybe_jax(x):
  if JAX_MODE:
    from jax import random as jaxrand  # pylint: disable=g-import-not-at-top
    x = jaxrand.PRNGKey(x)
  return x


@test_util.run_all_in_graph_and_eager_modes
class SeedSettingTest(tf.test.TestCase):

  def testTypeCorrectness(self):
    assert isinstance(tfp_test_util.test_seed_stream(),
                      seed_stream.SeedStream)
    assert isinstance(tfp_test_util.test_seed_stream(hardcoded_seed=7),
                      seed_stream.SeedStream)
    assert isinstance(tfp_test_util.test_seed_stream(salt='foo'),
                      seed_stream.SeedStream)

  @flagsaver.flagsaver(vary_seed=False)
  def testSameness(self):
    self.assertAllEqual(tfp_test_util.test_seed(), tfp_test_util.test_seed())
    self.assertAllEqual(tfp_test_util.test_seed_stream()(),
                        tfp_test_util.test_seed_stream()())
    with flagsaver.flagsaver(fixed_seed=None):
      x = 47
      expected = _maybe_jax(x)
      self.assertAllEqual(expected, tfp_test_util.test_seed(hardcoded_seed=x))

  @flagsaver.flagsaver(vary_seed=True, fixed_seed=None)
  def testVariation(self):
    self.assertFalse(
        np.all(tfp_test_util.test_seed() == tfp_test_util.test_seed()))
    self.assertFalse(
        np.all(tfp_test_util.test_seed_stream()() ==
               tfp_test_util.test_seed_stream()()))
    x = 47
    expect_not = _maybe_jax(x)
    self.assertFalse(
        np.all(expect_not == tfp_test_util.test_seed(hardcoded_seed=x)))

  def testFixing(self):
    expected = _maybe_jax(58)
    with flagsaver.flagsaver(fixed_seed=58):
      self.assertAllEqual(expected, tfp_test_util.test_seed())
      self.assertAllEqual(expected, tfp_test_util.test_seed(hardcoded_seed=47))


if __name__ == '__main__':
  tf.test.main()
