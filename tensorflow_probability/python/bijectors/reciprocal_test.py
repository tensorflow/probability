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
"""Tests for tensorflow_probability.python.bijectors.reciprocal."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class ReciprocalTest(test_util.TestCase):
  """Tests correctness of the `b(x) = 1 / x` bijector."""

  @parameterized.named_parameters(
      dict(
          testcase_name='positive',
          lower=1e-3,
          upper=10.
          ),
      dict(
          testcase_name='negative',
          lower=-10.,
          upper=-1e-3
          )
      )
  def testBijector(self, lower, upper):
    bijector = tfb.Reciprocal()
    self.assertStartsWith(bijector.name, 'reciprocal')
    x = tf.linspace(lower, upper, 100)
    y = 1. / x
    self.assertAllClose(self.evaluate(y), self.evaluate(bijector.forward(x)))
    self.assertAllClose(self.evaluate(x), self.evaluate(bijector.inverse(y)))

  @parameterized.named_parameters(
      dict(
          testcase_name='Positive',
          lower_x=.1,
          upper_x=10.
          ),
      dict(
          testcase_name='Negative',
          lower_x=-10.,
          upper_x=-.1
          )
      )
  def testScalarCongruency(self, lower_x, upper_x):
    bijector = tfb.Reciprocal()
    bijector_test_util.assert_scalar_congruency(
        bijector, lower_x=lower_x, upper_x=upper_x, eval_func=self.evaluate,
        rtol=0.2)

  @parameterized.named_parameters(
      dict(
          testcase_name='positive',
          lower=.1,
          upper=10.
          ),
      dict(
          testcase_name='negative',
          lower=-10.,
          upper=-.1
          )
      )
  def testBijectiveAndFinite(self, lower, upper):
    bijector = tfb.Reciprocal()
    x = np.linspace(lower, upper, num=100).astype(np.float32)
    y = np.linspace(lower, upper, num=100).astype(np.float32)
    bijector_test_util.assert_bijective_and_finite(
        bijector, x, y, eval_func=self.evaluate, event_ndims=0)


if __name__ == '__main__':
  tf.test.main()
