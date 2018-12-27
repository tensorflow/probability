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
"""Identity Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import bijector_test_util
tfe = tf.contrib.eager


@tfe.run_all_tests_in_graph_and_eager_modes
class IdentityTest(tf.test.TestCase):
  """Tests correctness of the Y = g(X) = X transformation."""

  def testBijector(self):
    bijector = tfb.Identity(validate_args=True)
    self.assertEqual("identity", bijector.name)
    x = [[[0.], [1.]]]
    self.assertAllEqual(x, self.evaluate(bijector.forward(x)))
    self.assertAllEqual(x, self.evaluate(bijector.inverse(x)))
    self.assertAllEqual(
        0., self.evaluate(bijector.inverse_log_det_jacobian(x, event_ndims=3)))
    self.assertAllEqual(
        0., self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=3)))

  def testScalarCongruency(self):
    bijector = tfb.Identity()
    bijector_test_util.assert_scalar_congruency(
        bijector, lower_x=-2., upper_x=2., eval_func=self.evaluate)


if __name__ == "__main__":
  tf.test.main()
