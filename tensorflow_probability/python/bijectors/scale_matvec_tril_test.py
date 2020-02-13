# Copyright 2019 The TensorFlow Probability Authors.
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
"""ScaleMatvecTriL Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class ScaleMatvecTriLTest(test_util.TestCase, parameterized.TestCase):
  """Tests correctness of the Y = scale @ x transformation."""

  @parameterized.named_parameters(
      dict(testcase_name='static', is_static=True),
      dict(testcase_name='dynamic', is_static=False),
  )
  def testNoBatch(self, is_static):
    bijector = tfb.ScaleMatvecTriL(
        scale_tril=[[2., 0.], [2., 2.]])
    x = self.maybe_static([[1., 2.]], is_static)
    self.assertAllClose([[2., 6.]], bijector.forward(x))
    self.assertAllClose([[.5, .5]], bijector.inverse(x))
    self.assertAllClose(
        -np.abs(np.log(4.)),
        bijector.inverse_log_det_jacobian(x, event_ndims=1))

  @parameterized.named_parameters(
      dict(testcase_name='static', is_static=True),
      dict(testcase_name='dynamic', is_static=False),
  )
  def testBatch(self, is_static):
    bijector = tfb.ScaleMatvecTriL(
        scale_tril=[[[2., 0.], [2., 2.]],
                    [[3., 0.], [3., 3.]]])
    x = self.maybe_static([[1., 2.]], is_static)
    self.assertAllClose([[2., 6.], [3., 9.]], bijector.forward(x))
    self.assertAllClose([[.5, .5], [1./3., 1./3.]], bijector.inverse(x))
    self.assertAllClose(
        [-np.abs(np.log(4.)), -np.abs(np.log(9.))],
        bijector.inverse_log_det_jacobian(x, event_ndims=1))

  def testRaisesWhenSingular(self):
    with self.assertRaisesRegexp(
        Exception,
        '.*Singular operator:  Diagonal contained zero values.*'):
      bijector = tfb.ScaleMatvecTriL(
          # Has zero on the diagonal.
          scale_tril=[[0., 0.], [1., 1.]],
          validate_args=True)
      self.evaluate(bijector.forward([1., 1.]))


if __name__ == '__main__':
  tf.test.main()
