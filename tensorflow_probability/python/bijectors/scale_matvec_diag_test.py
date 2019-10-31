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
"""ScaleMatvecDiag Tests."""

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
class ScaleMatvecDiagTest(test_util.TestCase, parameterized.TestCase):
  """Tests correctness of the Y = scale @ x transformation."""

  @parameterized.named_parameters(
      dict(testcase_name='static', is_static=True),
      dict(testcase_name='dynamic', is_static=False),
  )
  def testNoBatch(self, is_static):
    # Corresponds to scale = [[2., 0], [0, 1.]]
    bijector = tfb.ScaleMatvecDiag(scale_diag=[2., 1])
    x = self.maybe_static([1., 1], is_static)

    # matmul(sigma, x)
    self.assertAllClose([2., 1], bijector.forward(x))
    self.assertAllClose([0.5, 1], bijector.inverse(x))
    self.assertAllClose(
        -np.log(2.),
        bijector.inverse_log_det_jacobian(x, event_ndims=1))

    # x is a 2-batch of 2-vectors.
    # The first vector is [1, 1], the second is [-1, -1].
    # Each undergoes matmul(sigma, x).
    x = self.maybe_static([[1., 1],
                           [-1., -1]], is_static)
    self.assertAllClose([[2., 1],
                         [-2., -1]],
                        bijector.forward(x))
    self.assertAllClose([[0.5, 1],
                         [-0.5, -1]],
                        bijector.inverse(x))
    self.assertAllClose(
        -np.log(2.),
        bijector.inverse_log_det_jacobian(x, event_ndims=1))

  @parameterized.named_parameters(
      dict(testcase_name='static', is_static=True),
      dict(testcase_name='dynamic', is_static=False),
  )
  def testBatch(self, is_static):
    # Corresponds to 1 2x2 matrix, with twos on the diagonal.
    scale_diag = [[2., 2]]
    bijector = tfb.ScaleMatvecDiag(scale_diag=scale_diag)
    x = self.maybe_static([[[1., 1]]], is_static)
    self.assertAllClose([[[2., 2]]], bijector.forward(x))
    self.assertAllClose([[[0.5, 0.5]]], bijector.inverse(x))
    self.assertAllClose(
        [-np.log(4)],
        bijector.inverse_log_det_jacobian(x, event_ndims=1))

  def testRaisesWhenSingular(self):
    with self.assertRaisesRegexp(
        Exception,
        'Singular operator:  Diagonal contained zero values'):
      bijector = tfb.ScaleMatvecDiag(
          # Has zero on the diagonal.
          scale_diag=[0., 1],
          validate_args=True)
      self.evaluate(bijector.forward([1., 1.]))


if __name__ == '__main__':
  tf.test.main()
