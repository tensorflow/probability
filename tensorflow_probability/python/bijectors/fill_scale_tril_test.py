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
"""Tests for FillScaleTriL bijector."""

import numpy as np
from tensorflow_probability.python.bijectors import exp
from tensorflow_probability.python.bijectors import fill_scale_tril
from tensorflow_probability.python.bijectors import softplus
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class FillScaleTriLBijectorTest(test_util.TestCase):
  """Tests the correctness of the FillScaleTriL bijector."""

  def testComputesCorrectValues(self):
    shift = 1.61803398875
    x = np.float32(np.array([-1, .5, 2]))
    y = np.float32(np.array([[np.exp(2) + shift, 0.],
                             [.5, np.exp(-1) + shift]]))

    b = fill_scale_tril.FillScaleTriL(diag_bijector=exp.Exp(), diag_shift=shift)

    y_ = self.evaluate(b.forward(x))
    self.assertAllClose(y, y_, rtol=1e-4)

    x_ = self.evaluate(b.inverse(y))
    self.assertAllClose(x, x_, rtol=1e-4)

  def testInvertible(self):

    # Generate random inputs from an unconstrained space, with
    # event size 6 to specify 3x3 triangular matrices.
    batch_shape = [2, 1]
    x = np.random.randn(*(batch_shape + [6])).astype(np.float32)
    b = fill_scale_tril.FillScaleTriL(
        diag_bijector=softplus.Softplus(), diag_shift=3.14159)
    y = self.evaluate(b.forward(x))
    self.assertAllEqual(y.shape, batch_shape + [3, 3])

    x_ = self.evaluate(b.inverse(y))
    self.assertAllClose(x, x_, rtol=1e-4)

    fldj = self.evaluate(b.forward_log_det_jacobian(x, event_ndims=1))
    ildj = self.evaluate(b.inverse_log_det_jacobian(y, event_ndims=2))
    self.assertAllClose(fldj, -ildj, rtol=1e-4)


if __name__ == '__main__':
  test_util.main()
