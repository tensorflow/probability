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
"""Tests for TransformDiagonal bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl import logging
import hypothesis as hp
import hypothesis.strategies as hps
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.bijectors import hypothesis_testlib as bijector_hps
from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import test_util


def _preserves_vector_dim(dim):
  return lambda bijector: bijector.forward_event_shape([dim]) == [dim]


@test_util.test_all_tf_execution_regimes
class TransformDiagonalBijectorTest(test_util.TestCase):
  """Tests correctness of the TransformDiagonal bijector."""

  def testBijector(self):
    x = np.float32(np.random.randn(3, 4, 4))

    y = x.copy()
    for i in range(x.shape[0]):
      np.fill_diagonal(y[i, :, :], np.exp(np.diag(x[i, :, :])))

    exp = tfb.Exp()
    b = tfb.TransformDiagonal(diag_bijector=exp)

    y_ = self.evaluate(b.forward(x))
    self.assertAllClose(y, y_)

    x_ = self.evaluate(b.inverse(y))
    self.assertAllClose(x, x_)

    fldj = self.evaluate(b.forward_log_det_jacobian(x, event_ndims=2))
    ildj = self.evaluate(b.inverse_log_det_jacobian(y, event_ndims=2))
    self.assertAllEqual(
        fldj,
        self.evaluate(exp.forward_log_det_jacobian(
            np.array([np.diag(x_mat) for x_mat in x]),
            event_ndims=1)))
    self.assertAllEqual(
        ildj,
        self.evaluate(exp.inverse_log_det_jacobian(
            np.array([np.diag(y_mat) for y_mat in y]),
            event_ndims=1)))

  @test_util.numpy_disable_gradient_test
  def testTheoreticalFldjNormalCDF(self):
    # b/137367959 test failure trigger case (resolved by using
    # experimental_use_pfor=False as fallback instead of primary in
    # bijector_test_util.get_fldj_theoretical)
    bijector = tfb.TransformDiagonal(diag_bijector=tfb.NormalCDF())
    x = np.zeros([0, 0])
    fldj = bijector.forward_log_det_jacobian(x, event_ndims=2)
    fldj_theoretical = bijector_test_util.get_fldj_theoretical(
        bijector,
        x,
        event_ndims=2,
        inverse_event_ndims=2)
    self.assertAllClose(
        self.evaluate(fldj_theoretical),
        self.evaluate(fldj),
        atol=1e-5,
        rtol=1e-5)

  @test_util.numpy_disable_gradient_test
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings(default_max_examples=5)
  def testTheoreticalFldj(self, data):
    dim = data.draw(hps.integers(min_value=0, max_value=10))
    diag_bijector = data.draw(
        bijector_hps.unconstrained_bijectors(
            max_forward_event_ndims=1,
            must_preserve_event_ndims=True).filter(_preserves_vector_dim(dim)))
    logging.info('Using diagonal bijector %s %s', diag_bijector.name,
                 diag_bijector)

    bijector = tfb.TransformDiagonal(diag_bijector=diag_bijector)
    ensure_nonzero_batch = lambda shape: [d if d > 0 else 1 for d in shape]
    shape = data.draw(tfp_hps.shapes().map(ensure_nonzero_batch)) + [dim, dim]
    x = np.random.randn(*shape).astype(np.float64)
    y = self.evaluate(bijector.forward(x))
    bijector_test_util.assert_bijective_and_finite(
        bijector,
        x,
        y,
        eval_func=self.evaluate,
        event_ndims=2,
        inverse_event_ndims=2,
        rtol=1e-5)
    fldj = bijector.forward_log_det_jacobian(x, event_ndims=2)
    # For constant-jacobian bijectors, the zero fldj may not be broadcast.
    fldj = fldj + tf.zeros(tf.shape(x)[:-2], dtype=x.dtype)
    fldj_theoretical = bijector_test_util.get_fldj_theoretical(
        bijector,
        x,
        event_ndims=2,
        inverse_event_ndims=2)
    self.assertAllClose(
        self.evaluate(fldj_theoretical),
        self.evaluate(fldj),
        atol=1e-5,
        rtol=1e-5)


if __name__ == '__main__':
  tf.test.main()
