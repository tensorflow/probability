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
"""Tests for {f,i}ldj_ratio."""

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import exp
from tensorflow_probability.python.bijectors import ldj_ratio
from tensorflow_probability.python.bijectors import reciprocal
from tensorflow_probability.python.bijectors import scale
from tensorflow_probability.python.bijectors import scale_matvec_diag
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class LdjRatioTest(test_util.TestCase):

  def test_scale(self):
    self.assertAllClose(
        np.log(3.) - np.log(2.),
        ldj_ratio.forward_log_det_jacobian_ratio(
            scale.Scale(3.), 0., scale.Scale(-2.), 0., event_ndims=0))
    self.assertAllClose(
        np.log(2.) - np.log(3.),
        ldj_ratio.inverse_log_det_jacobian_ratio(
            scale.Scale(3.), 0., scale.Scale(-2.), 0., event_ndims=0))

  def test_scale_2d(self):
    x = tf.zeros([3, 2])
    self.assertAllClose(
        2 * (np.log(3.) - np.log(2.)),
        ldj_ratio.forward_log_det_jacobian_ratio(
            scale.Scale(3.), x, scale.Scale(-2.), x, event_ndims=1))
    self.assertAllClose(
        2 * (np.log(2.) - np.log(3.)),
        ldj_ratio.inverse_log_det_jacobian_ratio(
            scale.Scale(3.), x, scale.Scale(-2.), x, event_ndims=1))

  def test_scale_matvec_diag(self):
    z = tf.zeros([2])
    self.assertAllClose((np.log([3., 2]) - np.log([2., 7])).sum(),
                        ldj_ratio.forward_log_det_jacobian_ratio(
                            scale_matvec_diag.ScaleMatvecDiag([3., 2]),
                            z,
                            scale_matvec_diag.ScaleMatvecDiag([-2., 7]),
                            z,
                            event_ndims=1))
    self.assertAllClose((np.log([2., 7]) - np.log([3., 2])).sum(),
                        ldj_ratio.inverse_log_det_jacobian_ratio(
                            scale_matvec_diag.ScaleMatvecDiag([3., 2]),
                            z,
                            scale_matvec_diag.ScaleMatvecDiag([-2., 7]),
                            z,
                            event_ndims=1))

  def test_exp(self):
    b = exp.Exp()
    x = self.evaluate(tf.random.normal([2, 4], seed=test_util.test_seed()))
    y = self.evaluate(tf.random.normal([2, 4], seed=test_util.test_seed()))
    self.assertAllClose(
        (x - y).sum(-1),
        ldj_ratio.forward_log_det_jacobian_ratio(b, x, b, y, event_ndims=1))
    self.assertAllClose((y - x).sum(-1),
                        ldj_ratio.inverse_log_det_jacobian_ratio(
                            b, b(x), b, b(y), event_ndims=1))

  def test_recip_scale_exp(self):
    p = reciprocal.Reciprocal()(scale.Scale(3.)(exp.Exp()))
    stream = test_util.test_seed_stream()
    dim = 2
    x = self.evaluate(tf.random.uniform([4, dim], seed=stream()))
    q = reciprocal.Reciprocal()(scale.Scale(2.)(exp.Exp()))
    y = self.evaluate(tf.random.uniform([4, dim], seed=stream()))
    expected_fldjr = (
        (x - y) +  # Exp.fldj
        (np.log(3) - np.log(2)) +  # Scale.fldj
        -2 * (np.log(3. * np.exp(x) / (2 * np.exp(y))))  # Reciprocal.fldj
        ).sum(-1)
    self.assertAllClose(
        expected_fldjr,
        ldj_ratio.forward_log_det_jacobian_ratio(p, x, q, y, event_ndims=1))
    self.assertAllClose(
        expected_fldjr,
        p.forward_log_det_jacobian(x, 1) - q.forward_log_det_jacobian(y, 1))
    self.assertAllClose(
        p.inverse_log_det_jacobian(x, 1) - q.inverse_log_det_jacobian(y, 1),
        ldj_ratio.inverse_log_det_jacobian_ratio(p, x, q, y, event_ndims=1))


if __name__ == '__main__':
  test_util.main()
