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
"""Tests for Bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class InvertBijectorTest(test_util.TestCase):
  """Tests the correctness of the Y = Invert(bij) transformation."""

  def testBijector(self):
    for fwd in [
        tfb.Identity(),
        tfb.Exp(),
        tfb.Affine(shift=[0., 1.], scale_diag=[2., 3.]),
        tfb.Softplus(),
        tfb.SoftmaxCentered(),
    ]:
      rev = tfb.Invert(fwd)
      self.assertStartsWith(rev.name, "_".join(["invert", fwd.name]))
      x = [[[1., 2.],
            [2., 3.]]]
      self.assertAllClose(
          self.evaluate(fwd.inverse(x)), self.evaluate(rev.forward(x)))
      self.assertAllClose(
          self.evaluate(fwd.forward(x)), self.evaluate(rev.inverse(x)))
      self.assertAllClose(
          self.evaluate(fwd.forward_log_det_jacobian(x, event_ndims=1)),
          self.evaluate(rev.inverse_log_det_jacobian(x, event_ndims=1)))
      self.assertAllClose(
          self.evaluate(fwd.inverse_log_det_jacobian(x, event_ndims=1)),
          self.evaluate(rev.forward_log_det_jacobian(x, event_ndims=1)))

  def testScalarCongruency(self):
    bijector = tfb.Invert(tfb.Exp())
    bijector_test_util.assert_scalar_congruency(
        bijector, lower_x=1e-3, upper_x=1.5, eval_func=self.evaluate, rtol=0.05)

  def testShapeGetters(self):
    bijector = tfb.Invert(
        tfb.SoftmaxCentered(validate_args=True))
    x = tf.TensorShape([2])
    y = tf.TensorShape([1])
    self.assertAllEqual(y, bijector.forward_event_shape(x))
    self.assertAllEqual(
        tensorshape_util.as_list(y),
        self.evaluate(
            bijector.forward_event_shape_tensor(tensorshape_util.as_list(x))))
    self.assertAllEqual(x, bijector.inverse_event_shape(y))
    self.assertAllEqual(
        tensorshape_util.as_list(x),
        self.evaluate(
            bijector.inverse_event_shape_tensor(tensorshape_util.as_list(y))))

  def testDocstringExample(self):
    exp_gamma_distribution = (
        tfd.TransformedDistribution(
            distribution=tfd.Gamma(concentration=1., rate=2.),
            bijector=tfb.Invert(tfb.Exp())))
    self.assertAllEqual(
        [],
        self.evaluate(
            tf.shape(
                exp_gamma_distribution.sample(seed=test_util.test_seed()))))

  def testInvertCallStillWorks(self):
    x = [1., 2.]
    self.assertAllClose(np.log(x), tfb.Invert(tfb.Exp())(x),
                        atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
  tf.test.main()
