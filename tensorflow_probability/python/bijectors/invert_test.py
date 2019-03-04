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

import tensorflow as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd

from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class InvertBijectorTest(tf.test.TestCase):
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
      self.assertEqual("_".join(["invert", fwd.name]), rev.name)
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
        y.as_list(),
        self.evaluate(bijector.forward_event_shape_tensor(x.as_list())))
    self.assertAllEqual(x, bijector.inverse_event_shape(y))
    self.assertAllEqual(
        x.as_list(),
        self.evaluate(bijector.inverse_event_shape_tensor(y.as_list())))

  def testDocstringExample(self):
    exp_gamma_distribution = (
        tfd.TransformedDistribution(
            distribution=tfd.Gamma(concentration=1., rate=2.),
            bijector=tfb.Invert(tfb.Exp())))
    self.assertAllEqual([],
                        self.evaluate(
                            tf.shape(input=exp_gamma_distribution.sample())))


if __name__ == "__main__":
  tf.test.main()
