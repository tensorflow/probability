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
"""Chain Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf
from tensorflow_probability.python import bijectors as tfb

from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


class ShapeChanging(tfb.Bijector):
  """Only used for op_ndims manipulation."""

  def __init__(self, forward_min_event_ndims=0, inverse_min_event_ndims=3):
    super(ShapeChanging, self).__init__(
        forward_min_event_ndims=forward_min_event_ndims,
        inverse_min_event_ndims=inverse_min_event_ndims,
        validate_args=False, name="shape_changer")


@test_util.run_all_in_graph_and_eager_modes
class ChainBijectorTest(tf.test.TestCase):
  """Tests the correctness of the Y = Chain(bij1, bij2, bij3) transformation."""

  def testBijector(self):
    chain = tfb.Chain((tfb.Exp(), tfb.Softplus()))
    self.assertEqual("chain_of_exp_of_softplus", chain.name)
    x = np.asarray([[[1., 2.],
                     [2., 3.]]])
    self.assertAllClose(1. + np.exp(x), self.evaluate(chain.forward(x)))
    self.assertAllClose(np.log(x - 1.), self.evaluate(chain.inverse(x)))
    self.assertAllClose(
        -np.sum(np.log(x - 1.), axis=2),
        self.evaluate(chain.inverse_log_det_jacobian(x, event_ndims=1)))
    self.assertAllClose(
        np.sum(x, axis=2),
        self.evaluate(chain.forward_log_det_jacobian(x, event_ndims=1)))

  def testBijectorIdentity(self):
    chain = tfb.Chain()
    self.assertEqual("identity", chain.name)
    x = np.asarray([[[1., 2.],
                     [2., 3.]]])
    self.assertAllClose(x, self.evaluate(chain.forward(x)))
    self.assertAllClose(x, self.evaluate(chain.inverse(x)))
    self.assertAllClose(
        0., self.evaluate(chain.inverse_log_det_jacobian(x, event_ndims=1)))
    self.assertAllClose(
        0., self.evaluate(chain.forward_log_det_jacobian(x, event_ndims=1)))

  def testScalarCongruency(self):
    chain = tfb.Chain((tfb.Exp(), tfb.Softplus()))
    bijector_test_util.assert_scalar_congruency(
        chain, lower_x=1e-3, upper_x=1.5, rtol=0.05, eval_func=self.evaluate)

  def testShapeGetters(self):
    chain = tfb.Chain([
        tfb.SoftmaxCentered(validate_args=True),
        tfb.SoftmaxCentered(validate_args=True),
    ])
    x = tf.TensorShape([1])
    y = tf.TensorShape([2 + 1])
    self.assertAllEqual(y, chain.forward_event_shape(x))
    self.assertAllEqual(
        y.as_list(),
        self.evaluate(chain.forward_event_shape_tensor(x.as_list())))
    self.assertAllEqual(x, chain.inverse_event_shape(y))
    self.assertAllEqual(
        x.as_list(),
        self.evaluate(chain.inverse_event_shape_tensor(y.as_list())))

  def testMinEventNdimsChain(self):
    chain = tfb.Chain([tfb.Exp(), tfb.Exp(), tfb.Exp()])
    self.assertEqual(0, chain.forward_min_event_ndims)
    self.assertEqual(0, chain.inverse_min_event_ndims)

    chain = tfb.Chain([tfb.Affine(), tfb.Affine(), tfb.Affine()])
    self.assertEqual(1, chain.forward_min_event_ndims)
    self.assertEqual(1, chain.inverse_min_event_ndims)

    chain = tfb.Chain([tfb.Exp(), tfb.Affine()])
    self.assertEqual(1, chain.forward_min_event_ndims)
    self.assertEqual(1, chain.inverse_min_event_ndims)

    chain = tfb.Chain([tfb.Affine(), tfb.Exp()])
    self.assertEqual(1, chain.forward_min_event_ndims)
    self.assertEqual(1, chain.inverse_min_event_ndims)

    chain = tfb.Chain([tfb.Affine(), tfb.Exp(), tfb.Softplus(), tfb.Affine()])
    self.assertEqual(1, chain.forward_min_event_ndims)
    self.assertEqual(1, chain.inverse_min_event_ndims)

  def testMinEventNdimsShapeChangingAddDims(self):
    chain = tfb.Chain([ShapeChanging()])
    self.assertEqual(0, chain.forward_min_event_ndims)
    self.assertEqual(3, chain.inverse_min_event_ndims)

    chain = tfb.Chain([ShapeChanging(), tfb.Affine()])
    self.assertEqual(1, chain.forward_min_event_ndims)
    self.assertEqual(4, chain.inverse_min_event_ndims)

    chain = tfb.Chain([tfb.Affine(), ShapeChanging()])
    self.assertEqual(0, chain.forward_min_event_ndims)
    self.assertEqual(3, chain.inverse_min_event_ndims)

    chain = tfb.Chain([ShapeChanging(), ShapeChanging()])
    self.assertEqual(0, chain.forward_min_event_ndims)
    self.assertEqual(6, chain.inverse_min_event_ndims)

  def testMinEventNdimsShapeChangingRemoveDims(self):
    chain = tfb.Chain([ShapeChanging(3, 0)])
    self.assertEqual(3, chain.forward_min_event_ndims)
    self.assertEqual(0, chain.inverse_min_event_ndims)

    chain = tfb.Chain([ShapeChanging(3, 0), tfb.Affine()])
    self.assertEqual(3, chain.forward_min_event_ndims)
    self.assertEqual(0, chain.inverse_min_event_ndims)

    chain = tfb.Chain([tfb.Affine(), ShapeChanging(3, 0)])
    self.assertEqual(4, chain.forward_min_event_ndims)
    self.assertEqual(1, chain.inverse_min_event_ndims)

    chain = tfb.Chain([ShapeChanging(3, 0), ShapeChanging(3, 0)])
    self.assertEqual(6, chain.forward_min_event_ndims)
    self.assertEqual(0, chain.inverse_min_event_ndims)

  def testMinEventNdimsShapeChangingAddRemoveDims(self):
    chain = tfb.Chain(
        [ShapeChanging(2, 1),
         ShapeChanging(3, 0),
         ShapeChanging(1, 2)])
    self.assertEqual(4, chain.forward_min_event_ndims)
    self.assertEqual(1, chain.inverse_min_event_ndims)

  def testChainExpAffine(self):
    scale_diag = np.array([1., 2., 3.], dtype=np.float32)
    chain = tfb.Chain([tfb.Exp(), tfb.Affine(scale_diag=scale_diag)])
    x = [0., np.log(2., dtype=np.float32), np.log(3., dtype=np.float32)]
    y = [1., 4., 27.]
    self.assertAllClose(y, self.evaluate(chain.forward(x)))
    self.assertAllClose(x, self.evaluate(chain.inverse(y)))
    self.assertAllClose(
        np.log(6, dtype=np.float32) + np.sum(scale_diag * x),
        self.evaluate(chain.forward_log_det_jacobian(x, event_ndims=1)))

    self.assertAllClose(
        -np.log(6, dtype=np.float32) - np.sum(scale_diag * x),
        self.evaluate(chain.inverse_log_det_jacobian(y, event_ndims=1)))

  def testChainAffineExp(self):
    scale_diag = np.array([1., 2., 3.], dtype=np.float32)
    chain = tfb.Chain([tfb.Affine(scale_diag=scale_diag), tfb.Exp()])
    x = [0., np.log(2., dtype=np.float32), np.log(3., dtype=np.float32)]
    y = [1., 4., 9.]
    self.assertAllClose(y, self.evaluate(chain.forward(x)))
    self.assertAllClose(x, self.evaluate(chain.inverse(y)))
    self.assertAllClose(
        np.log(6, dtype=np.float32) + np.sum(x),
        self.evaluate(chain.forward_log_det_jacobian(x, event_ndims=1)))

    self.assertAllClose(
        -np.log(6, dtype=np.float32) - np.sum(x),
        self.evaluate(chain.inverse_log_det_jacobian(y, event_ndims=1)))

  def testChainIldjWithPlaceholder(self):
    chain = tfb.Chain((tfb.Exp(), tfb.Exp()))
    samples = tf.compat.v1.placeholder_with_default(
        np.zeros([2, 10], np.float32), shape=None)
    ildj = chain.inverse_log_det_jacobian(samples, event_ndims=0)
    self.assertTrue(ildj is not None)
    self.evaluate(ildj)


if __name__ == "__main__":
  tf.test.main()
