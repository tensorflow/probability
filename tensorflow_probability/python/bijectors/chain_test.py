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
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util


class ShapeChanging(tfb.Bijector):
  """Only used for op_ndims manipulation."""

  def __init__(self, forward_min_event_ndims=0, inverse_min_event_ndims=3):
    super(ShapeChanging, self).__init__(
        forward_min_event_ndims=forward_min_event_ndims,
        inverse_min_event_ndims=inverse_min_event_ndims,
        validate_args=False, name="shape_changer")


@test_util.test_all_tf_execution_regimes
class ChainBijectorTest(test_util.TestCase):
  """Tests the correctness of the Y = Chain(bij1, bij2, bij3) transformation."""

  def testBijector(self):
    chain = tfb.Chain((tfb.Exp(), tfb.Softplus()))
    self.assertStartsWith(chain.name, "chain_of_exp_of_softplus")
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
    self.assertStartsWith(chain.name, "identity")
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
        tensorshape_util.as_list(y),
        self.evaluate(
            chain.forward_event_shape_tensor(tensorshape_util.as_list(x))))
    self.assertAllEqual(x, chain.inverse_event_shape(y))
    self.assertAllEqual(
        tensorshape_util.as_list(x),
        self.evaluate(
            chain.inverse_event_shape_tensor(tensorshape_util.as_list(y))))

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
    samples = tf1.placeholder_with_default(
        np.zeros([2, 10], np.float32), shape=None)
    ildj = chain.inverse_log_det_jacobian(samples, event_ndims=0)
    self.assertTrue(ildj is not None)
    self.evaluate(ildj)

  def testChainDynamicToStatic(self):
    if tf.executing_eagerly():
      return

    def xform_dynamic(x):
      return tf1.placeholder_with_default(x, shape=None)

    def xform_static(x):
      tensorshape_util.set_shape(x, [1])
      return x

    def ldj(_):
      return tf.constant(0.)

    # The issue was that the sample's shape was going in-and-out of being fully
    # specified, causing internal consistency issues inside the bijector.
    chain = tfb.Chain([
        tfb.Inline(
            inverse_log_det_jacobian_fn=ldj,
            inverse_fn=xform_dynamic,
            forward_min_event_ndims=0,
            forward_log_det_jacobian_fn=ldj,
            forward_fn=xform_dynamic),
        tfb.Inline(
            inverse_log_det_jacobian_fn=ldj,
            inverse_fn=xform_static,
            forward_min_event_ndims=0,
            forward_log_det_jacobian_fn=ldj,
            forward_fn=xform_static),
        tfb.Inline(
            inverse_log_det_jacobian_fn=ldj,
            inverse_fn=xform_dynamic,
            forward_min_event_ndims=0,
            forward_log_det_jacobian_fn=ldj,
            forward_fn=xform_dynamic)
    ])

    ildj = chain.inverse_log_det_jacobian([0.], event_ndims=0)
    # The static shape information is lost on the account of the final bijector
    # being dynamic.
    self.assertFalse(tensorshape_util.is_fully_defined(ildj.shape))
    fldj = chain.forward_log_det_jacobian([0.], event_ndims=0)
    # Ditto.
    self.assertFalse(tensorshape_util.is_fully_defined(fldj.shape))


if __name__ == "__main__":
  tf.test.main()
