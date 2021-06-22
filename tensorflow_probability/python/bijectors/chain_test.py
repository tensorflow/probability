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
import mock

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util


class ShapeChanging(tfb.Bijector):
  """Only used for op_ndims manipulation."""

  def __init__(self,
               forward_min_event_ndims=0,
               inverse_min_event_ndims=3):
    super(ShapeChanging, self).__init__(
        forward_min_event_ndims=forward_min_event_ndims,
        inverse_min_event_ndims=inverse_min_event_ndims,
        validate_args=False, name="shape_changer")


class PermuteParts(tfb.Bijector):
  """Only used for op_ndims manipulation."""

  def __init__(self):
    super(PermuteParts, self).__init__(
        forward_min_event_ndims=[0, 0],
        inverse_min_event_ndims=[0, 0],
        validate_args=False, name="permute_parts")

  def forward_event_ndims(self, event_ndims):
    return [event_ndims[1], event_ndims[0]]

  def inverse_event_ndims(self, event_ndims):
    return [event_ndims[1], event_ndims[0]]

  @property
  def _parts_interact(self):
    return False


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

  def testNestedDtype(self):
    chain = tfb.Chain([
        tfb.Identity(),
        tfb.Scale(tf.constant(2., tf.float64)),
        tfb.Identity()
    ])

    self.assertAllClose(tf.constant([2, 4, 6], tf.float64),
                        self.evaluate(chain.forward([1, 2, 3])))

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

  def _validateChainMinEventNdims(self,
                                  bijectors,
                                  forward_min_event_ndims,
                                  inverse_min_event_ndims):
    chain = tfb.Chain(bijectors)
    self.assertAllEqual(forward_min_event_ndims,
                        chain.forward_min_event_ndims)
    self.assertAllEqual(inverse_min_event_ndims,
                        chain.inverse_min_event_ndims)

    chain_inverse = tfb.Chain([tfb.Invert(b) for b in reversed(bijectors)])
    self.assertAllEqual(forward_min_event_ndims,
                        chain_inverse.inverse_min_event_ndims)
    self.assertAllEqual(inverse_min_event_ndims,
                        chain_inverse.forward_min_event_ndims)

  def testMinEventNdimsChain(self):
    self._validateChainMinEventNdims(
        bijectors=[
            tfb.Exp(),
            tfb.Exp(),
            tfb.Exp()
        ],
        forward_min_event_ndims=0,
        inverse_min_event_ndims=0)

    self._validateChainMinEventNdims(
        bijectors=[
            tfb.ScaleMatvecDiag(scale_diag=[1., 1.]),
            tfb.ScaleMatvecDiag(scale_diag=[1., 1.]),
            tfb.ScaleMatvecDiag(scale_diag=[1., 1.])
        ],
        forward_min_event_ndims=1,
        inverse_min_event_ndims=1)

    self._validateChainMinEventNdims(
        bijectors=[
            tfb.Exp(),
            tfb.ScaleMatvecDiag(scale_diag=[1., 1.])
        ],
        forward_min_event_ndims=1,
        inverse_min_event_ndims=1)

    self._validateChainMinEventNdims(
        bijectors=[
            tfb.ScaleMatvecDiag(scale_diag=[1., 1.]),
            tfb.Exp(),
            tfb.Softplus(),
            tfb.ScaleMatvecDiag(scale_diag=[1., 1.])
        ],
        forward_min_event_ndims=1,
        inverse_min_event_ndims=1)

  def testMinEventNdimsShapeChangingAddDims(self):
    self._validateChainMinEventNdims(
        bijectors=[
            ShapeChanging()
        ],
        forward_min_event_ndims=0,
        inverse_min_event_ndims=3)

    self._validateChainMinEventNdims(
        bijectors=[
            ShapeChanging(),
            tfb.ScaleMatvecDiag(scale_diag=[1., 1.])
        ],
        forward_min_event_ndims=1,
        inverse_min_event_ndims=4)

    self._validateChainMinEventNdims(
        bijectors=[
            ShapeChanging(),
            ShapeChanging()
        ],
        forward_min_event_ndims=0,
        inverse_min_event_ndims=6)

  def testMinEventNdimsShapeChangingAddRemoveDims(self):
    self._validateChainMinEventNdims(
        bijectors=[
            ShapeChanging(2, 1),
            ShapeChanging(3, 0),
            ShapeChanging(1, 2)
        ],
        forward_min_event_ndims=4,
        inverse_min_event_ndims=1)

  def testMinEventNdimsWithJointMap(self):
    jm_0 = tfb.JointMap([ShapeChanging(1, 1), ShapeChanging(3, 1)])
    split = ShapeChanging(1, [1, 1])
    concat = ShapeChanging([1, 1], 1)
    jm_1 = tfb.JointMap([ShapeChanging(1, 0), ShapeChanging(1, 1)])
    permute = PermuteParts()

    self._validateChainMinEventNdims(
        bijectors=[jm_0, split, concat, jm_1],
        forward_min_event_ndims=[4, 3],
        inverse_min_event_ndims=[3, 1])

    self._validateChainMinEventNdims(
        bijectors=[jm_0, jm_1],
        forward_min_event_ndims=[2, 3],
        inverse_min_event_ndims=[1, 1])

    self._validateChainMinEventNdims(
        bijectors=[jm_1, jm_0],
        forward_min_event_ndims=[1, 3],
        inverse_min_event_ndims=[0, 1])

    self._validateChainMinEventNdims(
        bijectors=[jm_1, permute, jm_0],
        forward_min_event_ndims=[1, 3],
        inverse_min_event_ndims=[0, 1])

    self._validateChainMinEventNdims(
        bijectors=[jm_0, split],
        forward_min_event_ndims=3,
        inverse_min_event_ndims=[3, 1])

    self._validateChainMinEventNdims(
        bijectors=[permute, jm_1, split],
        forward_min_event_ndims=1,
        inverse_min_event_ndims=[1, 0])

  def testMinEventNdimsWithPartiallyDependentJointMap(self):

    dependent = tfb.Chain([tfb.Split(2), tfb.Invert(tfb.Split(2))])
    wrap_in_list = tfb.Restructure(input_structure=[0, 1],
                                   output_structure=[[0, 1]])
    dependent_as_chain = tfb.Chain([
        tfb.Invert(wrap_in_list),
        tfb.JointMap([dependent]),
        wrap_in_list])
    self.assertAllEqualNested(dependent.forward_min_event_ndims,
                              dependent_as_chain.forward_min_event_ndims)
    self.assertAllEqualNested(dependent.inverse_min_event_ndims,
                              dependent_as_chain.inverse_min_event_ndims)
    self.assertAllEqualNested(dependent._parts_interact,
                              dependent_as_chain._parts_interact)

  def testInvalidChainNdimsRaisesError(self):
    with self.assertRaisesRegexp(
        ValueError,
        "Differences between `event_ndims` and `min_event_ndims must be equal"):
      tfb.Chain([ShapeChanging([1, 1], [1, 1]),
                 ShapeChanging([1, 1], [2, 1])])

  def testChainExpAffine(self):
    scale_diag = np.array([1., 2., 3.], dtype=np.float32)
    chain = tfb.Chain([tfb.Exp(), tfb.ScaleMatvecDiag(scale_diag=scale_diag)])
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
    chain = tfb.Chain([tfb.ScaleMatvecDiag(scale_diag=scale_diag), tfb.Exp()])
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

  def testEventNdimsIsOptional(self):
    scale_diag = np.array([1., 2., 3.], dtype=np.float32)
    chain = tfb.Chain([tfb.ScaleMatvecDiag(scale_diag=scale_diag), tfb.Exp()])
    x = [0., np.log(2., dtype=np.float32), np.log(3., dtype=np.float32)]
    y = [1., 4., 9.]
    self.assertAllClose(
        np.log(6, dtype=np.float32) + np.sum(x),
        self.evaluate(chain.forward_log_det_jacobian(x)))
    self.assertAllClose(
        -np.log(6, dtype=np.float32) - np.sum(x),
        self.evaluate(chain.inverse_log_det_jacobian(y)))

  def testChainIldjWithPlaceholder(self):
    chain = tfb.Chain((tfb.Exp(), tfb.Exp()))
    samples = tf1.placeholder_with_default(
        np.zeros([2, 10], np.float32), shape=None)
    ildj = chain.inverse_log_det_jacobian(samples, event_ndims=0)
    self.assertIsNotNone(ildj)
    self.evaluate(ildj)

  def testChainDynamicToStatic(self):
    if tf.executing_eagerly():
      return

    def xform_dynamic(x):
      return tf1.placeholder_with_default(x, shape=None)

    def xform_static(x):
      # Copy the Tensor, because otherwise the set_shape can pass information
      # into the past.
      x = tf.identity(x)
      tensorshape_util.set_shape(x, [1])
      return x

    def ldj(_):
      return tf.constant(1.)

    # The issue was that the sample's shape was going in-and-out of being fully
    # specified, causing internal consistency issues inside the bijector.
    chain = tfb.Chain([
        tfb.Inline(
            inverse_fn=xform_dynamic,
            forward_min_event_ndims=0,
            forward_log_det_jacobian_fn=ldj,
            forward_fn=xform_dynamic),
        tfb.Inline(
            inverse_fn=xform_static,
            forward_min_event_ndims=0,
            forward_log_det_jacobian_fn=ldj,
            forward_fn=xform_static),
        tfb.Inline(
            inverse_fn=xform_dynamic,
            forward_min_event_ndims=0,
            forward_log_det_jacobian_fn=ldj,
            forward_fn=xform_dynamic)
    ])

    ildj = chain.inverse_log_det_jacobian(
        tf.zeros((2, 3), dtype=tf.float32), event_ndims=1)

    # The shape of `ildj` is known statically to be scalar; its value is
    # not statically known.
    self.assertTrue(tensorshape_util.is_fully_defined(ildj.shape))

    # `ldj_reduce_shape` uses `prefer_static` to get input shapes. That means
    # that we respect statically-known shape information where present.
    # In this case, the manually-assigned static shape is incorrect.
    self.assertEqual(self.evaluate(ildj), -7.)

    # Ditto.
    fldj = chain.forward_log_det_jacobian([0.], event_ndims=0)
    self.assertTrue(tensorshape_util.is_fully_defined(fldj.shape))
    self.assertEqual(self.evaluate(fldj), 3.)

  def testDofChangeError(self):
    exp = tfb.Exp()
    smc = tfb.SoftmaxCentered()

    # Increase in event-size is the last step. No problems here.
    safe_bij = tfb.Chain([smc, exp],
                         validate_args=True,
                         validate_event_size=True)
    self.evaluate(safe_bij.forward_log_det_jacobian([1., 2., 3.], 1))

    # Increase in event-size before Exp.
    raise_bij = tfb.Chain([exp, smc],
                          validate_args=True,
                          validate_event_size=True)
    with self.assertRaisesRegex((ValueError, tf.errors.InvalidArgumentError),
                                r".+degrees of freedom.+"):
      self.evaluate(raise_bij.forward_log_det_jacobian([1., 2., 3.], 1))

    # When validate_args is False, warns instead of raising.
    warn_bij = tfb.Chain([exp, smc],
                         validate_args=False,
                         validate_event_size=True)
    with mock.patch.object(tf, "print", return_value=tf.no_op()) as mock_print:
      self.evaluate(warn_bij.forward_log_det_jacobian([1., 2., 3.], 1))
      print_args, _ = mock_print.call_args
      self.assertRegex(print_args[0], r"WARNING:.+degrees of freedom")

    # When validate_event_shape is False, neither warns nor raises.
    ignore_bij = tfb.Chain([exp, smc], validate_event_size=False)
    self.evaluate(ignore_bij.forward_log_det_jacobian([1., 2., 3.], 1))

  @test_util.disable_test_for_backend(
      disable_numpy=True, disable_jax=True,
      reason="Numpy and JAX have no notion of CompositeTensor/saved_model.")
  def testCompositeTensor(self):
    exp = tfb.Exp()
    sp = tfb.Softplus()
    aff = tfb.Scale(scale=2.)
    chain = tfb.Chain(bijectors=[exp, sp, aff])
    self.assertIsInstance(chain, tf.__internal__.CompositeTensor)

    # Bijector may be flattened into `Tensor` components and rebuilt.
    flat = tf.nest.flatten(chain, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(chain, flat, expand_composites=True)
    self.assertIsInstance(unflat, tfb.Chain)

    # Bijector may be input to a `tf.function`-decorated callable.
    @tf.function
    def call_forward(bij, x):
      return bij.forward(x)

    x = tf.ones([2, 3], dtype=tf.float32)
    self.assertAllClose(call_forward(unflat, x), chain.forward(x))

    # TypeSpec can be encoded/decoded.
    struct_coder = tf.__internal__.saved_model.StructureCoder()
    enc = struct_coder.encode_structure(chain._type_spec)
    dec = struct_coder.decode_proto(enc)
    self.assertEqual(chain._type_spec, dec)

  def testNonCompositeTensor(self):

    class NonCompositeScale(tfb.Bijector):
      """Bijector that is not a `CompositeTensor`."""

      def __init__(self, scale):
        parameters = dict(locals())
        self.scale = scale
        super(NonCompositeScale, self).__init__(
            validate_args=True,
            forward_min_event_ndims=0.,
            parameters=parameters,
            name="non_composite_scale")

      def _forward(self, x):
        return x * self.scale

      def _inverse(self, y):
        return y / self.scale

    exp = tfb.Exp()
    scale = NonCompositeScale(scale=tf.constant(3.))
    chain = tfb.Chain(bijectors=[exp, scale])
    self.assertNotIsInstance(chain, tf.__internal__.CompositeTensor)
    self.assertAllClose(chain.forward([1.]), exp.forward(scale.forward([1.])))

if __name__ == "__main__":
  tf.test.main()
