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

# Dependency imports
import mock

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import bijector as bijector_lib
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.bijectors import chain
from tensorflow_probability.python.bijectors import exp
from tensorflow_probability.python.bijectors import identity
from tensorflow_probability.python.bijectors import inline
from tensorflow_probability.python.bijectors import invert
from tensorflow_probability.python.bijectors import joint_map
from tensorflow_probability.python.bijectors import restructure
from tensorflow_probability.python.bijectors import scale
from tensorflow_probability.python.bijectors import scale_matvec_diag
from tensorflow_probability.python.bijectors import softmax_centered
from tensorflow_probability.python.bijectors import softplus
from tensorflow_probability.python.bijectors import split as split_lib
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util


class ShapeChanging(bijector_lib.Bijector):
  """Only used for op_ndims manipulation."""

  def __init__(self,
               forward_min_event_ndims=0,
               inverse_min_event_ndims=3):
    super(ShapeChanging, self).__init__(
        forward_min_event_ndims=forward_min_event_ndims,
        inverse_min_event_ndims=inverse_min_event_ndims,
        validate_args=False, name="shape_changer")


class PermuteParts(bijector_lib.Bijector):
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
    bijector = chain.Chain((exp.Exp(), softplus.Softplus()))
    self.assertStartsWith(bijector.name, "chain_of_exp_of_softplus")
    x = np.asarray([[[1., 2.],
                     [2., 3.]]])
    self.assertAllClose(1. + np.exp(x), self.evaluate(bijector.forward(x)))
    self.assertAllClose(np.log(x - 1.), self.evaluate(bijector.inverse(x)))
    self.assertAllClose(
        -np.sum(np.log(x - 1.), axis=2),
        self.evaluate(bijector.inverse_log_det_jacobian(x, event_ndims=1)))
    self.assertAllClose(
        np.sum(x, axis=2),
        self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=1)))

  def testBijectorIdentity(self):
    bijector = chain.Chain()
    self.assertStartsWith(bijector.name, "identity")
    x = np.asarray([[[1., 2.],
                     [2., 3.]]])
    self.assertAllClose(x, self.evaluate(bijector.forward(x)))
    self.assertAllClose(x, self.evaluate(bijector.inverse(x)))
    self.assertAllClose(
        0., self.evaluate(bijector.inverse_log_det_jacobian(x, event_ndims=1)))
    self.assertAllClose(
        0., self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=1)))

  def testNestedDtype(self):
    bijector = chain.Chain([
        identity.Identity(),
        scale.Scale(tf.constant(2., tf.float64)),
        identity.Identity()
    ])

    self.assertAllClose(
        tf.constant([2, 4, 6], tf.float64),
        self.evaluate(bijector.forward([1, 2, 3])))

  def testScalarCongruency(self):
    bijector = chain.Chain((exp.Exp(), softplus.Softplus()))
    bijector_test_util.assert_scalar_congruency(
        bijector, lower_x=1e-3, upper_x=1.5, rtol=0.05, eval_func=self.evaluate)

  def testShapeGetters(self):
    bijector = chain.Chain([
        softmax_centered.SoftmaxCentered(validate_args=True),
        softmax_centered.SoftmaxCentered(validate_args=True),
    ])
    x = tf.TensorShape([1])
    y = tf.TensorShape([2 + 1])
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

  def _validateChainMinEventNdims(self,
                                  bijectors,
                                  forward_min_event_ndims,
                                  inverse_min_event_ndims):
    bijector = chain.Chain(bijectors)
    self.assertAllEqual(forward_min_event_ndims,
                        bijector.forward_min_event_ndims)
    self.assertAllEqual(inverse_min_event_ndims,
                        bijector.inverse_min_event_ndims)

    chain_inverse = chain.Chain([invert.Invert(b) for b in reversed(bijectors)])
    self.assertAllEqual(forward_min_event_ndims,
                        chain_inverse.inverse_min_event_ndims)
    self.assertAllEqual(inverse_min_event_ndims,
                        chain_inverse.forward_min_event_ndims)

  def testMinEventNdimsChain(self):
    self._validateChainMinEventNdims(
        bijectors=[exp.Exp(), exp.Exp(), exp.Exp()],
        forward_min_event_ndims=0,
        inverse_min_event_ndims=0)

    self._validateChainMinEventNdims(
        bijectors=[
            scale_matvec_diag.ScaleMatvecDiag(scale_diag=[1., 1.]),
            scale_matvec_diag.ScaleMatvecDiag(scale_diag=[1., 1.]),
            scale_matvec_diag.ScaleMatvecDiag(scale_diag=[1., 1.])
        ],
        forward_min_event_ndims=1,
        inverse_min_event_ndims=1)

    self._validateChainMinEventNdims(
        bijectors=[
            exp.Exp(),
            scale_matvec_diag.ScaleMatvecDiag(scale_diag=[1., 1.])
        ],
        forward_min_event_ndims=1,
        inverse_min_event_ndims=1)

    self._validateChainMinEventNdims(
        bijectors=[
            scale_matvec_diag.ScaleMatvecDiag(scale_diag=[1., 1.]),
            exp.Exp(),
            softplus.Softplus(),
            scale_matvec_diag.ScaleMatvecDiag(scale_diag=[1., 1.])
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
            scale_matvec_diag.ScaleMatvecDiag(scale_diag=[1., 1.])
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
    jm_0 = joint_map.JointMap([ShapeChanging(1, 1), ShapeChanging(3, 1)])
    split = ShapeChanging(1, [1, 1])
    concat = ShapeChanging([1, 1], 1)
    jm_1 = joint_map.JointMap([ShapeChanging(1, 0), ShapeChanging(1, 1)])
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

    dependent = chain.Chain(
        [split_lib.Split(2), invert.Invert(split_lib.Split(2))])
    wrap_in_list = restructure.Restructure(
        input_structure=[0, 1], output_structure=[[0, 1]])
    dependent_as_chain = chain.Chain([
        invert.Invert(wrap_in_list),
        joint_map.JointMap([dependent]), wrap_in_list
    ])
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
      chain.Chain(
          [ShapeChanging([1, 1], [1, 1]),
           ShapeChanging([1, 1], [2, 1])])

  def testChainExpAffine(self):
    scale_diag = np.array([1., 2., 3.], dtype=np.float32)
    bijector = chain.Chain(
        [exp.Exp(),
         scale_matvec_diag.ScaleMatvecDiag(scale_diag=scale_diag)])
    x = [0., np.log(2., dtype=np.float32), np.log(3., dtype=np.float32)]
    y = [1., 4., 27.]
    self.assertAllClose(y, self.evaluate(bijector.forward(x)))
    self.assertAllClose(x, self.evaluate(bijector.inverse(y)))
    self.assertAllClose(
        np.log(6, dtype=np.float32) + np.sum(scale_diag * x),
        self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=1)))

    self.assertAllClose(
        -np.log(6, dtype=np.float32) - np.sum(scale_diag * x),
        self.evaluate(bijector.inverse_log_det_jacobian(y, event_ndims=1)))

  def testChainAffineExp(self):
    scale_diag = np.array([1., 2., 3.], dtype=np.float32)
    bijector = chain.Chain(
        [scale_matvec_diag.ScaleMatvecDiag(scale_diag=scale_diag),
         exp.Exp()])
    x = [0., np.log(2., dtype=np.float32), np.log(3., dtype=np.float32)]
    y = [1., 4., 9.]
    self.assertAllClose(y, self.evaluate(bijector.forward(x)))
    self.assertAllClose(x, self.evaluate(bijector.inverse(y)))
    self.assertAllClose(
        np.log(6, dtype=np.float32) + np.sum(x),
        self.evaluate(bijector.forward_log_det_jacobian(x, event_ndims=1)))

    self.assertAllClose(
        -np.log(6, dtype=np.float32) - np.sum(x),
        self.evaluate(bijector.inverse_log_det_jacobian(y, event_ndims=1)))

  def testEventNdimsIsOptional(self):
    scale_diag = np.array([1., 2., 3.], dtype=np.float32)
    bijector = chain.Chain(
        [scale_matvec_diag.ScaleMatvecDiag(scale_diag=scale_diag),
         exp.Exp()])
    x = [0., np.log(2., dtype=np.float32), np.log(3., dtype=np.float32)]
    y = [1., 4., 9.]
    self.assertAllClose(
        np.log(6, dtype=np.float32) + np.sum(x),
        self.evaluate(bijector.forward_log_det_jacobian(x)))
    self.assertAllClose(-np.log(6, dtype=np.float32) - np.sum(x),
                        self.evaluate(bijector.inverse_log_det_jacobian(y)))

  def testChainIldjWithPlaceholder(self):
    bijector = chain.Chain((exp.Exp(), exp.Exp()))
    samples = tf1.placeholder_with_default(
        np.zeros([2, 10], np.float32), shape=None)
    ildj = bijector.inverse_log_det_jacobian(samples, event_ndims=0)
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
    bijector = chain.Chain([
        inline.Inline(
            inverse_fn=xform_dynamic,
            forward_min_event_ndims=0,
            forward_log_det_jacobian_fn=ldj,
            forward_fn=xform_dynamic),
        inline.Inline(
            inverse_fn=xform_static,
            forward_min_event_ndims=0,
            forward_log_det_jacobian_fn=ldj,
            forward_fn=xform_static),
        inline.Inline(
            inverse_fn=xform_dynamic,
            forward_min_event_ndims=0,
            forward_log_det_jacobian_fn=ldj,
            forward_fn=xform_dynamic)
    ])

    ildj = bijector.inverse_log_det_jacobian(
        tf.zeros((2, 3), dtype=tf.float32), event_ndims=1)

    # The shape of `ildj` is known statically to be scalar; its value is
    # not statically known.
    self.assertTrue(tensorshape_util.is_fully_defined(ildj.shape))

    # `ldj_reduce_shape` uses `prefer_static` to get input shapes. That means
    # that we respect statically-known shape information where present.
    # In this case, the manually-assigned static shape is incorrect.
    self.assertEqual(self.evaluate(ildj), -7.)

    # Ditto.
    fldj = bijector.forward_log_det_jacobian([0.], event_ndims=0)
    self.assertTrue(tensorshape_util.is_fully_defined(fldj.shape))
    self.assertEqual(self.evaluate(fldj), 3.)

  def testDofChangeError(self):
    e = exp.Exp()
    smc = softmax_centered.SoftmaxCentered()

    # Increase in event-size is the last step. No problems here.
    safe_bij = chain.Chain([smc, e],
                           validate_args=True,
                           validate_event_size=True)
    self.evaluate(safe_bij.forward_log_det_jacobian([1., 2., 3.], 1))

    # Increase in event-size before Exp.
    raise_bij = chain.Chain([e, smc],
                            validate_args=True,
                            validate_event_size=True)
    with self.assertRaisesRegex((ValueError, tf.errors.InvalidArgumentError),
                                r".+degrees of freedom.+"):
      self.evaluate(raise_bij.forward_log_det_jacobian([1., 2., 3.], 1))

    # When validate_args is False, warns instead of raising.
    warn_bij = chain.Chain([e, smc],
                           validate_args=False,
                           validate_event_size=True)
    with mock.patch.object(tf, "print", return_value=tf.no_op()) as mock_print:
      self.evaluate(warn_bij.forward_log_det_jacobian([1., 2., 3.], 1))
      print_args, _ = mock_print.call_args
      self.assertRegex(print_args[0], r"WARNING:.+degrees of freedom")

    # When validate_event_shape is False, neither warns nor raises.
    ignore_bij = chain.Chain([e, smc], validate_event_size=False)
    self.evaluate(ignore_bij.forward_log_det_jacobian([1., 2., 3.], 1))

  def testDofValidationDoesNoHarm(self):
    # Chain with no change in degrees-of-freedom.
    bij = chain.Chain([exp.Exp()], validate_args=True, validate_event_size=True)
    self.evaluate(bij.forward_log_det_jacobian([1., 2., 3.], 1))

  @test_util.disable_test_for_backend(
      disable_numpy=True, disable_jax=True,
      reason="Numpy and JAX have no notion of CompositeTensor/saved_model.")
  def testCompositeTensor(self):
    e = exp.Exp()
    sp = softplus.Softplus()
    aff = scale.Scale(scale=2.)
    bijector = chain.Chain(bijectors=[e, sp, aff])
    self.assertIsInstance(bijector, tf.__internal__.CompositeTensor)

    # Bijector may be flattened into `Tensor` components and rebuilt.
    flat = tf.nest.flatten(bijector, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(bijector, flat, expand_composites=True)
    self.assertIsInstance(unflat, chain.Chain)

    # Bijector may be input to a `tf.function`-decorated callable.
    @tf.function
    def call_forward(bij, x):
      return bij.forward(x)

    x = tf.ones([2, 3], dtype=tf.float32)
    self.assertAllClose(call_forward(unflat, x), bijector.forward(x))

    # TypeSpec can be encoded/decoded.
    enc = tf.__internal__.saved_model.encode_structure(bijector._type_spec)
    dec = tf.__internal__.saved_model.decode_proto(enc)
    self.assertEqual(bijector._type_spec, dec)

  def testNonCompositeTensor(self):
    e = exp.Exp()
    s = bijector_test_util.NonCompositeTensorScale(scale=tf.constant(3.))
    bijector = chain.Chain(bijectors=[e, s])
    self.assertNotIsInstance(bijector, tf.__internal__.CompositeTensor)
    self.assertAllClose(bijector.forward([1.]), e.forward(s.forward([1.])))

if __name__ == "__main__":
  test_util.main()
