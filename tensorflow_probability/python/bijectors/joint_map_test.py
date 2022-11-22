# Copyright 2020 The TensorFlow Probability Authors.
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
"""JointMap Tests."""

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.bijectors import exp
from tensorflow_probability.python.bijectors import invert
from tensorflow_probability.python.bijectors import joint_map
from tensorflow_probability.python.bijectors import ldj_ratio
from tensorflow_probability.python.bijectors import reshape
from tensorflow_probability.python.bijectors import scale
from tensorflow_probability.python.bijectors import shift
from tensorflow_probability.python.bijectors import softplus
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class JointMapBijectorTest(test_util.TestCase):
  """Tests the correctness of the Y = JointMap({nested}) transformation."""

  def assertShapeIs(self, expect_shape, observed):
    self.assertEqual(expect_shape, np.asarray(observed))

  def testBijector(self):
    bij = joint_map.JointMap({
        'a': exp.Exp(),
        'b': scale.Scale(2.),
        'c': shift.Shift(3.)
    })

    a = np.asarray([[[1, 2], [2, 3]]], dtype=np.float32)   # shape=[1, 2, 2]
    b = np.asarray([[0, 4]], dtype=np.float32)             # shape=[1, 2]
    c = np.asarray([[5, 6]], dtype=np.float32)             # shape=[1, 2]

    inputs = {'a': a, 'b': b, 'c': c}  # Could be inputs to forward or inverse.
    event_ndims = {'a': 1, 'b': 0, 'c': 0}

    self.assertStartsWith(bij.name, 'jointmap_of_exp_and_scale')
    self.assertAllCloseNested({'a': np.exp(a), 'b': b * 2., 'c': c + 3},
                              self.evaluate(bij.forward(inputs)))
    self.assertAllCloseNested({'a': np.log(a), 'b': b / 2., 'c': c - 3},
                              self.evaluate(bij.inverse(inputs)))

    fldj = self.evaluate(bij.forward_log_det_jacobian(inputs, event_ndims))
    self.assertEqual((1, 2), fldj.shape)
    self.assertAllClose(np.sum(a, axis=-1) + np.log(2), fldj)

    ildj = self.evaluate(bij.inverse_log_det_jacobian(inputs, event_ndims))
    self.assertEqual((1, 2), ildj.shape)
    self.assertAllClose(-np.log(a).sum(axis=-1) - np.log(2), ildj)

  def testBijectorWithDeepStructure(self):
    bij = joint_map.JointMap({
        'a': exp.Exp(),
        'bc': joint_map.JointMap([scale.Scale(2.),
                                  shift.Shift(3.)])
    })

    a = np.asarray([[[1, 2], [2, 3]]], dtype=np.float32)   # shape=[1, 2, 2]
    b = np.asarray([[0, 4]], dtype=np.float32)             # shape=[1, 2]
    c = np.asarray([[5, 6]], dtype=np.float32)             # shape=[1, 2]

    inputs = {'a': a, 'bc': [b, c]}  # Could be inputs to forward or inverse.
    event_ndims = {'a': 1, 'bc': [0, 0]}

    self.assertStartsWith(bij.name, 'jointmap_of_exp_and_jointmap_of_')
    self.assertAllCloseNested({'a': np.exp(a), 'bc': [b * 2., c + 3]},
                              self.evaluate(bij.forward(inputs)))
    self.assertAllCloseNested({'a': np.log(a), 'bc': [b / 2., c - 3]},
                              self.evaluate(bij.inverse(inputs)))

    fldj = self.evaluate(bij.forward_log_det_jacobian(inputs, event_ndims))
    self.assertEqual((1, 2), fldj.shape)
    self.assertAllClose(np.sum(a, axis=-1) + np.log(2), fldj)

    ildj = self.evaluate(bij.inverse_log_det_jacobian(inputs, event_ndims))
    self.assertEqual((1, 2), ildj.shape)
    self.assertAllClose(-np.log(a).sum(axis=-1) - np.log(2), ildj)

  def testBatchShapeBroadcasts(self):
    bij = joint_map.JointMap({
        'a': exp.Exp(),
        'b': scale.Scale(10.)
    },
                             validate_args=True)
    self.assertStartsWith(bij.name, 'jointmap_of_exp_and_scale')

    a = np.asarray([[[1, 2]], [[2, 3]]], dtype=np.float32)  # shape=[2, 1, 2]
    b = np.asarray([[0, 1, 2]], dtype=np.float32)  # shape=[1, 3]

    inputs = {'a': a, 'b': b}  # Could be inputs to forward or inverse.

    self.assertAllClose(
        a.sum(axis=-1) + np.log(10.),
        self.evaluate(bij.forward_log_det_jacobian(inputs, {'a': 1, 'b': 0})))

    self.assertAllClose(
        a.sum(axis=-1) + 3 * np.log(10.),
        self.evaluate(bij.forward_log_det_jacobian(inputs, {'a': 1, 'b': 1})))

  @test_util.disable_test_for_backend(
      disable_numpy=True,
      reason='NumPy backend overrides dtypes in __init__.')
  def testMixedDtypeLogDetJacobian(self):
    bij = joint_map.JointMap({
        'a': scale.Scale(tf.constant(1, dtype=tf.float16)),
        'b': scale.Scale(tf.constant(2, dtype=tf.float32)),
        'c': scale.Scale(tf.constant(3, dtype=tf.float64))
    })

    fldj = bij.forward_log_det_jacobian(
        x={'a': 4, 'b': 5, 'c': 6},
        event_ndims=dict.fromkeys('abc', 0))
    self.assertDTypeEqual(fldj, np.float64)
    self.assertAllClose(np.log(1) + np.log(2) + np.log(3), self.evaluate(fldj))

  def test_inverse_has_event_ndims(self):
    bij_reshape = invert.Invert(joint_map.JointMap([reshape.Reshape([])]))
    bij_reshape.inverse_event_ndims([10])  # expect [9]
    self.assertEqual(bij_reshape.inverse_event_ndims([10]), [9])

  @test_util.disable_test_for_backend(
      disable_numpy=True, disable_jax=True,
      reason='Numpy and JAX have no notion of CompositeTensor/saved_model.')
  def testCompositeTensor(self):
    e = exp.Exp()
    sp = softplus.Softplus()
    aff = scale.Scale(scale=2.)
    bij = joint_map.JointMap(bijectors=[e, sp, aff])
    self.assertIsInstance(bij, tf.__internal__.CompositeTensor)

    # Bijector may be flattened into `Tensor` components and rebuilt.
    flat = tf.nest.flatten(bij, expand_composites=True)
    unflat = tf.nest.pack_sequence_as(bij, flat, expand_composites=True)
    self.assertIsInstance(unflat, joint_map.JointMap)

    # Bijector may be input to a `tf.function`-decorated callable.
    @tf.function
    def call_forward(bij, x):
      return bij.forward(x)

    x = [1., 2., 3.]
    self.assertAllCloseNested(call_forward(unflat, x), bij.forward(x))

    # Type spec can be encoded/decoded.
    enc = tf.__internal__.saved_model.encode_structure(bij._type_spec)
    dec = tf.__internal__.saved_model.decode_proto(enc)
    self.assertEqual(bij._type_spec, dec)

  def testNonCompositeTensor(self):
    e = exp.Exp()
    s = bijector_test_util.NonCompositeTensorScale(scale=tf.constant(3.))
    bij = joint_map.JointMap(bijectors=[e, s])
    self.assertNotIsInstance(bij, tf.__internal__.CompositeTensor)
    self.assertAllCloseNested(
        bij.forward([1., 1.]), [e.forward(1.), s.forward(1.)])

  def testLDJRatio(self):
    q = joint_map.JointMap({
        'a': exp.Exp(),
        'b': scale.Scale(2.),
        'c': shift.Shift(3.)
    })
    p = joint_map.JointMap({
        'a': exp.Exp(),
        'b': scale.Scale(3.),
        'c': shift.Shift(4.)
    })

    a = np.asarray([[[1, 2], [2, 3]]], dtype=np.float32)   # shape=[1, 2, 2]
    b = np.asarray([[0, 4]], dtype=np.float32)             # shape=[1, 2]
    c = np.asarray([[5, 6]], dtype=np.float32)             # shape=[1, 2]

    x = {'a': a, 'b': b, 'c': c}
    y = {'a': a + 1, 'b': b + 1, 'c': c + 1}
    event_ndims = {'a': 1, 'b': 0, 'c': 0}

    fldj_ratio_true = p.forward_log_det_jacobian(
        x, event_ndims) - q.forward_log_det_jacobian(y, event_ndims)
    fldj_ratio = ldj_ratio.forward_log_det_jacobian_ratio(
        p, x, q, y, event_ndims)
    self.assertAllClose(fldj_ratio_true, fldj_ratio)

    ildj_ratio_true = p.inverse_log_det_jacobian(
        x, event_ndims) - q.inverse_log_det_jacobian(y, event_ndims)
    ildj_ratio = ldj_ratio.inverse_log_det_jacobian_ratio(
        p, x, q, y, event_ndims)
    self.assertAllClose(ildj_ratio_true, ildj_ratio)

    event_ndims = {'a': 1, 'b': 2, 'c': 0}

    fldj_ratio_true = p.forward_log_det_jacobian(
        x, event_ndims) - q.forward_log_det_jacobian(y, event_ndims)
    fldj_ratio = ldj_ratio.forward_log_det_jacobian_ratio(
        p, x, q, y, event_ndims)
    self.assertAllClose(fldj_ratio_true, fldj_ratio)

    ildj_ratio_true = p.inverse_log_det_jacobian(
        x, event_ndims) - q.inverse_log_det_jacobian(y, event_ndims)
    ildj_ratio = ldj_ratio.inverse_log_det_jacobian_ratio(
        p, x, q, y, event_ndims)
    self.assertAllClose(ildj_ratio_true, ildj_ratio)


if __name__ == '__main__':
  test_util.main()
