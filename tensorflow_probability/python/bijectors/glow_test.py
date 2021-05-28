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
"""Tests for tensorflow_probability.python.bijectors.glow."""

import functools
# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.bijectors import blockwise
from tensorflow_probability.python.bijectors import composition
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.gradient import batch_jacobian

tfb = tfp.bijectors
tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class GlowTest(test_util.TestCase):

  def setUp(self):
    super(GlowTest, self).setUp()

    self.batch_shape = 5
    self.output_shape = [16, 16, 3]
    self.minval = -10.
    self.maxval = 10.

  def _create_glow(self, actnorm=False):

    glow_net = functools.partial(tfb.GlowDefaultNetwork, num_hidden=32)
    glow_exit = tfb.GlowDefaultExitNetwork
    glow_bijector = tfb.Glow(
        output_shape=self.output_shape,
        num_glow_blocks=2,
        num_steps_per_block=1,
        coupling_bijector_fn=glow_net,
        exit_bijector_fn=glow_exit,
        grab_after_block=[0.5, 0.5],
        use_actnorm=actnorm,
        seed=tfp.util.SeedStream(seed=42, salt='glow'))
    sigm = tfb.sigmoid.Sigmoid(
        low=self.minval, high=self.maxval)
    output = tfb.chain.Chain(
        [sigm, glow_bijector])
    return output

  def _make_images(self):
    sc = np.float32(self.maxval-self.minval)*0.999
    sh = np.float32(self.minval)+0.0005
    these_images = np.random.random([self.batch_shape]+self.output_shape)*sc+sh
    return np.float32(these_images)

  def testBijector(self):
    """Verify that the bijector is properly invertible."""
    bijection = self._create_glow()
    self.assertStartsWith(bijection.bijectors[1].name, 'glow')
    self.evaluate([v.initializer for v in bijection.variables])
    x = self._make_images()

    x = tf.constant(x, tf.float32)
    z = bijection.inverse(x)
    xprime = bijection.forward(tf.identity(z))
    zprime = bijection.inverse(xprime)

    # Absolute errors are < 1e-4, and need to use absolute since dealing with
    # values surrounding 0.
    self.assertAllClose(x, xprime, rtol=0., atol=1.e-4)
    self.assertAllClose(z, zprime, rtol=0., atol=1.e-4)
    self.assertAllClose(
        self.evaluate(-bijection.inverse_log_det_jacobian(x, 3)),
        self.evaluate(bijection.forward_log_det_jacobian(z, 1)),
        rtol=1e-4,
        atol=0.)

  def testBijectiveAndFinite(self):
    """Test that the bijector is invertible and that outputs are finite."""
    bijection = self._create_glow()
    self.evaluate([v.initializer for v in bijection.variables])
    ims = np.zeros([10]+self.output_shape)
    ims += np.linspace(self.minval+0.0001,
                       self.maxval-0.0001,
                       10, dtype=np.float32).reshape([10, 1, 1, 1])

    zs = np.float32(np.zeros([10, np.prod(self.output_shape)]))
    zs += np.linspace(0., 10., 10,
                      dtype=np.float32).reshape(10, 1)  # 10-sigma is far away.

    bijector_test_util.assert_bijective_and_finite(
        bijection, np.float32(zs), np.float32(ims), event_ndims=1,
        eval_func=self.evaluate, inverse_event_ndims=3,
        rtol=1e-3)

  def testDataInit_inverse(self):
    """Test that actnorm data dependent initialization works on inverse pass."""
    bijection = self._create_glow(actnorm=True)
    self.evaluate([v.initializer for v in bijection.variables])
    x = self._make_images()
    nblocks = 0
    made_a_check = False
    splits = bijection.bijectors[1].blockwise_splits
    splits = [[bs[0]+bs[1], bs[2]] for bs in splits]

    x = bijection.bijectors[0].inverse(x)
    for b in bijection.bijectors[1].bijectors:
      if isinstance(b, blockwise._Blockwise):
        x1, x2 = tf.split(x, splits[-2-nblocks], axis=-1)

        for bb in b.bijectors[0].bijectors:
          if isinstance(bb, tfb.glow.ActivationNormalization):
            x1 = self.evaluate(bb.inverse(x1))
            mean = self.evaluate(tf.reduce_mean(x1, axis=(-4, -3, -2)))
            stddev = self.evaluate(tf.math.reduce_std(x1, axis=(-4, -3, -2)))
            self.assertAllClose(mean, tf.zeros_like(mean), atol=1e-5)
            self.assertAllClose(stddev, tf.ones_like(stddev), atol=0, rtol=5e-5)
            made_a_check = True
          else:
            x1 = self.evaluate(bb.inverse(x1))
        x = tf.concat([x1, x2], axis=-1)
        nblocks += 1

      elif isinstance(b, composition.Composition):
        for bb in b.bijectors:
          x = self.evaluate(bb.inverse(x))
          if isinstance(bb, tfb.glow.ActivationNormalization):
            mean = tf.reduce_mean(x, axis=(-4, -3, -2))
            stddev = tf.math.reduce_std(x, axis=(-4, -3, -2))
            self.assertAllClose(mean, tf.zeros_like(mean), atol=1e-5)
            self.assertAllClose(stddev, tf.ones_like(stddev), atol=0, rtol=5e-5)
            made_a_check = True
      else:
        x = self.evaluate(b.inverse(x))

    assert made_a_check

  def testDataInit_forward(self):
    """Test that actnorm data dependent initialization works on forward pass."""
    bijection = self._create_glow(actnorm=True)
    self.evaluate([v.initializer for v in bijection.variables])
    y = np.float32(np.random.normal(0., 1., [5, 768]))
    nblocks = 0
    made_a_check = False
    splits = bijection.bijectors[1].blockwise_splits
    splits = [[bs[0]+bs[1], bs[2]] for bs in splits]

    for b in reversed(bijection.bijectors[1].bijectors):
      if isinstance(b, blockwise._Blockwise):
        y1, y2 = tf.split(y, splits[nblocks], axis=-1)

        for bb in reversed(b.bijectors[0].bijectors):
          if isinstance(bb, tfb.glow.ActivationNormalization):
            y1 = self.evaluate(bb.forward(y1))
            mean = self.evaluate(tf.reduce_mean(y1, axis=(-4, -3, -2)))
            stddev = self.evaluate(tf.math.reduce_std(y1, axis=(-4, -3, -2)))
            self.assertAllClose(mean, tf.zeros_like(mean), atol=1e-5)
            self.assertAllClose(stddev, tf.ones_like(stddev), atol=0, rtol=5e-5)
            made_a_check = True
          else:
            y1 = self.evaluate(bb.forward(y1))
        y = tf.concat([y1, y2], axis=-1)
        nblocks += 1

      elif isinstance(b, composition.Composition):
        for bb in reversed(b.bijectors):
          y = self.evaluate(bb.forward(y))
          if isinstance(bb, tfb.glow.ActivationNormalization):
            mean = tf.reduce_mean(y, axis=(-4, -3, -2))
            stddev = tf.math.reduce_std(y, axis=(-4, -3, -2))
            self.assertAllClose(mean, tf.zeros_like(mean), atol=1e-5)
            self.assertAllClose(stddev, tf.ones_like(stddev), atol=0, rtol=5e-5)
            made_a_check = True
      else:
        y = self.evaluate(b.forward(y))
    assert made_a_check

  def testInverseLogDetJacobian(self):
    """Test if log-det-jacobian agrees with numerical computation."""
    if not tf.executing_eagerly():
      self.skipTest('Theres a problem with numerical computation of Jacobian.'
                    'the bijector jacobian implementation  still returns'
                    'roughly the same values as it does in eager mode, so I'
                    'think our computation works here.')

    x = self._make_images()
    x = tf.constant(x, tf.float32)

    bijection = self._create_glow(actnorm=True)

    self.evaluate([v.initializer for v in bijection.variables])
    jacob_manual = self.evaluate(batch_jacobian(bijection.inverse, x))
    _, ldj_manual = np.linalg.slogdet(jacob_manual.reshape([5, 768, 768]))

    jacob = self.evaluate(bijection.inverse_log_det_jacobian(x, 3))
    self.assertAllClose(ldj_manual, jacob, rtol=1.e-5)

  def testBadBijectorFn(self):
    """Test a bad bijector function.

    Ensures that an error happens if the bijector outputs a tensor whose shape
    does not give a straightforward way to produce a bijector.
    """
    def bad_glow_net(input_shape):
      del input_shape
      return tfb.Pad()

    x = self._make_images()
    x = tf.constant(x, tf.float32)

    bijection = tfb.Glow(
        output_shape=self.output_shape,
        num_glow_blocks=2,
        num_steps_per_block=1,
        coupling_bijector_fn=bad_glow_net,
        exit_bijector_fn=tfb.GlowDefaultExitNetwork,
        grab_after_block=[0.5, 0.5],
        use_actnorm=False)
    self.evaluate([v.initializer for v in bijection.variables])
    try:
      z = bijection.inverse(x)
      del z
      raise Exception('If youre reading this, then something is wrong.')
    except ValueError:
      pass

  def testOtherBadBijectorFn(self):
    """Test another bad bijector function.

    This tests that an error is thrown if the bijector function produces
    something other than a bijector or a tensor
    """
    def other_bad_glow_net(input_shape):
      del input_shape
      return lambda x: tfd.Normal(x, 0)

    x = self._make_images()
    x = tf.constant(x, tf.float32)

    bijection = tfb.Glow(
        output_shape=self.output_shape,
        num_glow_blocks=2,
        num_steps_per_block=1,
        coupling_bijector_fn=other_bad_glow_net,
        exit_bijector_fn=tfb.GlowDefaultExitNetwork,
        grab_after_block=[0.5, 0.5],
        use_actnorm=False)
    self.evaluate([v.initializer for v in bijection.variables])
    try:
      z = bijection.inverse(x)
      del z
      raise Exception('If youre reading this, then something is wrong.')

    except ValueError:
      pass

  def testMultiDimensionalSampling(self):
    """This tests that the model can take samples in variable dimensions."""
    def dummy_glow_net(input_shape):
      del input_shape
      return lambda x: tfb.Identity()

    def dummy_exit_net(input_shape, output_chan):
      del input_shape, output_chan
      return lambda x: tfb.Identity()

    bijection = tfb.Glow(
        output_shape=self.output_shape,
        num_glow_blocks=2,
        num_steps_per_block=1,
        coupling_bijector_fn=dummy_glow_net,
        exit_bijector_fn=dummy_exit_net,
        grab_after_block=[0.5, 0.5],
        use_actnorm=False)
    dist = bijection(tfd.Independent(
        tfd.Normal(tf.zeros([16*16*3]), tf.ones(16*16*3)),
        reinterpreted_batch_ndims=1))

    single_samp2 = dist.sample(1)
    single_samp3 = dist.sample([1, 1])
    single_samp = dist.sample()
    self.assertShapeEqual(np.zeros([16, 16, 3]), single_samp)
    self.assertShapeEqual(np.zeros([1, 16, 16, 3]), single_samp2)
    self.assertShapeEqual(np.zeros([1, 1, 16, 16, 3]), single_samp3)

  def testMultiDimensionalInput(self):
    """This tests if the model runs with different batch shapes."""
    def dummy_glow_net(input_shape):
      del input_shape
      return lambda x: tfb.Identity()

    def dummy_exit_net(input_shape, output_chan):
      del input_shape, output_chan
      return lambda x: tfb.Identity()

    bijection = tfb.Glow(
        output_shape=self.output_shape,
        num_glow_blocks=2,
        num_steps_per_block=1,
        coupling_bijector_fn=dummy_glow_net,
        exit_bijector_fn=dummy_exit_net,
        grab_after_block=[0.5, 0.5],
        use_actnorm=False)

    single_z = bijection.inverse(tf.zeros([16, 16, 3]))
    single_z2 = bijection.inverse(tf.zeros([1, 16, 16, 3]))
    single_z3 = bijection.inverse(tf.zeros([1, 1, 16, 16, 3]))
    self.assertShapeEqual(np.zeros([16 * 16 * 3]), single_z)
    self.assertShapeEqual(np.zeros([1, 16 * 16 * 3]), single_z2)
    self.assertShapeEqual(np.zeros([1, 1, 16 * 16 * 3]), single_z3)

  def testDtypes(self):
    """Test if the bijector identifies dtype correctly."""
    ims = self._make_images()
    bijection = self._create_glow(False)
    z = bijection.inverse(ims)

    def float64_net(input_shape):
      input_nchan = input_shape[-1]
      return tf.keras.Sequential([
          tf.keras.layers.Input(input_shape, dtype=tf.float64),
          tf.keras.layers.Conv2D(
              2 * input_nchan, 3, padding='same', dtype=tf.float64)])
    def float64_exit(input_shape, output_chan):
      return tf.keras.Sequential([
          tf.keras.layers.Input(input_shape, dtype=tf.float64),
          tf.keras.layers.Conv2D(
              2*output_chan, 3, padding='same', dtype=tf.float64)])

    float64_bijection = tfb.Glow(
        output_shape=self.output_shape,
        num_glow_blocks=2,
        num_steps_per_block=1,
        coupling_bijector_fn=float64_net,
        exit_bijector_fn=float64_exit,
        grab_after_block=[0.5, 0.5]
        )
    zf64 = float64_bijection.inverse(tf.cast(ims, tf.float64))
    self.evaluate([v.initializer for v in bijection.variables])
    self.evaluate([v.initializer for v in float64_bijection.variables])
    self.assertAllFinite(self.evaluate(z))
    self.assertAllFinite(self.evaluate(zf64))

  def testBijectorFn(self):
    """Test if the bijector function works for additive coupling."""
    ims = self._make_images()
    def shiftfn(input_shape):
      input_nchan = input_shape[-1]
      return tf.keras.Sequential([
          tf.keras.layers.Input(input_shape),
          tf.keras.layers.Conv2D(
              input_nchan, 3, padding='same')])

    def shiftexitfn(input_shape, output_chan):
      return tf.keras.Sequential([
          tf.keras.layers.Input(input_shape),
          tf.keras.layers.Conv2D(
              output_chan, 3, padding='same')])

    shiftonlyglow = tfb.Glow(
        output_shape=self.output_shape,
        num_glow_blocks=2,
        num_steps_per_block=1,
        coupling_bijector_fn=shiftfn,
        exit_bijector_fn=shiftexitfn,
        grab_after_block=[0.5, 0.5]
        )
    z = shiftonlyglow.inverse(ims)
    self.evaluate([v.initializer for v in shiftonlyglow.variables])
    self.assertAllFinite(self.evaluate(z))


if __name__ == '__main__':
  tf.test.main()
