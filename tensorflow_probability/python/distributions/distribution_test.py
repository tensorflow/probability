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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfe = tf.contrib.eager


@tfe.run_all_tests_in_graph_and_eager_modes
class DistributionTest(tf.test.TestCase):

  def testParamShapesAndFromParams(self):
    classes = [
        tfd.Normal,
        tfd.Bernoulli,
        tfd.Beta,
        tfd.Chi2,
        tfd.Exponential,
        tfd.Gamma,
        tfd.InverseGamma,
        tfd.Laplace,
        tfd.StudentT,
        tfd.Uniform,
    ]

    sample_shapes = [(), (10,), (10, 20, 30)]
    for cls in classes:
      for sample_shape in sample_shapes:
        param_shapes = cls.param_shapes(sample_shape)
        params = dict([(name, tf.random_normal(shape))
                       for name, shape in param_shapes.items()])
        dist = cls(**params)
        self.assertAllEqual(sample_shape, self.evaluate(
            tf.shape(dist.sample())))
        dist_copy = dist.copy()
        self.assertAllEqual(sample_shape,
                            self.evaluate(tf.shape(dist_copy.sample())))
        self.assertEqual(dist.parameters, dist_copy.parameters)

  def testCopyExtraArgs(self):
    # Note: we cannot easily test all distributions since each requires
    # different initialization arguments. We therefore spot test a few.
    normal = tfd.Normal(loc=1., scale=2., validate_args=True)
    self.assertEqual(normal.parameters, normal.copy().parameters)
    wishart = tfd.Wishart(df=2, scale=[[1., 2], [2, 5]], validate_args=True)
    self.assertEqual(wishart.parameters, wishart.copy().parameters)

  def testCopyOverride(self):
    normal = tfd.Normal(loc=1., scale=2., validate_args=True)
    unused_normal_copy = normal.copy(validate_args=False)
    base_params = normal.parameters.copy()
    copy_params = normal.copy(validate_args=False).parameters.copy()
    self.assertNotEqual(
        base_params.pop("validate_args"), copy_params.pop("validate_args"))
    self.assertEqual(base_params, copy_params)

  def testIsScalar(self):
    mu = 1.
    sigma = 2.

    normal = tfd.Normal(mu, sigma, validate_args=True)
    self.assertTrue(tf.contrib.util.constant_value(normal.is_scalar_event()))
    self.assertTrue(tf.contrib.util.constant_value(normal.is_scalar_batch()))

    normal = tfd.Normal([mu], [sigma], validate_args=True)
    self.assertTrue(tf.contrib.util.constant_value(normal.is_scalar_event()))
    self.assertFalse(tf.contrib.util.constant_value(normal.is_scalar_batch()))

    mvn = tfd.MultivariateNormalDiag([mu], [sigma], validate_args=True)
    self.assertFalse(tf.contrib.util.constant_value(mvn.is_scalar_event()))
    self.assertTrue(tf.contrib.util.constant_value(mvn.is_scalar_batch()))

    mvn = tfd.MultivariateNormalDiag([[mu]], [[sigma]], validate_args=True)
    self.assertFalse(tf.contrib.util.constant_value(mvn.is_scalar_event()))
    self.assertFalse(tf.contrib.util.constant_value(mvn.is_scalar_batch()))

    # We now test every codepath within the underlying is_scalar_helper
    # function.

    # Test case 1, 2.
    x = tf.placeholder_with_default(input=1, shape=[])
    # None would fire an exception were it actually executed.
    self.assertTrue(normal._is_scalar_helper(x.shape, lambda: None))
    self.assertTrue(
        normal._is_scalar_helper(tf.TensorShape(None), lambda: tf.shape(x)))

    x = tf.placeholder_with_default(input=[1], shape=[1])
    # None would fire an exception were it actually executed.
    self.assertFalse(normal._is_scalar_helper(x.shape, lambda: None))
    self.assertFalse(
        normal._is_scalar_helper(tf.TensorShape(None), lambda: tf.shape(x)))

    # There's no notion of partially known shapes in eager mode, so exit
    # early.
    if tf.executing_eagerly():
      return

    # Test case 3.
    x = tf.placeholder_with_default(input=1, shape=None)
    is_scalar = normal._is_scalar_helper(x.shape, lambda: tf.shape(x))
    self.assertTrue(self.evaluate(is_scalar))

    x = tf.placeholder_with_default(input=[1], shape=None)
    is_scalar = normal._is_scalar_helper(x.shape, lambda: tf.shape(x))
    self.assertFalse(self.evaluate(is_scalar))

  def _GetFakeDistribution(self):
    class FakeDistribution(tfd.Distribution):
      """Fake Distribution for testing _set_sample_static_shape."""

      def __init__(self, batch_shape=None, event_shape=None):
        self._static_batch_shape = tf.TensorShape(batch_shape)
        self._static_event_shape = tf.TensorShape(event_shape)
        super(FakeDistribution, self).__init__(
            dtype=tf.float32,
            reparameterization_type=tfd.NOT_REPARAMETERIZED,
            validate_args=True,
            allow_nan_stats=True,
            name="DummyDistribution")

      def _batch_shape(self):
        return self._static_batch_shape

      def _event_shape(self):
        return self._static_event_shape

    return FakeDistribution

  def testSampleShapeHints(self):
    # In eager mode, all shapes are known, so these tests do not need to
    # execute.
    if tf.executing_eagerly():
      return

    fake_distribution = self._GetFakeDistribution()

    # Make a new session since we're playing with static shapes. [And below.]
    x = tf.placeholder_with_default(
        input=np.ones((6, 7, 2, 3, 5), dtype=np.float32), shape=None)
    dist = fake_distribution(batch_shape=[2, 3], event_shape=[5])
    sample_shape = tf.convert_to_tensor([6, 7], dtype=tf.int32)
    y = dist._set_sample_static_shape(x, sample_shape)
    # We use as_list since TensorShape comparison does not work correctly for
    # unknown values, ie, Dimension(None).
    self.assertAllEqual([6, 7, 2, 3, 5], y.shape.as_list())

    x = tf.placeholder_with_default(
        input=np.ones((6, 7, 2, 3, 5), dtype=np.float32), shape=None)
    dist = fake_distribution(batch_shape=[None, 3], event_shape=[5])
    sample_shape = tf.convert_to_tensor([6, 7], dtype=tf.int32)
    y = dist._set_sample_static_shape(x, sample_shape)
    self.assertAllEqual([6, 7, None, 3, 5], y.shape.as_list())

    x = tf.placeholder_with_default(
        input=np.ones((6, 7, 2, 3, 5), dtype=np.float32), shape=None)
    dist = fake_distribution(batch_shape=[None, 3], event_shape=[None])
    sample_shape = tf.convert_to_tensor([6, 7], dtype=tf.int32)
    y = dist._set_sample_static_shape(x, sample_shape)
    self.assertAllEqual([6, 7, None, 3, None], y.shape.as_list())

    x = tf.placeholder_with_default(
        input=np.ones((6, 7, 2, 3, 5), dtype=np.float32), shape=None)
    dist = fake_distribution(batch_shape=None, event_shape=None)
    sample_shape = tf.convert_to_tensor([6, 7], dtype=tf.int32)
    y = dist._set_sample_static_shape(x, sample_shape)
    self.assertTrue(y.shape.ndims is None)

    x = tf.placeholder_with_default(
        input=np.ones((6, 7, 2, 3, 5), dtype=np.float32), shape=None)
    dist = fake_distribution(batch_shape=[None, 3], event_shape=None)
    # There's no notion of partially known shapes in eager mode, so exit
    # early.
    sample_shape = tf.convert_to_tensor([6, 7], dtype=tf.int32)
    y = dist._set_sample_static_shape(x, sample_shape)
    self.assertTrue(y.shape.ndims is None)

  def testNameScopeWorksCorrectly(self):
    x = tfd.Normal(loc=0., scale=1., name="x")
    x_duplicate = tfd.Normal(loc=0., scale=1., name="x")
    with tf.name_scope("y") as name:
      y = tfd.Bernoulli(logits=0., name=name)
    x_sample = x.sample(name="custom_sample")
    x_sample_duplicate = x.sample(name="custom_sample")
    x_log_prob = x.log_prob(0., name="custom_log_prob")
    x_duplicate_sample = x_duplicate.sample(name="custom_sample")

    self.assertEqual(x.name, "x/")
    self.assertEqual(y.name, "y/")

    # There's no notion of graph, hence the same name will be reused.
    # Tensors also do not have names in eager mode, so exit early.
    if tf.executing_eagerly():
      return
    self.assertTrue(x_sample.name.startswith("x/custom_sample"))
    self.assertTrue(x_log_prob.name.startswith("x/custom_log_prob"))

    self.assertEqual(x_duplicate.name, "x_1/")
    self.assertTrue(x_duplicate_sample.name.startswith(
        "x_1/custom_sample"))
    self.assertTrue(x_sample_duplicate.name.startswith("x/custom_sample_1"))

  def testStrWorksCorrectlyScalar(self):
    # Usually we'd write np.float(X) here, but a recent Eager bug would
    # erroneously coerce the value to float32 anyway. We therefore use constants
    # here, until the bug is resolved in TensorFlow 1.12.
    normal = tfd.Normal(loc=tf.constant(0, tf.float16),
                        scale=tf.constant(1, tf.float16))
    self.assertEqual(
        str(normal),
        "tfp.distributions.Normal("
        "\"Normal/\", "
        "batch_shape=(), "
        "event_shape=(), "
        "dtype=float16)")

    chi2 = tfd.Chi2(df=np.float32([1., 2.]), name="silly")
    self.assertEqual(
        str(chi2),
        "tfp.distributions.Chi2("
        "\"silly/\", "  # What a silly name that is!
        "batch_shape=(2,), "
        "event_shape=(), "
        "dtype=float32)")

    # There's no notion of partially known shapes in eager mode, so exit
    # early.
    if tf.executing_eagerly():
      return

    exp = tfd.Exponential(rate=tf.placeholder_with_default(
        input=1., shape=None))
    self.assertEqual(
        str(exp),
        "tfp.distributions.Exponential(\"Exponential/\", "
        # No batch shape.
        "event_shape=(), "
        "dtype=float32)")

  def testStrWorksCorrectlyMultivariate(self):
    mvn_static = tfd.MultivariateNormalDiag(
        loc=np.zeros([2, 2]), name="MVN")
    self.assertEqual(
        str(mvn_static),
        "tfp.distributions.MultivariateNormalDiag("
        "\"MVN/\", "
        "batch_shape=(2,), "
        "event_shape=(2,), "
        "dtype=float64)")

    # There's no notion of partially known shapes in eager mode, so exit
    # early.
    if tf.executing_eagerly():
      return

    mvn_dynamic = tfd.MultivariateNormalDiag(
        loc=tf.placeholder_with_default(
            input=np.ones((3, 3), dtype=np.float32), shape=[None, 3]),
        name="MVN2")
    self.assertEqual(
        str(mvn_dynamic),
        "tfp.distributions.MultivariateNormalDiag("
        "\"MVN2/\", "
        "batch_shape=(?,), "  # Partially known.
        "event_shape=(3,), "
        "dtype=float32)")

  def testReprWorksCorrectlyScalar(self):
    # Usually we'd write np.float(X) here, but a recent Eager bug would
    # erroneously coerce the value to float32 anyway. We therefore use constants
    # here, until the bug is resolved in TensorFlow 1.12.
    normal = tfd.Normal(loc=tf.constant(0, tf.float16),
                        scale=tf.constant(1, tf.float16))
    self.assertEqual(
        repr(normal),
        "<tfp.distributions.Normal"
        " 'Normal/'"
        " batch_shape=()"
        " event_shape=()"
        " dtype=float16>")

    chi2 = tfd.Chi2(df=np.float32([1., 2.]), name="silly")
    self.assertEqual(
        repr(chi2),
        "<tfp.distributions.Chi2"
        " 'silly/'"  # What a silly name that is!
        " batch_shape=(2,)"
        " event_shape=()"
        " dtype=float32>")

    # There's no notion of partially known shapes in eager mode, so exit
    # early.
    if tf.executing_eagerly():
      return

    exp = tfd.Exponential(rate=tf.placeholder_with_default(
        input=1., shape=None))
    self.assertEqual(
        repr(exp),
        "<tfp.distributions.Exponential"
        " 'Exponential/'"
        " batch_shape=<unknown>"
        " event_shape=()"
        " dtype=float32>")

  def testReprWorksCorrectlyMultivariate(self):
    mvn_static = tfd.MultivariateNormalDiag(
        loc=np.zeros([2, 2]), name="MVN")
    self.assertEqual(
        repr(mvn_static),
        "<tfp.distributions.MultivariateNormalDiag"
        " 'MVN/'"
        " batch_shape=(2,)"
        " event_shape=(2,)"
        " dtype=float64>")

    # There's no notion of partially known shapes in eager mode, so exit
    # early.
    if tf.executing_eagerly():
      return

    mvn_dynamic = tfd.MultivariateNormalDiag(
        loc=tf.placeholder_with_default(
            input=np.ones((3, 3), dtype=np.float32), shape=[None, 3]),
        name="MVN2")
    self.assertEqual(
        repr(mvn_dynamic),
        "<tfp.distributions.MultivariateNormalDiag"
        " 'MVN2/'"
        " batch_shape=(?,)"  # Partially known.
        " event_shape=(3,)"
        " dtype=float32>")

  def testUnimplemtnedProbAndLogProbExceptions(self):
    class TerribleDistribution(tfd.Distribution):

      def __init__(self):
        super(TerribleDistribution, self).__init__(
            dtype=tf.float32,
            reparameterization_type=tfd.NOT_REPARAMETERIZED,
            validate_args=False,
            allow_nan_stats=False)

    terrible_distribution = TerribleDistribution()
    with self.assertRaisesRegexp(
        NotImplementedError, "prob is not implemented"):
      terrible_distribution.prob(1.)
    with self.assertRaisesRegexp(
        NotImplementedError, "log_prob is not implemented"):
      terrible_distribution.log_prob(1.)
    with self.assertRaisesRegexp(
        NotImplementedError, "cdf is not implemented"):
      terrible_distribution.cdf(1.)
    with self.assertRaisesRegexp(
        NotImplementedError, "log_cdf is not implemented"):
      terrible_distribution.log_cdf(1.)


if __name__ == "__main__":
  tf.test.main()
