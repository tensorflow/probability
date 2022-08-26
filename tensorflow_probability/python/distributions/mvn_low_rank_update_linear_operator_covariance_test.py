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
"""Tests for MultivariateNormalLowRankUpdateLinearOperatorCovariance."""

import collections

# Dependency imports

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import stats as tfps
from tensorflow_probability.python.distributions import mvn_low_rank_update_linear_operator_covariance
from tensorflow_probability.python.distributions import mvn_tril
from tensorflow_probability.python.internal import test_util

MultivariateNormalLowRankUpdateLinearOperatorCovariance = (
    mvn_low_rank_update_linear_operator_covariance
    .MultivariateNormalLowRankUpdateLinearOperatorCovariance)

JAX_MODE = False
NUMPY_MODE = False

# For holding a pair of MVNs, probably to compare them since they should be the
# same, statistically.
MVNPair = collections.namedtuple(
    'MVNPair',
    [
        # MultivariateNormalLowRankUpdateLinearOperatorCovariance
        'low_rank_update',
        # MultivariateNormalTriL
        'tril',
    ],
)


class MVNLowRankUpdateCovarianceDynamicShapeTest(test_util.TestCase):

  def _construct_mvn_for_shape_tests(self, loc_shape, cov_diag_shape):
    """Construct an MVN with possibly partially defined shapes."""
    bcast_shape = tf.broadcast_static_shape(
        tf.TensorShape(loc_shape),
        tf.TensorShape(cov_diag_shape),
    )
    bcast_shape.assert_is_fully_defined()
    bcast_shape = tuple(bcast_shape.as_list())

    dtype = np.float32

    # Use a slice of the fully defined bcast_shape to make numpy parameters.
    # This ensures the parameters are (i) fully defined, (ii) the correct rank.
    loc_ = np.ones(bcast_shape[-len(loc_shape):], dtype=dtype)
    covariance_diag_factor_ = np.ones(
        bcast_shape[-len(cov_diag_shape):], dtype=dtype)
    covariance_perturb_factor_ = np.ones(
        bcast_shape[-len(cov_diag_shape):] + (1,), dtype=dtype)

    loc = tf1.placeholder_with_default(loc_, shape=loc_shape)
    covariance_diag_factor = tf1.placeholder_with_default(
        covariance_diag_factor_, shape=cov_diag_shape)
    covariance_perturb_factor = tf1.placeholder_with_default(
        covariance_perturb_factor_, shape=cov_diag_shape + (1,))

    cov_operator = tf.linalg.LinearOperatorLowRankUpdate(
        base_operator=tf.linalg.LinearOperatorDiag(
            covariance_diag_factor, is_positive_definite=True),
        u=covariance_perturb_factor,
    )
    return MultivariateNormalLowRankUpdateLinearOperatorCovariance(
        loc=loc, cov_operator=cov_operator, validate_args=True)

  def testShapeInfoPreservedFullyStatic(self):
    with self.subTest('all static shape known - cov has more batch dims'):
      mvn = self._construct_mvn_for_shape_tests(
          loc_shape=(3, 4), cov_diag_shape=(2, 3, 4))
      self.assertAllEqual((2, 3), mvn.batch_shape)
      self.assertAllEqual((2, 3), mvn.batch_shape_tensor())
      self.assertAllEqual((4,), mvn.event_shape)
      self.assertAllEqual((4,), mvn.event_shape_tensor())

    with self.subTest('all static shape known - loc has more batch dims'):
      mvn = self._construct_mvn_for_shape_tests(
          loc_shape=(2, 3, 4), cov_diag_shape=(3, 4))
      self.assertAllEqual((2, 3), mvn.batch_shape)
      self.assertAllEqual((2, 3), mvn.batch_shape_tensor())
      self.assertAllEqual((4,), mvn.event_shape)
      self.assertAllEqual((4,), mvn.event_shape_tensor())

  def testShapeInfoPreservedMixedStaticStillFullyDefined(self):
    with self.subTest('mixed static shape known - cov has more batch dims'):
      mvn = self._construct_mvn_for_shape_tests(
          loc_shape=(3, None), cov_diag_shape=(2, None, 4))
      self.assertAllEqual((2, 3), mvn.batch_shape)
      self.assertAllEqual((2, 3), mvn.batch_shape_tensor())
      self.assertAllEqual((4,), mvn.event_shape)
      self.assertAllEqual((4,), mvn.event_shape_tensor())

    with self.subTest('mixed static shape known - loc has more batch dims'):
      mvn = self._construct_mvn_for_shape_tests(
          loc_shape=(2, None, 4), cov_diag_shape=(3, None))
      self.assertAllEqual((2, 3), mvn.batch_shape)
      self.assertAllEqual((2, 3), mvn.batch_shape_tensor())
      self.assertAllEqual((4,), mvn.event_shape)
      self.assertAllEqual((4,), mvn.event_shape_tensor())


if JAX_MODE or NUMPY_MODE:
  del MVNLowRankUpdateCovarianceDynamicShapeTest


@test_util.test_all_tf_execution_regimes
class MVNLowRankUpdateCovarianceJITShapes(test_util.TestCase):

  def _get_mvn_shape_tensors(self, loc_shape, cov_diag_shape):
    """Construct an MVN with fully defined shapes."""

    dtype = np.float32

    # Use a slice of the fully defined bcast_shape to make numpy parameters.
    # This ensures the parameters are (i) fully defined, (ii) the correct rank.
    def _make_mvn():
      loc = tf.ones(loc_shape, dtype=dtype)
      covariance_diag_factor = tf.ones(cov_diag_shape, dtype=dtype)
      covariance_perturb_factor = tf.ones(cov_diag_shape + (1,), dtype=dtype)

      cov_operator = tf.linalg.LinearOperatorLowRankUpdate(
          base_operator=tf.linalg.LinearOperatorDiag(
              covariance_diag_factor, is_positive_definite=True),
          u=covariance_perturb_factor,
      )
      return MultivariateNormalLowRankUpdateLinearOperatorCovariance(
          loc=loc, cov_operator=cov_operator, validate_args=True)

    @tf.function(jit_compile=True)
    def _get_shapes():
      mvn = _make_mvn()
      return {
          'event_shape': tf.convert_to_tensor(mvn.event_shape),
          'batch_shape': tf.convert_to_tensor(mvn.batch_shape),
      }

    return _get_shapes()

  def testShapeInfoPreserved(self):
    self.skip_if_no_xla()
    with self.subTest('all static shape known - cov has more batch dims'):
      shapes = self._get_mvn_shape_tensors(
          loc_shape=(3, 4), cov_diag_shape=(2, 3, 4))
      self.assertAllEqual((2, 3), shapes['batch_shape'])
      self.assertAllEqual((4,), shapes['event_shape'])

    with self.subTest('all static shape known - loc has more batch dims'):
      shapes = self._get_mvn_shape_tensors(
          loc_shape=(2, 3, 4), cov_diag_shape=(3, 4))
      self.assertAllEqual((2, 3), shapes['batch_shape'])
      self.assertAllEqual((4,), shapes['event_shape'])


@test_util.test_all_tf_execution_regimes
class MultivariateNormalLowRankUpdateLinearOperatorCovarianceTest(
    test_util.TestCase):

  def _construct_loc(self, loc_shape, dtype=np.float32):
    if loc_shape is None:
      return None
    return np.random.normal(size=loc_shape).astype(dtype)

  def _construct_cov_operator(self,
                              diag_shape,
                              update_shape,
                              include_diag_update=None,
                              dtype=np.float32):
    """Construct LinearOperatorLowRankUpdate covariance with constant params."""
    base_operator = tf.linalg.LinearOperatorDiag(
        np.random.uniform(low=1., high=2., size=diag_shape).astype(dtype),
        is_positive_definite=True)
    u = np.random.normal(size=update_shape).astype(dtype)
    if include_diag_update:
      diag_update = np.random.uniform(
          size=update_shape[:-2] + update_shape[-1:],
          low=1.,
          high=2.,
      ).astype(dtype)
    else:
      diag_update = None
    return tf.linalg.LinearOperatorLowRankUpdate(
        base_operator, u=u, diag_update=diag_update)

  def _mvn_pair(self, loc, cov_operator):
    """Construct a pair of MVNs."""
    tril = mvn_tril.MultivariateNormalTriL(
        loc=loc, scale_tril=cov_operator.cholesky().to_dense())
    low_rank_update = (
        MultivariateNormalLowRankUpdateLinearOperatorCovariance(
            loc=loc, cov_operator=cov_operator, validate_args=True))
    return MVNPair(low_rank_update=low_rank_update, tril=tril)

  @parameterized.named_parameters([
      dict(
          testcase_name='loc3_diag3_update31_diagupFalse_float32',
          loc_shape=(3,),
          diag_shape=(3,),
          update_shape=(3, 1),
          include_diag_update=False,
          dtype=np.float32),
      dict(
          testcase_name='loc23_diag3_update31_diagupFalse_float64',
          loc_shape=(2, 3),
          diag_shape=(3,),
          update_shape=(3, 1),
          include_diag_update=False,
          dtype=np.float64),
      dict(
          testcase_name='locNone_diag3_update31_diagupTrue_float32',
          loc_shape=None,
          diag_shape=(3,),
          update_shape=(3, 1),
          include_diag_update=True,
          dtype=np.float32),
      dict(
          testcase_name='loc23_diag3_update31_diagupTrue_float32',
          loc_shape=(2, 3),
          diag_shape=(3,),
          update_shape=(3, 1),
          include_diag_update=True,
          dtype=np.float32),
  ])
  def testVersusMVNTriL(
      self,
      loc_shape,
      diag_shape,
      update_shape,
      include_diag_update,
      dtype,
  ):

    mvn_pair = self._mvn_pair(
        loc=self._construct_loc(loc_shape, dtype=dtype),
        cov_operator=self._construct_cov_operator(
            diag_shape, update_shape, include_diag_update=False, dtype=dtype))
    tril = mvn_pair.tril
    low_rank_update = mvn_pair.low_rank_update

    with self.subTest('Shapes are equal'):
      self.assertAllEqual(tril.batch_shape, low_rank_update.batch_shape)
      self.assertAllEqual(tril.event_shape, low_rank_update.event_shape)
      self.assertAllEqual(*self.evaluate(
          [tril.batch_shape_tensor(),
           low_rank_update.batch_shape_tensor()]))
      self.assertAllEqual(*self.evaluate(
          [tril.event_shape_tensor(),
           low_rank_update.event_shape_tensor()]))

    with self.subTest('Statistics are almost equal'):
      self.assertAllClose(*self.evaluate([tril.mode(), low_rank_update.mode()]))
      self.assertAllClose(*self.evaluate([tril.mean(), low_rank_update.mean()]))
      self.assertAllClose(*self.evaluate(
          [tril.stddev(), low_rank_update.stddev()]))
      self.assertAllClose(*self.evaluate(
          [tril.covariance(), low_rank_update.covariance()]))
      self.assertAllClose(*self.evaluate(
          [tril.variance(), low_rank_update.variance()]))
      self.assertAllClose(*self.evaluate(
          [tril.entropy(), low_rank_update.entropy()]))

    with self.subTest('Samples are correct'):
      n = 10000
      samples = low_rank_update.sample(n, seed=test_util.test_seed())
      samples, sample_mean, sample_var, sample_cov = self.evaluate([
          samples,
          tf.reduce_mean(samples, axis=0),
          tfps.variance(samples, sample_axis=0),
          tfps.covariance(samples, sample_axis=0),
      ])

      ref_samples = tril.sample(n, seed=test_util.test_seed())
      self.assertAllEqual(ref_samples.shape, samples.shape)

      maxstddev = np.max(self.evaluate(low_rank_update.stddev()))
      self.assertAllMeansClose(
          samples,
          self.evaluate(low_rank_update.mean()),
          axis=0,
          atol=5 * maxstddev / np.sqrt(n))
      self.assertAllClose(
          sample_var,
          self.evaluate(low_rank_update.variance()),
          rtol=5 * maxstddev / np.sqrt(n))
      self.assertAllClose(
          sample_cov,
          self.evaluate(low_rank_update.covariance()),
          atol=10 * maxstddev**2 / np.sqrt(n))

    with self.subTest('prob(mean) is almost equal'):
      x = sample_mean
      self.assertAllClose(
          *self.evaluate([tril.prob(x), low_rank_update.prob(x)]), rtol=1e-5)
      self.assertAllClose(
          *self.evaluate([tril.log_prob(x),
                          low_rank_update.log_prob(x)]),
          rtol=1e-5)

    with self.subTest('prob(widely dispersed) is almost equal'):
      x = self.evaluate(ref_samples * 5)  # Widely dispersed samples.
      self.assertAllClose(
          *self.evaluate([tril.log_prob(x),
                          low_rank_update.log_prob(x)]),
          rtol=1e-5)

  def testRaisesIfCovarianceOperatorNotProvided(self):
    with self.assertRaisesRegex(ValueError, 'Missing.*cov_operator'):
      MultivariateNormalLowRankUpdateLinearOperatorCovariance(
          loc=[1., 1.], cov_operator=None, validate_args=True)

  @test_util.tf_tape_safety_test
  def testVariableLocation(self):
    loc = tf.Variable([1., 1.])
    cov_base_diag = tf.Variable([2., 2.])
    cov_operator = tf.linalg.LinearOperatorLowRankUpdate(
        base_operator=tf.linalg.LinearOperatorDiag(
            cov_base_diag, is_positive_definite=True),
        u=tf.ones((2, 2)),
    )
    d = MultivariateNormalLowRankUpdateLinearOperatorCovariance(
        loc, cov_operator, validate_args=True)
    self.evaluate([loc.initializer, cov_base_diag.initializer])

    with tf.GradientTape() as tape:
      lp = d.log_prob([0., 0.])
    self.assertIsNotNone(tape.gradient(lp, loc))

    with tf.GradientTape() as tape:
      lp = d.log_prob([0., 0.])
    self.assertIsNotNone(tape.gradient(lp, cov_base_diag))

  @test_util.numpy_disable_variable_test
  @test_util.jax_disable_variable_test
  def testVariableCovAssertions(self):
    # We test that changing the scale to be non-invertible raises an exception
    # when validate_args is True. This is really just testing the underlying
    # LinearOperator instance, but we include it to demonstrate that it
    # works as expected.
    base_operator_diag = tf.Variable([1., 1.])
    base_operator = tf.linalg.LinearOperatorDiag(
        base_operator_diag, is_positive_definite=True)
    cov_operator = tf.linalg.LinearOperatorLowRankUpdate(
        base_operator, u=tf.zeros((2, 2)))
    d = MultivariateNormalLowRankUpdateLinearOperatorCovariance(
        loc=None, cov_operator=cov_operator, validate_args=True)
    self.evaluate(base_operator_diag.initializer)
    with self.assertRaises(Exception):
      with tf.control_dependencies([base_operator_diag.assign([1., 0.])]):
        self.evaluate(d.sample(seed=test_util.test_seed()))

  def testAssertions(self):
    base_operator_diag = tf.constant([1., 0.])
    base_operator = tf.linalg.LinearOperatorDiag(
        base_operator_diag, is_positive_definite=True)
    cov_operator = tf.linalg.LinearOperatorLowRankUpdate(
        base_operator, u=tf.zeros((2, 2)))
    with self.assertRaises(Exception):
      dist = MultivariateNormalLowRankUpdateLinearOperatorCovariance(
          loc=None, cov_operator=cov_operator, validate_args=True)
      self.evaluate(dist.covariance())

  def testNonSPDHintsRaise(self):

    with self.subTest('base_operator must be positive definite'):
      base_operator = tf.linalg.LinearOperatorDiag(
          # It actually is PD, but the hint is set wrong
          [1., 1.],
          is_positive_definite=False)
      cov_operator = tf.linalg.LinearOperatorLowRankUpdate(
          base_operator, u=tf.zeros((2, 2)))
      with self.assertRaisesRegex(ValueError, 'must be positive'):
        MultivariateNormalLowRankUpdateLinearOperatorCovariance(
            cov_operator=cov_operator)

    with self.subTest('U must equal V'):
      base_operator = tf.linalg.LinearOperatorDiag([1., 1.],
                                                   is_positive_definite=True)
      cov_operator = tf.linalg.LinearOperatorLowRankUpdate(
          base_operator, u=tf.zeros((2, 2)), v=tf.ones((2, 2)))
      with self.assertRaisesRegex(ValueError, 'must be the same'):
        MultivariateNormalLowRankUpdateLinearOperatorCovariance(
            cov_operator=cov_operator)


if __name__ == '__main__':
  test_util.main()
