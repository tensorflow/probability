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
"""Tests for tensorflow_probability.python.experimental.distributions.mvn_precision_factor_linop."""

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf

import tensorflow_probability as tfp
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import test_combinations
from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions
tfd_e = tfp.experimental.distributions


@test_util.test_all_tf_execution_regimes
class MVNPrecisionFactorLinOpTest(test_util.TestCase):

  def _random_constant_spd_linop(
      self,
      event_size,
      batch_shape=(),
      conditioning=1.2,
      dtype=np.float32,
  ):
    """Randomly generate a constant SPD LinearOperator."""
    # The larger conditioning is, the better posed the matrix is.
    # With conditioning = 1, it will be on the edge of singular, and likely
    # numerically singular if event_size is large enough.
    # Conditioning on the small side is best, since then the matrix is not so
    # diagonally dominant, and we therefore test use of transpositions better.
    assert conditioning >= 1

    scale_wishart = tfd.WishartLinearOperator(
        df=dtype(conditioning * event_size),
        scale=tf.linalg.LinearOperatorIdentity(event_size, dtype=dtype),
        input_output_cholesky=False,
    )
    # Make sure to evaluate here. This ensures that the linear operator is a
    # constant rather than a random operator.
    matrix = self.evaluate(
        scale_wishart.sample(batch_shape, seed=test_util.test_seed()))
    return tf.linalg.LinearOperatorFullMatrix(
        matrix, is_positive_definite=True, is_self_adjoint=True)

  @test_combinations.generate(
      test_combinations.combine(
          use_loc=[True, False],
          use_precision=[True, False],
          event_size=[3],
          batch_shape=[(), (2,)],
          n_samples=[5000],
          dtype=[np.float32, np.float64],
      ),
  )
  def test_log_prob_and_sample(
      self,
      use_loc,
      use_precision,
      event_size,
      batch_shape,
      dtype,
      n_samples,
  ):
    cov = self._random_constant_spd_linop(
        event_size, batch_shape=batch_shape, dtype=dtype)
    precision = cov.inverse()
    precision_factor = precision.cholesky()

    # Make sure to evaluate here, else you'll have a random loc vector!
    if use_loc:
      loc = self.evaluate(
          tf.random.normal(
              batch_shape + (event_size,),
              dtype=dtype,
              seed=test_util.test_seed()))
    else:
      loc = None

    mvn_scale = tfd.MultivariateNormalTriL(
        loc=loc, scale_tril=cov.cholesky().to_dense())

    mvn_precision = tfd_e.MultivariateNormalPrecisionFactorLinearOperator(
        loc=loc,
        precision_factor=precision_factor,
        precision=precision if use_precision else None,
    )

    point = tf.random.normal(
        batch_shape + (event_size,), dtype=dtype, seed=test_util.test_seed())
    mvn_scale_log_prob, mvn_precision_log_prob = self.evaluate(
        [mvn_scale.log_prob(point),
         mvn_precision.log_prob(point)])
    self.assertAllClose(
        mvn_scale_log_prob, mvn_precision_log_prob, atol=5e-4, rtol=5e-4)

    batch_point = tf.random.normal(
        (2,) + batch_shape + (event_size,),
        dtype=dtype,
        seed=test_util.test_seed())
    mvn_scale_log_prob, mvn_precision_log_prob = self.evaluate(
        [mvn_scale.log_prob(batch_point),
         mvn_precision.log_prob(batch_point)])
    self.assertAllClose(
        mvn_scale_log_prob, mvn_precision_log_prob, atol=5e-4, rtol=5e-4)

    samples = mvn_precision.sample(n_samples, seed=test_util.test_seed())
    arrs = self.evaluate({
        'stddev': tf.sqrt(cov.diag_part()),
        'var': cov.diag_part(),
        'cov': cov.to_dense(),
        'sample_mean': tf.reduce_mean(samples, axis=0),
        'sample_var': tfp.stats.variance(samples, sample_axis=0),
        'sample_cov': tfp.stats.covariance(samples, sample_axis=0),
    })

    self.assertAllClose(
        arrs['sample_mean'],
        loc if loc is not None else np.zeros_like(arrs['cov'][..., 0]),
        atol=5 * np.max(arrs['stddev']) / np.sqrt(n_samples))
    self.assertAllClose(
        arrs['sample_var'],
        arrs['var'],
        atol=5 * np.sqrt(2) * np.max(arrs['var']) / np.sqrt(n_samples))
    self.assertAllClose(
        arrs['sample_cov'],
        arrs['cov'],
        atol=5 * np.sqrt(2) * np.max(arrs['var']) / np.sqrt(n_samples))

  def test_dynamic_shape(self):
    x = tf.Variable(ps.ones([7, 3]), shape=[7, None])
    self.evaluate(x.initializer)

    # Check that the shape is actually `None`.
    if not tf.executing_eagerly():
      last_shape = x.shape[-1]
      if last_shape is not None:  # This is a `tf.Dimension` in tf1.
        last_shape = last_shape.value
      self.assertIsNone(last_shape)
    dynamic_dist = tfd_e.MultivariateNormalPrecisionFactorLinearOperator(
        precision_factor=tf.linalg.LinearOperatorDiag(tf.ones_like(x)))
    static_dist = tfd_e.MultivariateNormalPrecisionFactorLinearOperator(
        precision_factor=tf.linalg.LinearOperatorDiag(tf.ones([7, 3])))
    in_ = tf.zeros([7, 3])
    self.assertAllClose(self.evaluate(dynamic_dist.log_prob(in_)),
                        static_dist.log_prob(in_))

  @test_combinations.generate(
      test_combinations.combine(
          batch_shape=[(), (2,)],
          dtype=[np.float32, np.float64],
      ),
  )
  def test_mean_and_mode(self, batch_shape, dtype):
    event_size = 3
    cov = self._random_constant_spd_linop(
        event_size, batch_shape=batch_shape, dtype=dtype)
    precision_factor = cov.inverse().cholesky()

    # Make sure to evaluate here, else you'll have a random loc vector!
    loc = self.evaluate(
        tf.random.normal(
            batch_shape + (event_size,),
            dtype=dtype,
            seed=test_util.test_seed()))

    mvn_precision = tfd_e.MultivariateNormalPrecisionFactorLinearOperator(
        loc=loc,
        precision_factor=precision_factor)
    self.assertAllClose(mvn_precision.mean(), loc)
    self.assertAllClose(mvn_precision.mode(), loc)

  @test_combinations.generate(
      test_combinations.combine(
          batch_shape=[(), (2,)],
          use_precision=[True, False],
          dtype=[np.float32, np.float64],
      ),
  )
  def test_cov_var_stddev(self, batch_shape, use_precision, dtype):
    event_size = 3
    cov = self._random_constant_spd_linop(
        event_size, batch_shape=batch_shape, dtype=dtype)
    precision = cov.inverse()
    precision_factor = precision.cholesky()

    # Make sure to evaluate here, else you'll have a random loc vector!
    loc = self.evaluate(
        tf.random.normal(
            batch_shape + (event_size,),
            dtype=dtype,
            seed=test_util.test_seed()))

    mvn_precision = tfd_e.MultivariateNormalPrecisionFactorLinearOperator(
        loc=loc,
        precision_factor=precision_factor,
        precision=precision if use_precision else None)
    self.assertAllClose(mvn_precision.covariance(), cov.to_dense(), atol=1e-4)
    self.assertAllClose(mvn_precision.variance(), cov.diag_part(), atol=1e-4)
    self.assertAllClose(mvn_precision.stddev(), tf.sqrt(cov.diag_part()),
                        atol=1e-5)


if __name__ == '__main__':
  test_util.main()
