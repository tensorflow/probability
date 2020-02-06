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
"""Tests for sample_halton_sequence.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import monte_carlo
from tensorflow_probability.python.internal import test_util


def _set_seed(seed):
  """Helper which uses graph seed if using TFE."""
  # TODO(b/68017812): Deprecate once TFE supports seed.
  if tf.executing_eagerly():
    tf.random.set_seed(seed)
    return None
  return seed


@test_util.test_all_tf_execution_regimes
class HaltonSequenceTest(test_util.TestCase):

  def test_known_values_small_bases(self):
    # The first five elements of the non-randomized Halton sequence
    # with base 2 and 3.
    expected = np.array([[1. / 2, 1. / 3],
                         [1. / 4, 2. / 3],
                         [3. / 4, 1. / 9],
                         [1. / 8, 4. / 9],
                         [5. / 8, 7. / 9]], dtype=np.float32)
    sample = tfp.mcmc.sample_halton_sequence(2, num_results=5, randomized=False)
    self.assertAllClose(expected, self.evaluate(sample), rtol=1e-6)

  def test_dynamic_num_samples(self):
    """Tests that num_samples argument supports Tensors."""
    # The first five elements of the non-randomized Halton sequence
    # with base 2 and 3.
    expected = np.array([[1. / 2, 1. / 3],
                         [1. / 4, 2. / 3],
                         [3. / 4, 1. / 9],
                         [1. / 8, 4. / 9],
                         [5. / 8, 7. / 9]], dtype=np.float32)
    sample = tfp.mcmc.sample_halton_sequence(
        2, num_results=tf.constant(5), randomized=False)
    self.assertAllClose(expected, self.evaluate(sample), rtol=1e-6)

  def test_sequence_indices(self):
    """Tests access of sequence elements by index."""
    dim = 5
    indices = tf.range(10, dtype=tf.int32)
    sample_direct = tfp.mcmc.sample_halton_sequence(
        dim, num_results=10, randomized=False)
    sample_from_indices = tfp.mcmc.sample_halton_sequence(
        dim, sequence_indices=indices, randomized=False)
    self.assertAllClose(
        self.evaluate(sample_direct), self.evaluate(sample_from_indices),
        rtol=1e-6)

  def test_dtypes_works_correctly(self):
    """Tests that all supported dtypes work without error."""
    dim = 3
    sample_float32 = tfp.mcmc.sample_halton_sequence(
        dim, num_results=10, dtype=tf.float32, seed=11)
    sample_float64 = tfp.mcmc.sample_halton_sequence(
        dim, num_results=10, dtype=tf.float64, seed=21)
    self.assertEqual(self.evaluate(sample_float32).dtype, np.float32)
    self.assertEqual(self.evaluate(sample_float64).dtype, np.float64)

  def test_normal_integral_mean_and_var_correctly_estimated(self):
    n = int(1000)
    # This test is almost identical to the similarly named test in
    # monte_carlo_test.py. The only difference is that we use the Halton
    # samples instead of the random samples to evaluate the expectations.
    # MC with pseudo random numbers converges at the rate of 1/ Sqrt(N)
    # (N=number of samples). For QMC in low dimensions, the expected convergence
    # rate is ~ 1/N. Hence we should only need 1e3 samples as compared to the
    # 1e6 samples used in the pseudo-random monte carlo.
    mu_p = tf.constant([-1., 1.], dtype=tf.float64)
    mu_q = tf.constant([0., 0.], dtype=tf.float64)
    sigma_p = tf.constant([0.5, 0.5], dtype=tf.float64)
    sigma_q = tf.constant([1., 1.], dtype=tf.float64)
    p = tfd.Normal(loc=mu_p, scale=sigma_p)
    q = tfd.Normal(loc=mu_q, scale=sigma_q)

    cdf_sample = tfp.mcmc.sample_halton_sequence(
        2, num_results=n, dtype=tf.float64, seed=1729)
    q_sample = q.quantile(cdf_sample)

    # Compute E_p[X].
    e_x = monte_carlo.expectation_importance_sampler(
        f=lambda x: x, log_p=p.log_prob, sampling_dist_q=q, z=q_sample, seed=42)

    # Compute E_p[X^2].
    e_x2 = monte_carlo.expectation_importance_sampler(
        f=tf.square, log_p=p.log_prob, sampling_dist_q=q, z=q_sample, seed=1412)

    stddev = tf.sqrt(e_x2 - tf.square(e_x))
    # Keep the tolerance levels the same as in monte_carlo_test.py.
    self.assertEqual(p.batch_shape, e_x.shape)
    self.assertAllClose(self.evaluate(p.mean()), self.evaluate(e_x), rtol=0.01)
    self.assertAllClose(
        self.evaluate(p.stddev()), self.evaluate(stddev), rtol=0.02)

  def test_docstring_example(self):
    # Produce the first 1000 members of the Halton sequence in 3 dimensions.
    num_results = 1000
    dim = 3
    sample = tfp.mcmc.sample_halton_sequence(
        dim, num_results=num_results, randomized=False)

    # Evaluate the integral of x_1 * x_2^2 * x_3^3  over the three dimensional
    # hypercube.
    powers = tf.range(1., limit=dim + 1)
    integral = tf.reduce_mean(tf.reduce_prod(sample**powers, axis=-1))
    true_value = 1. / tf.reduce_prod(powers + 1.)

    # Produces a relative absolute error of 1.7%.
    self.assertAllClose(
        self.evaluate(integral), self.evaluate(true_value), rtol=0.02)

    # Now skip the first 1000 samples and recompute the integral with the next
    # thousand samples. The sequence_indices argument can be used to do this.

    sequence_indices = tf.range(start=1000, limit=1000 + num_results,
                                dtype=tf.int32)
    sample_leaped = tfp.mcmc.sample_halton_sequence(
        dim, sequence_indices=sequence_indices, randomized=False)

    integral_leaped = tf.reduce_mean(
        tf.reduce_prod(sample_leaped**powers, axis=-1))
    self.assertAllClose(
        self.evaluate(integral_leaped), self.evaluate(true_value), rtol=0.05)

  def test_randomized_qmc_basic(self):
    """Tests the randomization of the Halton sequences."""
    # This test is identical to the example given in Owen (2017), Figure 5.

    dim = 20
    num_results = 2000
    replicas = 50

    samples = tfp.mcmc.sample_halton_sequence(
        dim, num_results=replicas * num_results,
        seed=test_util.test_seed_stream())
    samples = tf.reshape(samples, [replicas, num_results, dim])
    values = self.evaluate(
        tf.reduce_mean(tf.reduce_sum(samples, axis=-1)**2, axis=-1))
    self.assertAllClose(values.mean(), 101.6667,
                        atol=values.std() / np.sqrt(replicas))

  def test_partial_sum_func_qmc(self):
    """Tests the QMC evaluation of (x_j + x_{j+1} ...+x_{n})^2.

    A good test of QMC is provided by the function:

      f(x_1,..x_n, x_{n+1}, ..., x_{n+m}) = (x_{n+1} + ... x_{n+m} - m / 2)^2

    with the coordinates taking values in the unit interval. The mean and
    variance of this function (with the uniform distribution over the
    unit-hypercube) is exactly calculable:

      <f> = m / 12, Var(f) = m (5m - 3) / 360

    The purpose of the "shift" (if n > 0) in the coordinate dependence of the
    function is to provide a test for Halton sequence which exhibit more
    dependence in the higher axes.

    This test confirms that the mean squared error of RQMC estimation falls
    as O(N^(2-e)) for any e>0.
    """

    n, m = 10, 10
    dim = n + m
    num_results_lo, num_results_hi = 1000, 10000
    replica = 100
    true_mean = m / 12.

    def func_estimate(x):
      return tf.reduce_mean(
          tf.math.squared_difference(
              tf.reduce_sum(x[..., -m:], axis=-1),
              m / 2.),
          axis=-1)

    sample_lo = tfp.mcmc.sample_halton_sequence(
        dim, num_results=replica * num_results_lo,
        seed=test_util.test_seed_stream())
    sample_hi = tfp.mcmc.sample_halton_sequence(
        dim, num_results=replica * num_results_hi,
        seed=test_util.test_seed_stream())

    sample_lo = tf.reshape(sample_lo, [replica, -1, dim])
    sample_hi = tf.reshape(sample_hi, [replica, -1, dim])

    f_lo = self.evaluate(func_estimate(sample_lo))
    f_hi = self.evaluate(func_estimate(sample_hi))
    var_lo = np.mean((f_lo - true_mean) ** 2, axis=0)
    var_hi = np.mean((f_hi - true_mean) ** 2, axis=0)

    # Expect that the variance scales as N^(-2-epsilon) (see section 3 of
    # https://arxiv.org/pdf/1706.02808.pdf). Thus, since
    # num_results_hi = 10 * num_results_lo, we expect to have
    #
    #   var_hi / var_lo ~= k / 100
    #
    # with k a fudge factor accounting for the residual N dependence of the QMC
    # error and the sampling error. Here we take log(k) = 1.5.
    log_rel_err = np.log(100 * var_hi / var_lo)
    self.assertAllLess(np.abs(log_rel_err), 1.5)

  def test_seed_implies_deterministic_results(self):
    dim = 20
    num_results = 100
    sample1 = tfp.mcmc.sample_halton_sequence(
        dim, num_results=num_results, seed=_set_seed(1925))
    sample2 = tfp.mcmc.sample_halton_sequence(
        dim, num_results=num_results, seed=_set_seed(1925))
    [sample1_, sample2_] = self.evaluate([sample1, sample2])
    self.assertAllClose(sample1_, sample2_, atol=0., rtol=1e-6)


if __name__ == "__main__":
  tf.test.main()
