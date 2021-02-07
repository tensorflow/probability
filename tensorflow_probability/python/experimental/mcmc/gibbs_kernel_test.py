from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util
from gemlib.mcmc.gibbs_kernel import GibbsKernel


@test_util.test_all_tf_execution_regimes
class TestGibbsKernel(test_util.TestCase):
    def test_2d_mvn(self):
        """Sample from 2-variate MVN Distribution."""
        dtype = np.float32
        true_mean = dtype([1, 1])
        true_cov = dtype([[1, 0.5], [0.5, 1]])
        target = tfd.MultivariateNormalTriL(
            loc=true_mean, scale_tril=tf.linalg.cholesky(true_cov)
        )

        def logp(x1, x2):
            return target.log_prob([x1, x2])

        def kernel_make_fn(target_log_prob_fn, state):
            return tfp.mcmc.RandomWalkMetropolis(
                target_log_prob_fn=target_log_prob_fn
            )

        kernel_list = [(0, kernel_make_fn), (1, kernel_make_fn)]
        kernel = GibbsKernel(target_log_prob_fn=logp, kernel_list=kernel_list)
        samples = tfp.mcmc.sample_chain(
            num_results=2000,
            current_state=[dtype(1), dtype(1)],
            kernel=kernel,
            num_burnin_steps=500,
            trace_fn=None,
        )

        sample_mean = tf.math.reduce_mean(samples, axis=1)
        [sample_mean_] = self.evaluate([sample_mean])
        self.assertAllClose(sample_mean_, true_mean, atol=0.2, rtol=0.2)

        sample_cov = tfp.stats.covariance(tf.transpose(samples))
        sample_cov_ = self.evaluate(sample_cov)
        self.assertAllClose(sample_cov_, true_cov, atol=0.1, rtol=0.1)

    def test_float64(self):
        """Sample with dtype float64."""
        dtype = np.float64
        true_mean = dtype([1, 1])
        true_cov = dtype([[1, 0.5], [0.5, 1]])
        target = tfd.MultivariateNormalTriL(
            loc=true_mean, scale_tril=tf.linalg.cholesky(true_cov)
        )

        def logp(x1, x2):
            return target.log_prob([x1, x2])

        def kernel_make_fn(target_log_prob_fn, state):
            return tfp.mcmc.RandomWalkMetropolis(
                target_log_prob_fn=target_log_prob_fn
            )

        kernel_list = [(0, kernel_make_fn), (1, kernel_make_fn)]
        kernel = GibbsKernel(target_log_prob_fn=logp, kernel_list=kernel_list)
        samples = tfp.mcmc.sample_chain(
            num_results=20,
            current_state=[dtype(1), dtype(1)],
            kernel=kernel,
            trace_fn=None,
        )

    def test_bijector(self):
        """Employ bijector when sampling."""
        dtype = np.float32
        true_mean = dtype([1, 1])
        true_cov = dtype([[1, 0.5], [0.5, 1]])
        target = tfd.MultivariateNormalTriL(
            loc=true_mean, scale_tril=tf.linalg.cholesky(true_cov)
        )

        def logp(x1, x2):
            return target.log_prob([x1, x2])

        def kernel_make_fn(target_log_prob_fn, state):
            inner_kernel = tfp.mcmc.RandomWalkMetropolis(
                target_log_prob_fn=target_log_prob_fn
            )
            return tfp.mcmc.TransformedTransitionKernel(
                inner_kernel=inner_kernel, bijector=tfp.bijectors.Exp()
            )

        kernel_list = [(0, kernel_make_fn), (1, kernel_make_fn)]
        kernel = GibbsKernel(target_log_prob_fn=logp, kernel_list=kernel_list)
        samples = tfp.mcmc.sample_chain(
            num_results=20,
            current_state=[dtype(1), dtype(1)],
            kernel=kernel,
            trace_fn=None,
        )

    def test_gradient_based_sampler(self):
        """Make sure Gibbs kernel is compatible with gradient-based
        samplers"""
        dtype = np.float32
        true_mean = dtype([1, 1])
        true_cov = dtype([[1, 0.5], [0.5, 1]])
        target = tfd.MultivariateNormalTriL(
            loc=true_mean, scale_tril=tf.linalg.cholesky(true_cov)
        )

        def logp(x1, x2):
            return target.log_prob([x1, x2])

        def kernel_make_rwm_fn(target_log_prob_fn, state):
            inner_kernel = tfp.mcmc.RandomWalkMetropolis(
                target_log_prob_fn=target_log_prob_fn
            )
            return tfp.mcmc.TransformedTransitionKernel(
                inner_kernel=inner_kernel, bijector=tfp.bijectors.Exp()
            )

        def kernel_make_hmc_fn(target_log_prob_fn, state):
            inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=target_log_prob_fn,
                step_size=0.1,
                num_leapfrog_steps=3,
            )
            return tfp.mcmc.TransformedTransitionKernel(
                inner_kernel=inner_kernel, bijector=tfp.bijectors.Exp()
            )

        kernel_list = [(0, kernel_make_rwm_fn), (1, kernel_make_hmc_fn)]
        kernel = GibbsKernel(target_log_prob_fn=logp, kernel_list=kernel_list)
        samples = tfp.mcmc.sample_chain(
            num_results=20,
            current_state=[dtype(1), dtype(1)],
            kernel=kernel,
            trace_fn=None,
        )

    def test_is_calibrated(self):
        dtype = np.float32
        true_mean = dtype([1, 1])
        true_cov = dtype([[1, 0.5], [0.5, 1]])
        target = tfd.MultivariateNormalTriL(
            loc=true_mean, scale_tril=tf.linalg.cholesky(true_cov)
        )

        def logp(x1, x2):
            return target.log_prob([x1, x2])

        def kernel_make_fn(target_log_prob_fn, state):
            return tfp.mcmc.RandomWalkMetropolis(
                target_log_prob_fn=target_log_prob_fn
            )

        kernel_list = [(0, kernel_make_fn), (1, kernel_make_fn)]
        kernel = GibbsKernel(target_log_prob_fn=logp, kernel_list=kernel_list)
        self.assertTrue(kernel.is_calibrated)


if __name__ == "__main__":
    tf.test.main()
