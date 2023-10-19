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
"""Tests for TFP-internal random samplers."""

from absl import flags
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util

flags.DEFINE_enum('test_tfp_jax_prng', 'default', ['default', 'rbg'],
                  'Which PRNG implementation to test with.')

FLAGS = flags.FLAGS
JAX_MODE = False
NUMPY_MODE = False


@test_util.test_all_tf_execution_regimes
class RandomTest(test_util.TestCase):

  def setUp(self):
    super().setUp()

    if JAX_MODE and FLAGS.test_tfp_jax_prng != 'default':
      from jax import config  # pylint: disable=g-import-not-at-top
      config.update('jax_default_prng_impl', FLAGS.test_tfp_jax_prng)

  @test_util.substrate_disable_stateful_random_test
  def test_sanitize_int(self):
    seed1 = samplers.sanitize_seed(seed=123)
    seed2 = samplers.sanitize_seed(seed=123)
    if tf.executing_eagerly():
      self.assertNotAllEqual(seed1, seed2)
    else:
      self.assertAllEqual(seed1, seed2)

  @test_util.substrate_disable_stateful_random_test
  def test_sanitize_then_split_equivalent_split_int(self):
    seed = test_util.test_seed()
    sanitized = samplers.sanitize_seed(seed, salt='please pass the')
    s1 = samplers.split_seed(sanitized, n=3)
    if tf.executing_eagerly():
      tf.random.set_seed(seed)
    s2 = samplers.split_seed(seed, n=3, salt='please pass the')
    self.assertAllAssertsNested(self.assertAllEqual, s1, s2)

  @test_util.substrate_disable_stateful_random_test
  def test_sanitize_none(self):
    seed1 = samplers.sanitize_seed(seed=None)
    seed2 = samplers.sanitize_seed(seed=None)
    self.assertNotAllEqual(seed1, seed2)

  def test_sanitize_stateless_seeds(self):
    seed = test_util.test_seed(sampler_type='stateless')
    seed1 = samplers.sanitize_seed(seed=self.evaluate(seed))
    seed2 = samplers.sanitize_seed(seed)
    seed1, seed2 = self.evaluate([seed1, seed2])
    self.assertAllEqual(seed1, seed2)

  @test_util.disable_test_for_backend(
      disable_jax=True, reason='JAX seeds must be jax.random.PRNGKey objects')
  def test_sanitize_tensor_or_tensorlike(self):
    # TODO(b/223267515): In JAX mode, test that we raise a user-friendly error
    # when the seed is not a PRNGKey.
    seed1 = samplers.sanitize_seed([0, 1])
    seed2 = samplers.sanitize_seed(np.array([0, 1]))
    seed1, seed2 = self.evaluate([seed1, seed2])
    self.assertAllEqual(seed1, seed2)

  def test_split(self):
    seed = test_util.test_seed(sampler_type='stateless')
    seed1, seed2 = samplers.split_seed(seed)
    seed3, seed4 = samplers.split_seed(seed)
    seed, seed1, seed2, seed3, seed4 = self.evaluate(
        [seed, seed1, seed2, seed3, seed4])
    self.assertNotAllEqual(seed, seed1)
    self.assertNotAllEqual(seed, seed2)
    self.assertNotAllEqual(seed1, seed2)
    self.assertAllEqual(seed1, seed3)
    self.assertAllEqual(seed2, seed4)

  def test_salted_split(self):
    seed = test_util.test_seed(sampler_type='stateless')
    seed1, seed2 = samplers.split_seed(seed, salt='normal')
    seed3, seed4 = samplers.split_seed(seed, salt='lognormal')
    seed, seed1, seed2, seed3, seed4 = self.evaluate(
        [seed, seed1, seed2, seed3, seed4])
    self.assertNotAllEqual(seed, seed1)
    self.assertNotAllEqual(seed, seed2)
    self.assertNotAllEqual(seed1, seed2)
    self.assertNotAllEqual(seed1, seed3)
    self.assertNotAllEqual(seed2, seed4)
    self.assertNotAllEqual(seed3, seed4)

  @parameterized.named_parameters(
      dict(testcase_name='_categorical',
           sampler=samplers.categorical,
           kwargs=dict(logits=[[1, 1.05, 1]], num_samples=5)),
      dict(testcase_name='_gamma',
           sampler=samplers.gamma,
           kwargs=dict(shape=[2, 3], alpha=[.5, 1, 2.2], beta=0.75)),
      dict(testcase_name='_normal',
           sampler=samplers.normal,
           kwargs=dict(shape=[2])),
      dict(testcase_name='_poisson',
           sampler=samplers.poisson,
           kwargs=dict(shape=[2, 3], lam=[1.5, 5.5, 8.5])),
      dict(testcase_name='_poisson_scalar',
           sampler=samplers.poisson,
           kwargs=dict(shape=[], lam=[1.5, 5.5, 8.5])),
      dict(testcase_name='_shuffle',
           sampler=samplers.shuffle,
           kwargs=dict(value=list(range(10)))),
      dict(testcase_name='_uniform',
           sampler=samplers.uniform,
           kwargs=dict(shape=[2])))
  def test_sampler(self, sampler, kwargs):
    if FLAGS.test_tfp_jax_prng == 'rbg' and sampler == samplers.gamma:
      self.skipTest('gamma sampler not implemented for rbg PRNG.')
    seed = test_util.test_seed(sampler_type='stateless')
    s1 = sampler(seed=seed, **kwargs)
    s2 = sampler(seed=seed, **kwargs)
    self.assertAllEqual(s1, s2)

    # We don't test these scenarios for numpy, jax, where we don't support
    # stateful sampling.
    if not JAX_MODE and not NUMPY_MODE:
      self.verify_tf_behavior_match(sampler, kwargs)

  def verify_tf_behavior_match(self, sampler, kwargs):
    s1 = sampler(seed=123, **kwargs)
    s2 = sampler(seed=123, **kwargs)
    tf_sampler = getattr(tf.random, sampler.__name__)
    tf_s1 = tf_sampler(seed=123, **kwargs)
    tf_s2 = tf_sampler(seed=123, **kwargs)
    if tf.executing_eagerly():
      self.assertNotAllEqual(s1, s2)
      self.assertNotAllEqual(tf_s1, tf_s2)
    else:
      self.assertAllEqual(s1, s2)
      self.assertAllEqual(tf_s1, tf_s2)


if __name__ == '__main__':
  test_util.main()
