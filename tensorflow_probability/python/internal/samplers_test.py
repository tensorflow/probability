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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class RandomTest(test_util.TestCase):

  @test_util.substrate_disable_stateful_random_test
  def test_sanitize_int(self):
    seed1 = samplers.sanitize_seed(seed=123)
    seed2 = samplers.sanitize_seed(seed=123)
    if tf.executing_eagerly():
      self.assertNotAllEqual(seed1, seed2)
    else:
      self.assertAllEqual(seed1, seed2)

  @test_util.substrate_disable_stateful_random_test
  def test_sanitize_none(self):
    seed1 = samplers.sanitize_seed(seed=None)
    seed2 = samplers.sanitize_seed(seed=None)
    self.assertNotAllEqual(seed1, seed2)

  def test_sanitize_tensor_or_tensorlike(self):
    seed = test_util.test_seed(sampler_type='stateless')
    seed1 = samplers.sanitize_seed(seed=self.evaluate(seed))
    seed2 = samplers.sanitize_seed(seed)
    self.assertAllEqual(seed1, seed2)

  def test_split(self):
    seed = test_util.test_seed(sampler_type='stateless')
    seed1, seed2 = samplers.split_seed(seed)
    seed3, seed4 = samplers.split_seed(seed)
    self.assertNotAllEqual(seed, seed1)
    self.assertNotAllEqual(seed, seed2)
    self.assertNotAllEqual(seed1, seed2)
    self.assertAllEqual(self.evaluate([seed1, seed2]),
                        self.evaluate([seed3, seed4]))

  def test_salted_split(self):
    seed = test_util.test_seed(sampler_type='stateless')
    seed1, seed2 = samplers.split_seed(seed, salt='normal')
    seed3, seed4 = samplers.split_seed(seed, salt='lognormal')
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
    s1 = sampler(seed=(1, 2), **kwargs)
    s2 = sampler(seed=(1, 2), **kwargs)
    self.assertAllEqual(s1, s2)
    self.verify_tf_behavior_match(sampler, kwargs)

  @test_util.substrate_disable_stateful_random_test
  def verify_tf_behavior_match(self, sampler, kwargs):
    # We don't test these scenarios for numpy, jax, where we don't support
    # stateful sampling.
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
  tf.test.main()
