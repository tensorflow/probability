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
"""Tests for JointDistribution utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions


class JointDistributionUtilTest(test_util.TestCase):

  def _test_independent_joint_distribution_from_structure_helper(
      self, structure, expect_isinstance):
    distribution = tfd.independent_joint_distribution_from_structure(structure)
    self.assertIsInstance(distribution, expect_isinstance)

    self.assertAllEqualNested(
        distribution.dtype,
        tf.nest.map_structure(lambda d: d.dtype, structure))
    self.assertAllEqualNested(
        distribution.event_shape,
        tf.nest.map_structure(lambda d: d.event_shape, structure))

    x = self.evaluate(distribution.sample(seed=test_util.test_seed()))
    joint_logprob = distribution.log_prob(x)
    indep_logprobs = sum([d.log_prob(x_part)
                          for (d, x_part) in
                          zip(tf.nest.flatten(structure), tf.nest.flatten(x))])
    self.assertAllClose(*self.evaluate((joint_logprob, indep_logprobs)))

  def test_independent_jd_from_distribution_input(self):
    self._test_independent_joint_distribution_from_structure_helper(
        structure=tfd.Normal(0., 1.), expect_isinstance=tfd.Normal)

  def test_independent_jd_from_tuple_input(self):
    self._test_independent_joint_distribution_from_structure_helper(
        structure=(tfd.Normal(0., 1.), tfd.Poisson(1.)),
        expect_isinstance=tfd.JointDistributionSequential)

  def test_independent_jd_from_dict_input(self):
    self._test_independent_joint_distribution_from_structure_helper(
        structure={'a': tfd.Normal(0., 1.), 'b': tfd.Poisson(1.)},
        expect_isinstance=tfd.JointDistributionNamed)

  def test_independent_jd_from_namedtuple_input(self):
    self._test_independent_joint_distribution_from_structure_helper(
        structure=collections.namedtuple('Pair', ['a', 'b'])(
            a=tfd.Normal(0., 1.), b=tfd.Poisson(1.)),
        expect_isinstance=tfd.JointDistributionNamed)

  def test_independent_jd_from_ordereddict_input(self):
    self._test_independent_joint_distribution_from_structure_helper(
        structure=collections.OrderedDict((('a', tfd.Normal(0., 1.)),
                                           ('b', tfd.Poisson(1.)))),
        expect_isinstance=tfd.JointDistributionNamed)

  def test_independent_jd_from_nested_input(self):
    self._test_independent_joint_distribution_from_structure_helper(
        structure=[tfd.Normal(0., 1.),
                   {'b': tfd.Poisson(1.),
                    'c': (tfd.Dirichlet([1., 1.]),)}],
        expect_isinstance=tfd.JointDistributionSequential)

if __name__ == '__main__':
  tf.test.main()
