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

import collections

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import dirichlet
from tensorflow_probability.python.distributions import joint_distribution_named as jdn
from tensorflow_probability.python.distributions import joint_distribution_sequential as jds
from tensorflow_probability.python.distributions import joint_distribution_util as jdu
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import poisson
from tensorflow_probability.python.internal import test_util


class JointDistributionUtilTest(test_util.TestCase):

  def _test_independent_joint_distribution_from_structure_helper(
      self, structure, expect_isinstance):
    distribution = jdu.independent_joint_distribution_from_structure(structure)
    self.assertIsInstance(distribution, expect_isinstance)

    self.assertEqual(
        distribution.dtype,
        tf.nest.map_structure(lambda d: d.dtype, structure))
    self.assertEqual(
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
        structure=normal.Normal(0., 1.), expect_isinstance=normal.Normal)

  def test_independent_jd_from_tuple_input(self):
    self._test_independent_joint_distribution_from_structure_helper(
        structure=(normal.Normal(0., 1.), poisson.Poisson(1.)),
        expect_isinstance=jds.JointDistributionSequential)

  def test_independent_jd_from_dict_input(self):
    self._test_independent_joint_distribution_from_structure_helper(
        structure={'a': normal.Normal(0., 1.), 'b': poisson.Poisson(1.)},
        expect_isinstance=jdn.JointDistributionNamed)

  def test_independent_jd_from_namedtuple_input(self):
    self._test_independent_joint_distribution_from_structure_helper(
        structure=collections.namedtuple('Pair', ['a', 'b'])(
            a=normal.Normal(0., 1.), b=poisson.Poisson(1.)),
        expect_isinstance=jdn.JointDistributionNamed)

  def test_independent_jd_from_ordereddict_input(self):
    self._test_independent_joint_distribution_from_structure_helper(
        structure=collections.OrderedDict((('a', normal.Normal(0., 1.)),
                                           ('b', poisson.Poisson(1.)))),
        expect_isinstance=jdn.JointDistributionNamed)

  def test_independent_jd_from_nested_input(self):
    self._test_independent_joint_distribution_from_structure_helper(
        structure=[normal.Normal(0., 1.),
                   {'b': poisson.Poisson(1.),
                    'c': (dirichlet.Dirichlet([1., 1.]),)}],
        expect_isinstance=jds.JointDistributionSequential)

  def test_independent_jd_from_nested_input_one_empty(self):
    self._test_independent_joint_distribution_from_structure_helper(
        structure={'a': {'b': normal.Normal(0., 1.)},
                   'c': {'d': normal.Normal(0., 1.)}},
        expect_isinstance=jdn.JointDistributionNamed)

  def test_batch_ndims_nested_input(self):
    dist = jdu.independent_joint_distribution_from_structure(
        [normal.Normal(0., tf.ones([5, 4])),
         {'b': poisson.Poisson(tf.ones([5])),
          'c': dirichlet.Dirichlet(tf.ones([5, 3]))}],
        batch_ndims=1)
    self.assertAllEqualNested(dist.event_shape, [[4], {'b': [], 'c': [3]}])
    self.assertAllEqual(dist.batch_shape, [5])

  def test_batch_ndims_single_distribution_input(self):
    dist = jdu.independent_joint_distribution_from_structure(
        normal.Normal(0., tf.ones([5, 4])), batch_ndims=2)
    self.assertAllEqual(dist.event_shape, [])
    self.assertAllEqual(dist.batch_shape, [5, 4])

    dist = jdu.independent_joint_distribution_from_structure(
        normal.Normal(0., tf.ones([5, 4])), batch_ndims=1)
    self.assertAllEqual(dist.event_shape, [4])
    self.assertAllEqual(dist.batch_shape, [5])

    dist = jdu.independent_joint_distribution_from_structure(
        normal.Normal(0., tf.ones([5, 4])), batch_ndims=0)
    self.assertAllEqual(dist.event_shape, [5, 4])
    self.assertAllEqual(dist.batch_shape, [])


if __name__ == '__main__':
  test_util.main()
