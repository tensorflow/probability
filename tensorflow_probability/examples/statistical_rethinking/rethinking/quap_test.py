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
"""Tests for quap."""

import collections

# Dependency imports
from absl.testing import parameterized

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.examples.statistical_rethinking import rethinking
from tensorflow_probability.python.internal import test_util
tfd = tfp.distributions


class QuapTestJDNamed(test_util.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': 'jd_named',
       'joint_distribution': tfd.JointDistributionNamed},
      {'testcase_name': 'jd_named_autobatched',
       'joint_distribution': tfd.JointDistributionNamedAutoBatched})
  def test_no_conditioning(self, joint_distribution):
    model = joint_distribution({
        'a': tfd.Normal(1., 1.),
        'b': tfd.Normal(3., 2.),
        'c': lambda a, b: tfd.Normal(a + b, 5.)})

    approx = rethinking.quap(model)
    self.assertAllCloseNested(
        approx.mean(),
        dict(a=1., b=3., c=4.))

    self.assertAllCloseNested(
        approx.stddev(),
        dict(a=1., b=2., c=30 ** 0.5),
        atol=1e-04)

  @parameterized.named_parameters(
      {'testcase_name': 'jd_named',
       'joint_distribution': tfd.JointDistributionNamed},
      {'testcase_name': 'jd_named_autobatched',
       'joint_distribution': tfd.JointDistributionNamedAutoBatched})
  def test_conditioning(self, joint_distribution):
    model = joint_distribution({
        'a': tfd.Normal(1., 2.),
        'b': lambda a: tfd.Normal(2 * a + 1., 5.)})

    approx = rethinking.quap(model,
                             data={'b': 2.},
                             initial_position={'a': 1.5})

    dists, _ = approx.sample_distributions()
    # See, e.g., Bishop, equation 2.116
    self.assertAllClose(dists['a'].mean(), 33. / 41.)
    self.assertAllClose(dists['a'].stddev(), (100. / 41.) ** 0.5)


class QuapTestJDNamedTuple(test_util.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': 'jd_named',
       'joint_distribution': tfd.JointDistributionNamed},
      {'testcase_name': 'jd_named_autobatched',
       'joint_distribution': tfd.JointDistributionNamedAutoBatched})
  def test_no_conditioning(self, joint_distribution):
    ModelSpec = collections.namedtuple('ModelSpec', 'a,b,c')
    model = joint_distribution(ModelSpec(
        a=tfd.Normal(1., 1.),
        b=tfd.Normal(3., 2.),
        c=lambda a, b: tfd.Normal(a + b, 5.)))

    approx = rethinking.quap(model)
    self.assertAllCloseNested(
        approx.mean(),
        ModelSpec(a=1., b=3., c=4.))

    self.assertAllCloseNested(
        approx.stddev(),
        ModelSpec(a=1., b=2., c=30 ** 0.5),
        atol=1e-04)

  @parameterized.named_parameters(
      {'testcase_name': 'jd_named',
       'joint_distribution': tfd.JointDistributionNamed},
      {'testcase_name': 'jd_named_autobatched',
       'joint_distribution': tfd.JointDistributionNamedAutoBatched})
  def test_conditioning(self, joint_distribution):
    ModelSpec = collections.namedtuple('ModelSpec', 'a,b')
    model = joint_distribution(ModelSpec(
        a=tfd.Normal(1., 2.),
        b=lambda a: tfd.Normal(2 * a + 1., 5.)))

    approx = rethinking.quap(model,
                             data=ModelSpec(a=None, b=2.),
                             initial_position=ModelSpec(a=1.5, b=None))

    dists, _ = approx.sample_distributions()
    # See, e.g., Bishop, equation 2.116
    self.assertAllClose(dists.a.mean(), 33. / 41.)
    self.assertAllClose(dists.a.stddev(), (100. / 41.) ** 0.5)


class QuapTestJDSequential(test_util.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': 'jd_sequential',
       'joint_distribution': tfd.JointDistributionSequential},
      {'testcase_name': 'jd_sequential_autobatched',
       'joint_distribution': tfd.JointDistributionSequentialAutoBatched})
  def test_no_conditioning(self, joint_distribution):
    model = joint_distribution([
        tfd.Normal(1., 1.),
        tfd.Normal(3., 2.),
        lambda b, a: tfd.Normal(a + b, 5.)])

    approx = rethinking.quap(model)
    self.assertAllCloseNested(
        approx.mean(),
        [1., 3., 4.])

    self.assertAllCloseNested(
        approx.stddev(),
        [1., 2., 30 ** 0.5],
        atol=1e-04)

  @parameterized.named_parameters(
      {'testcase_name': 'jd_sequential',
       'joint_distribution': tfd.JointDistributionSequential},
      {'testcase_name': 'jd_sequential_autobatched',
       'joint_distribution': tfd.JointDistributionSequentialAutoBatched})
  def test_conditioning(self, joint_distribution):
    model = joint_distribution([
        tfd.Normal(1., 2.),
        lambda a: tfd.Normal(2 * a + 1., 5.)])

    approx = rethinking.quap(model,
                             data=[None, 2.],
                             initial_position=[1.5])

    dists, _ = approx.sample_distributions()
    # See, e.g., Bishop, equation 2.116
    self.assertAllClose(dists[0].mean(), 33. / 41.)
    self.assertAllClose(dists[0].stddev(), (100. / 41.) ** 0.5)


if __name__ == '__main__':
  tf.test.main()
