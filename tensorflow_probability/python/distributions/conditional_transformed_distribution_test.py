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
"""Tests for ConditionalTransformedDistribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.distributions import transformed_distribution_test

tfd = tfp.distributions
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


class _ChooseLocation(tfp.bijectors.ConditionalBijector):
  """A Bijector which chooses between one of two location parameters."""

  def __init__(self, loc, name="ChooseLocation"):
    self._graph_parents = []
    self._name = name
    with self._name_scope("init", values=[loc]):
      self._loc = tf.convert_to_tensor(value=loc, name="loc")
      super(_ChooseLocation, self).__init__(
          graph_parents=[self._loc],
          is_constant_jacobian=True,
          validate_args=False,
          forward_min_event_ndims=0,
          name=name)

  def _forward(self, x, z):
    return x + self._gather_loc(z)

  def _inverse(self, x, z):
    return x - self._gather_loc(z)

  def _inverse_log_det_jacobian(self, x, event_ndims, z=None):
    return 0.

  def _gather_loc(self, z):
    z = tf.convert_to_tensor(value=z)
    z = tf.cast((1 + z) / 2, tf.int32)
    return tf.gather(self._loc, z)


@test_util.run_all_in_graph_and_eager_modes
class ConditionalTransformedDistributionTest(
    transformed_distribution_test.TransformedDistributionTest):

  def _cls(self):
    return tfd.ConditionalTransformedDistribution

  def testConditioning(self):
    conditional_normal = tfd.ConditionalTransformedDistribution(
        distribution=tfd.Normal(loc=0., scale=1.),
        bijector=_ChooseLocation(loc=[-100., 100.]))
    z = [-1, +1, -1, -1, +1]
    self.assertAllClose(
        np.sign(
            self.evaluate(
                conditional_normal.sample(5, bijector_kwargs={"z": z}))), z)


# TODO(b/122840816): Should these tests also run in eager mode?  Tests in
# `transformed_distribution_test.ScalarToMultiTest` are not currently run in
# eager mode.
class ConditionalScalarToMultiTest(
    transformed_distribution_test.ScalarToMultiTest):

  def _cls(self):
    return tfd.ConditionalTransformedDistribution


if __name__ == "__main__":
  tf.test.main()
