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
"""Tests for MCMC utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
# Dependency imports
import numpy as np

import tensorflow as tf

from tensorflow_probability.python.mcmc.util import choose
from tensorflow_probability.python.mcmc.util import is_namedtuple_like
from tensorflow.python.framework import test_util


class ChooseTest(tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def test_works_for_nested_namedtuple(self):
    Results = collections.namedtuple('Results', ['field1', 'inner'])  # pylint: disable=invalid-name
    InnerResults = collections.namedtuple('InnerResults', ['fieldA', 'fieldB'])  # pylint: disable=invalid-name
    accepted = Results(
        field1=np.int32([1, 3]),
        inner=InnerResults(
            fieldA=np.float32([5, 7]),
            fieldB=[
                np.float32([9, 11]),
                np.float64([13, 15]),
            ]))
    rejected = Results(
        field1=np.int32([0, 2]),
        inner=InnerResults(
            fieldA=np.float32([4, 6]),
            fieldB=[
                np.float32([8, 10]),
                np.float64([12, 14]),
            ]))
    chosen = choose(
        tf.constant([False, True]),
        accepted,
        rejected)
    chosen_ = self.evaluate(chosen)
    # Lhs should be 0,4,8,12 and rhs=lhs+3.
    expected = Results(
        field1=np.int32([0, 3]),
        inner=InnerResults(
            fieldA=np.float32([4, 7]),
            fieldB=[
                np.float32([8, 11]),
                np.float64([12, 15]),
            ]))
    self.assertAllClose(expected, chosen_, atol=0., rtol=1e-5)


class IsNamedTupleLikeTest(tf.test.TestCase):

  def test_true_for_namedtuple_without_fields(self):
    NoFields = collections.namedtuple('NoFields', [])  # pylint: disable=invalid-name
    no_fields = NoFields()
    self.assertTrue(is_namedtuple_like(no_fields))

  def test_true_for_namedtuple_with_fields(self):
    HasFields = collections.namedtuple('HasFields', ['a', 'b'])  # pylint: disable=invalid-name
    has_fields = HasFields(a=1, b=2)
    self.assertTrue(is_namedtuple_like(has_fields))

  def test_false_for_base_case(self):
    self.assertFalse(is_namedtuple_like(tuple([1, 2])))
    self.assertFalse(is_namedtuple_like(list([3., 4.])))
    self.assertFalse(is_namedtuple_like(dict(a=5, b=6)))
    self.assertFalse(is_namedtuple_like(tf.constant(1.)))
    self.assertFalse(is_namedtuple_like(np.int32()))


if __name__ == '__main__':
  tf.test.main()
