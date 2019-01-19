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
import tensorflow_probability as tfp

from tensorflow_probability.python.mcmc.util import choose
from tensorflow_probability.python.mcmc.util import is_namedtuple_like
from tensorflow_probability.python.mcmc.util import maybe_call_fn_and_grads
from tensorflow_probability.python.mcmc.util import smart_for_loop
tfd = tfp.distributions
tfe = tf.contrib.eager


class ChooseTest(tf.test.TestCase):

  @tfe.run_test_in_graph_and_eager_modes()
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

  @tfe.run_test_in_graph_and_eager_modes()
  def test_selects_batch_members_from_list_of_arrays(self):
    # Shape of each array: [2, 3] = [batch_size, event_size]
    # This test verifies that is_accepted selects batch members, despite the
    # "usual" broadcasting being applied on the right first (event first).
    zeros_states = [np.zeros((2, 3))]
    ones_states = [np.ones((2, 3))]
    chosen = choose(
        tf.constant([True, False]),
        zeros_states,
        ones_states)
    chosen_ = self.evaluate(chosen)

    # Make sure outer list wasn't interpreted as a dimenion of an array.
    self.assertIsInstance(chosen_, list)
    expected_array = np.array([
        [0., 0., 0.],  # zeros_states selected for first batch
        [1., 1., 1.],  # ones_states selected for second
    ])
    expected = [expected_array]
    self.assertAllEqual(expected, chosen_)


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


@tfe.run_all_tests_in_graph_and_eager_modes
class GradientTest(tf.test.TestCase):

  def testGradientComputesCorrectly(self):
    dtype = np.float32
    def fn(x, y):
      return x**2 + y**2

    fn_args = [dtype(3), dtype(3)]
    # Convert function input to a list of tensors
    fn_args = [tf.convert_to_tensor(arg) for arg in fn_args]
    fn_result, grads = maybe_call_fn_and_grads(fn, fn_args)
    fn_result_, grads_ = self.evaluate([fn_result, grads])
    self.assertNear(18., fn_result_, err=1e-5)
    for grad in grads_:
      self.assertAllClose(grad, dtype(6), atol=0., rtol=1e-5)

  def testGradientWorksDespiteBijectorCaching(self):
    d = tfd.LogNormal(loc=0., scale=1.)
    x = tf.constant(2.)
    fn_result, grads = maybe_call_fn_and_grads(lambda x: d.log_prob(x), x)  # pylint: disable=unnecessary-lambda
    self.assertAllEqual(False, fn_result is None)
    self.assertAllEqual([False], [g is None for g in grads])

  def testNoGradientsNiceError(self):
    dtype = np.float32

    def fn(x, y):
      return x**2 + tf.stop_gradient(y)**2

    fn_args = [dtype(3), dtype(3)]
    # Convert function input to a list of tensors
    fn_args = [
        tf.convert_to_tensor(arg, name='arg{}'.format(i))
        for i, arg in enumerate(fn_args)
    ]
    if tf.executing_eagerly():
      with self.assertRaisesRegexp(
          ValueError, 'Encountered `None`.*\n.*fn_arg_list.*\n.*None'):
        maybe_call_fn_and_grads(fn, fn_args)
    else:
      with self.assertRaisesRegexp(
          ValueError, 'Encountered `None`.*\n.*fn_arg_list.*arg1.*\n.*None'):
        maybe_call_fn_and_grads(fn, fn_args)


class SmartForLoopTest(tf.test.TestCase):

  def test_python_for_loop(self):
    n = tf.constant(10, dtype=tf.int64)
    counter = collections.Counter()
    def body(x):
      counter['body_calls'] += 1
      return [x + 1]

    result = smart_for_loop(
        loop_num_iter=n, body_fn=body, initial_loop_vars=[tf.constant(1)])
    self.assertEqual(10, counter['body_calls'])
    self.assertAllClose([11], self.evaluate(result))

  def test_tf_while_loop(self):
    iters = 10
    n = tf.placeholder_with_default(input=np.int64(iters), shape=())
    counter = collections.Counter()
    def body(x):
      counter['body_calls'] += 1
      return [x + 1]

    result = smart_for_loop(
        loop_num_iter=n, body_fn=body, initial_loop_vars=[tf.constant(1)])
    self.assertEqual(iters if tf.executing_eagerly() else 1,
                     counter['body_calls'])
    self.assertAllClose([11], self.evaluate(result))


if __name__ == '__main__':
  tf.test.main()
