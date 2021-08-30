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
import warnings

# Dependency imports

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import auto_composite_tensor
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.mcmc.internal import util


JAX_MODE = False
NUMPY_MODE = False


@test_util.test_all_tf_execution_regimes
class ChooseTest(test_util.TestCase):

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
    chosen = util.choose(
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

  def test_selects_batch_members_from_list_of_arrays(self):
    # Shape of each array: [2, 3] = [batch_size, event_size]
    # This test verifies that is_accepted selects batch members, despite the
    # "usual" broadcasting being applied on the right first (event first).
    zeros_states = [np.zeros((2, 3))]
    ones_states = [np.ones((2, 3))]
    chosen = util.choose(
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

  @test_util.jax_disable_test_missing_functionality('no tf.TensorSpec')
  @test_util.numpy_disable_test_missing_functionality('no tf.TensorSpec')
  def test_conserves_partial_shapes(self):
    # Testing that `choose` correctly propagates shape information about the
    # arms, even when those shapes are partial and when the shape of the
    # condition is partial as well.
    input_signature = [tf.TensorSpec([], tf.int32)]
    @tf.function(input_signature=input_signature)
    def try_me(batch_size):
      # The while loop is necessary for this test, because it affects TF's shape
      # inference behavior somehow (maybe another appearance of b/139013403?).
      init = tf.zeros([batch_size, 1])
      def body_fn(state):
        arm_1 = state
        arm_2 = tf.ones([batch_size, 1])
        condition = tf.ones([batch_size], dtype=tf.bool)
        result = util.choose(condition, arm_1, arm_2)
        # The test is the automatic check inside `tf.while_loop` that the static
        # shape returned here is the same as the static shape of the input
        # `state`.
        return result
      tf.while_loop(
          cond=lambda *_: False,
          body=body_fn,
          loop_vars=(init,))
    try_me(5)

  def test_choose_from(self):
    options = [
        [{
            'value': 1.
        }, (2., 3)],
        [{
            'value': 2.
        }, (3., 4)],
        [{
            'value': 3.
        }, (4., 5)],
    ]
    first_option = util.choose_from(tf.constant(0), options)
    second_option = util.choose_from(tf.constant(1), options)
    third_option = util.choose_from(2, options)
    negative_option = util.choose_from(tf.constant(-10), options)
    large_option = util.choose_from(tf.constant(10), options)
    self.assertAllEqualNested(first_option, options[0], check_types=True)
    self.assertAllEqualNested(second_option, options[1], check_types=True)
    self.assertAllEqualNested(third_option, options[2], check_types=True)
    self.assertAllEqualNested(negative_option, options[0], check_types=True)
    self.assertAllEqualNested(large_option, options[2], check_types=True)


class IsNamedTupleLikeTest(test_util.TestCase):

  def test_true_for_namedtuple_without_fields(self):
    NoFields = collections.namedtuple('NoFields', [])  # pylint: disable=invalid-name
    no_fields = NoFields()
    self.assertTrue(util.is_namedtuple_like(no_fields))

  def test_true_for_namedtuple_with_fields(self):
    HasFields = collections.namedtuple('HasFields', ['a', 'b'])  # pylint: disable=invalid-name
    has_fields = HasFields(a=1, b=2)
    self.assertTrue(util.is_namedtuple_like(has_fields))

  def test_false_for_base_case(self):
    self.assertFalse(util.is_namedtuple_like(tuple([1, 2])))
    self.assertFalse(util.is_namedtuple_like(list([3., 4.])))
    self.assertFalse(util.is_namedtuple_like(dict(a=5, b=6)))
    self.assertFalse(util.is_namedtuple_like(tf.constant(1.)))
    self.assertFalse(util.is_namedtuple_like(np.int32()))


class GradientTest(test_util.TestCase):

  @test_util.numpy_disable_gradient_test
  def testGradientComputesCorrectly(self):
    dtype = np.float32
    def fn(x, y):
      return x**2 + y**2

    fn_args = [dtype(3), dtype(3)]
    # Convert function input to a list of tensors
    fn_args = [tf.convert_to_tensor(value=arg) for arg in fn_args]
    fn_result, grads = util.maybe_call_fn_and_grads(fn, fn_args)
    fn_result_, grads_ = self.evaluate([fn_result, grads])
    self.assertNear(18., fn_result_, err=1e-5)
    for grad in grads_:
      self.assertAllClose(grad, dtype(6), atol=0., rtol=1e-5)

  @test_util.numpy_disable_gradient_test
  def testGradientWorksDespiteBijectorCaching(self):
    x = tf.constant(2.)
    fn_result, grads = util.maybe_call_fn_and_grads(
        lambda x_: tfd.LogNormal(loc=0., scale=1.).log_prob(x_), x)
    self.assertAllEqual(False, fn_result is None)
    self.assertAllEqual([False], [g is None for g in grads])

  @test_util.numpy_disable_gradient_test
  def testGradientWorksForMultivariateNormalTriL(self):
    # TODO(b/72831017): Remove this once bijector cacheing is fixed for
    # graph mode.
    if not tf.executing_eagerly():
      self.skipTest('Gradients get None values in graph mode.')
    d = tfd.MultivariateNormalTriL(scale_tril=tf.eye(2))
    x = d.sample(seed=(0, 0))
    fn_result, grads = util.maybe_call_fn_and_grads(d.log_prob, x)
    self.assertAllEqual(False, fn_result is None)
    self.assertAllEqual([False], [g is None for g in grads])

  @test_util.jax_disable_test_missing_functionality('None gradients')
  @test_util.numpy_disable_gradient_test
  def testNoGradientsNiceError(self):
    dtype = np.float32

    def fn(x, y):
      return x**2 + tf.stop_gradient(y)**2

    fn_args = [dtype(3), dtype(3)]
    # Convert function input to a list of tensors
    fn_args = [
        tf.convert_to_tensor(value=arg, name='arg{}'.format(i))
        for i, arg in enumerate(fn_args)
    ]
    if tf.executing_eagerly():
      with self.assertRaisesRegexp(
          ValueError, 'Encountered `None`.*\n.*fn_arg_list.*\n.*None'):
        util.maybe_call_fn_and_grads(fn, fn_args)
    else:
      with self.assertRaisesRegexp(
          ValueError, 'Encountered `None`.*\n.*fn_arg_list.*arg1.*\n.*None'):
        util.maybe_call_fn_and_grads(fn, fn_args)

  @test_util.numpy_disable_gradient_test
  def testGradientNumBodyCalls(self):
    counter = collections.Counter()

    dtype = np.float32

    def fn(x, y):
      counter['body_calls'] += 1
      return x**2 + y**2

    fn_args = [dtype(3), dtype(3)]
    # Convert function input to a list of tensors.
    fn_args = [tf.convert_to_tensor(value=arg) for arg in fn_args]
    util.maybe_call_fn_and_grads(fn, fn_args)
    expected_num_calls = 1 if JAX_MODE or not tf.executing_eagerly() else 2
    self.assertEqual(expected_num_calls, counter['body_calls'])


@test_util.test_all_tf_execution_regimes
class SmartForLoopTest(test_util.TestCase):

  @parameterized.parameters(0, 1, 10)
  def test_static_num_iters(self, iters):
    counter = None
    # following loop variables not @parameterized because the tf.constants
    # would be executed outside the Eager mode that
    # @test_util.test_all_tf_execution_regimes creates
    for n in [iters, tf.constant(iters, dtype=tf.int64),
              tf.constant(iters, dtype=tf.int32)]:
      counter = collections.Counter()
      def body(x):
        counter['body_calls'] += 1
        return [x + 1]

      result = util.smart_for_loop(
          loop_num_iter=n, body_fn=body, initial_loop_vars=[tf.constant(1)])
      if JAX_MODE:  # JAX always traces loop bodies exactly once
        self.assertEqual(1, counter['body_calls'])
      elif tf.executing_eagerly():
        self.assertEqual(iters, counter['body_calls'])
      else:
        expected_num_calls = 1 if iters > 0 else 0
        self.assertEqual(expected_num_calls, counter['body_calls'])
      self.assertAllClose([iters + 1], self.evaluate(result))

  def test_placeholder_num_iters(self):
    iters = 10
    n = tf1.placeholder_with_default(np.int64(iters), shape=())
    counter = collections.Counter()
    def body(x):
      counter['body_calls'] += 1
      return [x + 1]

    result = util.smart_for_loop(
        loop_num_iter=n, body_fn=body, initial_loop_vars=[tf.constant(1)])
    if tf.executing_eagerly() and not JAX_MODE:  # JAX always traces loops
      self.assertEqual(iters, counter['body_calls'])
    else:
      self.assertEqual(1, counter['body_calls'])
    self.assertAllClose([11], self.evaluate(result))

  def test_unroll_threshold(self):
    iters = 50
    counter = collections.Counter()
    def body(x):
      counter['body_calls'] += 1
      return [x + 1]

    result = util.smart_for_loop(
        loop_num_iter=iters,
        body_fn=body,
        initial_loop_vars=[tf.constant(1)],
        unroll_threshold=iters)
    if JAX_MODE:  # JAX always traces loop bodies exactly once
      self.assertEqual(1, counter['body_calls'])
    else:
      self.assertEqual(iters, counter['body_calls'])
    self.assertAllClose([iters + 1], self.evaluate(result))


@test_util.test_all_tf_execution_regimes
class TraceScanTest(test_util.TestCase):

  def testBasic(self):

    def _loop_fn(state, element):
      return state + element

    def _trace_fn(state):
      return [state, state * 2]

    final_state, trace = util.trace_scan(
        loop_fn=_loop_fn, initial_state=0, elems=[1, 2], trace_fn=_trace_fn)

    self.assertAllClose([], tensorshape_util.as_list(final_state.shape))
    self.assertAllClose([2], tensorshape_util.as_list(trace[0].shape))
    self.assertAllClose([2], tensorshape_util.as_list(trace[1].shape))

    final_state, trace = self.evaluate([final_state, trace])

    self.assertAllClose(3, final_state)
    self.assertAllClose([1, 3], trace[0])
    self.assertAllClose([2, 6], trace[1])

  @test_util.jax_disable_test_missing_functionality('b/157611426')
  @parameterized.named_parameters(
      ('static_length', True),
      ('dynamic_length', False))
  def testTraceCriterion(self, static_length):
    final_state, trace = self.evaluate(
        util.trace_scan(
            loop_fn=lambda state, element: state + element,
            initial_state=0,
            elems=[1, 2, 3, 4, 5, 6, 7],
            trace_fn=lambda state: state / 2,
            trace_criterion_fn=lambda state: tf.equal(state % 2, 0),
            static_trace_allocation_size=3 if static_length else None))
    self.assertAllClose(7 + 6 + 5 + 4 + 3 + 2 + 1, final_state)
    self.assertAllClose([3, 5, 14], trace)

  @test_util.jax_disable_test_missing_functionality('b/157611426')
  @parameterized.named_parameters(
      ('static_length', True),
      ('dynamic_length', False))
  def testConditionFn(self, static_length):
    final_state, trace = self.evaluate(
        util.trace_scan(
            loop_fn=lambda state, element: state + element,
            initial_state=0,
            elems=[1, 2, 3, 4, 5, 6, 7],
            trace_fn=lambda state: state / 2,
            condition_fn=lambda step, state, num_traced, trace: state < 9,
            static_trace_allocation_size=4 if static_length else None))
    self.assertAllClose(10, final_state)
    self.assertAllClose([.5, 1.5, 3, 5], trace)

  @test_util.jax_disable_test_missing_functionality('b/171298381')
  @test_util.numpy_disable_test_missing_functionality('No expanding composites')
  def testComposite(self):
    auto_normal = auto_composite_tensor.auto_composite_tensor(
        tfd.Normal, omit_kwargs=('name',))

    def _loop_fn(state, element):
      return state + element

    def _trace_fn(state):
      return [state, 2 * state, auto_normal(state, 0.1)]

    final_state, trace = util.trace_scan(
        loop_fn=_loop_fn, initial_state=0., elems=[1., 2.], trace_fn=_trace_fn)

    self.assertAllClose([], tensorshape_util.as_list(final_state.shape))
    self.assertAllClose([2], tensorshape_util.as_list(trace[0].shape))
    self.assertAllClose([2], tensorshape_util.as_list(trace[1].shape))

    self.assertAllClose(3, final_state)
    self.assertAllClose([1, 3], trace[0])
    self.assertAllClose([2, 6], trace[1])

    self.assertIsInstance(trace[2], tfd.Normal)
    self.assertAllClose([1., 3.], trace[2].loc)
    self.assertAllClose([0.1, 0.1], trace[2].scale)


WrapperResults = collections.namedtuple('WrapperResults',
                                        'inner_results, value')
SimpleResults = collections.namedtuple('SimpleResults', 'value')


def _test_setter_fn(simple_results, increment=1):
  return simple_results._replace(value=simple_results.value + increment)


@test_util.test_all_tf_execution_regimes
class IndexRemappingGatherTest(test_util.TestCase):

  def test_rank_1_same_as_gather(self):
    params = [10, 11, 12, 13]
    indices = [3, 2, 0]

    expected = [13, 12, 10]
    result = util.index_remapping_gather(params, indices)
    self.assertAllEqual(np.asarray(indices).shape, result.shape)

    self.assertAllEqual(expected, self.evaluate(result))

  def test_rank_2_and_axis_0(self):
    params = [[95, 46, 17],
              [46, 29, 55]]
    indices = [[0, 0, 1],
               [1, 0, 1]]

    expected = [[95, 46, 55],
                [46, 46, 55]]
    result = util.index_remapping_gather(params, indices)

    self.assertAllEqual(np.asarray(params).shape, result.shape)

    self.assertAllEqual(expected, self.evaluate(result))

  def test_rank_3_and_axis_0(self):
    axis = 0
    params = np.random.randint(10, 100, size=(4, 5, 6))
    indices = np.random.randint(0, params.shape[axis], size=(3, 5, 6))

    result = util.index_remapping_gather(params, indices)
    self.assertAllEqual(indices.shape[:axis + 1] + params.shape[axis + 1:],
                        result.shape)
    result_ = self.evaluate(result)

    for i in range(indices.shape[0]):
      for j in range(params.shape[1]):
        for k in range(params.shape[2]):
          self.assertEqual(params[indices[i, j, k], j, k], result_[i, j, k])

  def test_params_rank3_indices_rank2_axis_0(self):
    axis = 0
    params = np.random.randint(10, 100, size=(4, 5, 2))
    indices = np.random.randint(0, params.shape[axis], size=(6, 5))

    result = util.index_remapping_gather(params, indices)
    self.assertAllEqual(indices.shape[:axis + 1] + params.shape[axis + 1:],
                        result.shape)
    result_ = self.evaluate(result)

    for i in range(indices.shape[0]):
      for j in range(params.shape[1]):
        for k in range(params.shape[2]):
          self.assertEqual(params[indices[i, j], j, k], result_[i, j, k])

  def test_params_rank3_indices_rank1_axis_1(self):
    axis = 1
    params = np.random.randint(10, 100, size=[4, 5, 2])
    indices = np.random.randint(0, params.shape[axis], size=[6])

    result = util.index_remapping_gather(params, indices, axis=axis)
    self.assertAllEqual(
        params.shape[:axis] +
        indices.shape[:1] +
        params.shape[axis + 1:],
        result.shape)
    result_ = self.evaluate(result)

    for i in range(params.shape[0]):
      for j in range(indices.shape[0]):
        for k in range(params.shape[2]):
          self.assertEqual(params[i, indices[j], k], result_[i, j, k])

  def test_params_rank5_indices_rank3_axis_2_iaxis_1(self):
    axis = 2
    indices_axis = 1
    params = np.random.randint(10, 100, size=[4, 5, 2, 3, 4])
    indices = np.random.randint(0, params.shape[axis], size=[5, 6, 3])

    result = util.index_remapping_gather(
        params, indices, axis=axis, indices_axis=indices_axis)
    self.assertAllEqual(
        params.shape[:axis] +
        indices.shape[indices_axis:indices_axis + 1] +
        params.shape[axis + 1:],
        result.shape)
    result_ = self.evaluate(result)

    for i in range(params.shape[0]):
      for j in range(params.shape[1]):
        for k in range(indices.shape[1]):
          for l in range(params.shape[3]):
            for m in range(params.shape[4]):
              self.assertEqual(params[i, j, indices[j, k, l], l, m],
                               result_[i, j, k, l, m])


class FakeWrapperOld(object):

  def __init__(self, inner_kernel):
    self.parameters = dict(inner_kernel=inner_kernel)


class FakeWrapperNew(object):

  def __init__(self, inner_kernel, store_parameters_in_results=False):
    self.parameters = dict(
        inner_kernel=inner_kernel,
        store_parameters_in_results=store_parameters_in_results)


class FakeInnerOld(object):

  def __init__(self):
    self.parameters = {}


class FakeInnerNew(object):

  def __init__(self, store_parameters_in_results=False):
    self.parameters = dict(
        store_parameters_in_results=store_parameters_in_results)


class FakeInnerNoParameters(object):
  pass


class EnableStoreParametersInResultsTest(test_util.TestCase):

  @parameterized.parameters(FakeInnerOld(),
                            FakeInnerNew(),
                            FakeWrapperOld(FakeInnerOld()),
                            FakeWrapperOld(FakeInnerNew()),
                            FakeWrapperNew(FakeInnerOld()),
                            FakeWrapperNew(FakeInnerNew()),
                            FakeWrapperOld(FakeWrapperOld(FakeInnerOld())),
                            FakeWrapperOld(FakeWrapperOld(FakeInnerNew())),
                            FakeWrapperOld(FakeWrapperNew(FakeInnerOld())),
                            FakeWrapperOld(FakeWrapperNew(FakeInnerNew())),
                            FakeWrapperNew(FakeWrapperOld(FakeInnerOld())),
                            FakeWrapperNew(FakeWrapperOld(FakeInnerNew())),
                            FakeWrapperNew(FakeWrapperNew(FakeInnerOld())),
                            FakeWrapperNew(FakeWrapperNew(FakeInnerNew())),
                           )
  def testAllCases(self, kernel):
    new_kernel = util.enable_store_parameters_in_results(kernel)

    flat_kernel = [kernel]
    while 'inner_kernel' in kernel.parameters:
      kernel = kernel.parameters['inner_kernel']
      flat_kernel.append(kernel)

    flat_new_kernel = [new_kernel]
    while 'inner_kernel' in new_kernel.parameters:
      new_kernel = new_kernel.parameters['inner_kernel']
      flat_new_kernel.append(new_kernel)

    self.assertEqual(len(flat_kernel), len(flat_new_kernel))
    for kernel, new_kernel in zip(flat_kernel, flat_new_kernel):
      self.assertIsNot(kernel, new_kernel)
      self.assertIs(type(kernel), type(new_kernel))
      if 'store_parameters_in_results' in new_kernel.parameters:
        self.assertTrue(new_kernel.parameters['store_parameters_in_results'])

  def testNoParameters(self):
    kernel = FakeInnerNoParameters()
    new_kernel = util.enable_store_parameters_in_results(kernel)
    self.assertIs(kernel, new_kernel)


class TensorConvertible(object):
  pass


tf.register_tensor_conversion_function(
    TensorConvertible, conversion_func=lambda *args: tf.constant(0))


class SimpleTensorWarningTest(test_util.TestCase):

  # We must defer creating the TF objects until the body of the test.
  # pylint: disable=unnecessary-lambda
  @parameterized.parameters([lambda: tf.Variable(0)],
                            [lambda: tf.Variable(0)],
                            [lambda: TensorConvertible()])
  @test_util.disable_test_for_backend(disable_numpy=True, disable_jax=True,
                                      reason='Variable/DeferredTensor')
  def testWarn(self, tensor_callable):
    tensor = tensor_callable()
    warnings.simplefilter('always')
    with warnings.catch_warnings(record=True) as triggered:
      util.warn_if_parameters_are_not_simple_tensors({'a': tensor})
    self.assertTrue(
        any('Please consult the docstring' in str(warning.message)
            for warning in triggered))

  @parameterized.parameters(
      [lambda: 1.], [lambda: np.array(1.)], [lambda: tf.constant(1.)])
  def testNoWarn(self, tensor_callable):
    tensor = tensor_callable()
    warnings.simplefilter('always')
    with warnings.catch_warnings(record=True) as triggered:
      util.warn_if_parameters_are_not_simple_tensors({'a': tensor})
    self.assertFalse(
        any('Please consult the docstring' in str(warning.message)
            for warning in triggered))


if __name__ == '__main__':
  test_util.main()
