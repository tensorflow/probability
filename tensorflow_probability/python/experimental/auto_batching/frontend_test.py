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
"""Tests for the AutoGraph frontend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
from absl import logging
import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.auto_batching import frontend
from tensorflow_probability.python.experimental.auto_batching import instructions
from tensorflow_probability.python.experimental.auto_batching import numpy_backend
from tensorflow_probability.python.experimental.auto_batching import tf_backend
from tensorflow_probability.python.internal import test_util

TF_BACKEND = tf_backend.TensorFlowBackend()

NP_BACKEND = numpy_backend.NumpyBackend()


# Eensy weensy test function
def fibonacci(n):
  if n <= 1:
    return 1
  else:
    left = fibonacci(n - 2)
    right = fibonacci(n - 1)
    return left + right


@test_util.test_all_tf_execution_regimes
class AutoGraphFrontendTest(test_util.TestCase):

  def testFibonacci(self):
    self.assertEqual(1, fibonacci(0))
    self.assertEqual(1, fibonacci(1))
    self.assertEqual(2, fibonacci(2))
    self.assertEqual(3, fibonacci(3))
    self.assertEqual(5, fibonacci(4))
    self.assertEqual(8, fibonacci(5))
    self.assertEqual(13, fibonacci(6))
    self.assertEqual(21, fibonacci(7))
    self.assertEqual(34, fibonacci(8))
    self.assertEqual(55, fibonacci(9))

  def testFibonacciNumpy(self):
    batch_fibo = frontend.Context().batch_uncurried(
        fibonacci,
        lambda *args: instructions.TensorType(np.int64, ()))
    self.assertEqual(
        [13, 21, 34, 55],
        list(batch_fibo(np.array([6, 7, 8, 9], dtype=np.int64),
                        max_stack_depth=15, backend=NP_BACKEND)))

  def testFibonacciNumpyStackless(self):
    if not tf.executing_eagerly():
      return
    batch_fibo = frontend.Context().batch_uncurried(
        fibonacci,
        lambda *args: instructions.TensorType(np.int64, ()))
    self.assertEqual(
        [3, 21, 5, 8],
        list(batch_fibo(np.array([3, 7, 4, 5], dtype=np.int64),
                        max_stack_depth=15, backend=NP_BACKEND,
                        stackless=True)))

  def testEvenOddWithContext(self):
    def pred_type(_):
      return instructions.TensorType(np.int32, ())
    ab = frontend.Context()

    @ab.batch(type_inference=pred_type)
    def even(n):
      if n <= 0:
        return True
      else:
        return odd(n - 1)

    @ab.batch(type_inference=pred_type)
    def odd(n):
      if n <= 0:
        return False
      else:
        return even(n - 1)

    inputs = np.array([5, 6, 8, 9], dtype=np.int32)
    outputs = np.array([False, True, True, False], dtype=np.bool)
    # pylint: disable=unexpected-keyword-arg
    # The `max_stack_depth` and `backend` keyword arguments to `even`
    # are introduced by the `@ab.batch` decorator, confusing pylint.
    self.assertAllEqual(
        outputs,
        self.evaluate(even(inputs, max_stack_depth=15, backend=TF_BACKEND)))

  def testSyntheticMultipleValueReturns(self):
    def id_type(args):
      return args[0]
    def function(x):
      a, b, c = [1, 2, 3]
      return x + a + b + c
    batched = frontend.Context().batch_uncurried(function, id_type)
    self.assertEqual(
        [12, 13, 14],
        list(batched(np.array([6, 7, 8], dtype=np.int64),
                     max_stack_depth=15, backend=NP_BACKEND)))

  def testNestedMultipleValueReturns(self):
    def my_primop():
      return [([1, 2], 3), 4]
    def id_type(args):
      return args[0]
    def function(x):
      [(a, b), c], d = my_primop()
      return x + a + b + c + d
    batched = frontend.Context().batch_uncurried(function, id_type)
    self.assertEqual(
        [16, 17, 18],
        list(batched(np.array([6, 7, 8], dtype=np.int64),
                     max_stack_depth=15, backend=NP_BACKEND)))

  def testNamedTuples(self):
    # Should be destructurable, and should be conserved
    # - on input to the auto-batched program
    # - on input to primops or functions
    # - on output from primops or functions, and
    # - on output from the auto-batched program.
    my_tuple = collections.namedtuple('my_tuple', ['x', 'y'])
    def my_primop(thing):
      return my_tuple(thing.x + 1, thing.y + 2)
    def id_type(args):
      return args[0]
    def function(x):
      thing1 = my_primop(x)
      thing2 = my_primop(thing1)
      return thing2
    def caller(x):
      thing1 = function(x)
      thing2 = function(thing1)
      return thing2
    ctx = frontend.Context()
    ctx.batch_uncurried(function, id_type)
    batched = ctx.batch_uncurried(caller, id_type)
    input_ = my_tuple(np.array([6, 7, 8], dtype=np.int64),
                      np.array([60, 70, 80], dtype=np.int64))
    output = my_tuple(np.array([10, 11, 12], dtype=np.int64),
                      np.array([68, 78, 88], dtype=np.int64))
    result = batched(input_, max_stack_depth=15, backend=NP_BACKEND)
    self.assertEqual(type(output), type(result))
    self.assertAllEqual(output, result)

  def testDisablingAutoBatching(self):
    execution_counter_box = [0]
    def count_executions(x):
      # The autobatching system will unconditionally run this exactly thrice per
      # occurrence in the source: once to infer the type, once to prove that
      # type inference stabilized, and once during graph construction.
      execution_counter_box[0] += 1
      return x
    def id_type(args):
      return args[0]
    def function(x):
      true = True
      if true:
        return count_executions(x)
      else:
        return count_executions(x)
    batched = frontend.Context().batch_uncurried(function, id_type)

    # Running the original function should increment the box once
    function(np.array([4]))
    self.assertEqual(execution_counter_box[0], 1)

    execution_counter_box[0] = 0
    if tf.executing_eagerly():
      # Running the batched version in eager mode should increment the box five
      # times (twice per static occurrence and once for the time it actually
      # executes)
      expected_execution_count = 5
    else:
      # Running the batched version in graph mode should increment the box six
      # times (thrice per occurrence)
      expected_execution_count = 6
    batched(np.array([4, 5, 6, 7, 8]))
    self.assertEqual(execution_counter_box[0], expected_execution_count)

    # Running the batched version in dry-run mode should increment the box once,
    # because that should mimic the original function.
    execution_counter_box[0] = 0
    batched(np.array([4]), dry_run=True)
    self.assertEqual(execution_counter_box[0], 1)

  def testDisablingAutoBatchingNested(self):
    execution_counter_box = [0]
    def count_executions(x):
      # The autobatching system will unconditionally run this exactly thrice per
      # occurrence in the source: once to infer the type, once to prove that
      # type inference stabilized, and once during graph construction.
      execution_counter_box[0] += 1
      return x
    def id_type(args):
      return args[0]
    ctx = frontend.Context()

    @ctx.batch(type_inference=id_type)
    def function(x):
      true = True
      if true:
        return count_executions(x)
      else:
        return count_executions(x)

    @ctx.batch(type_inference=id_type)
    def caller(x):
      return function(x)

    if tf.executing_eagerly():
      # Running the batched version in eager mode should increment the box five
      # times (twice per static occurrence and once for the time it actually
      # executes)
      expected_execution_count = 5
    else:
      # Running the batched version in graph mode should increment the box six
      # times (thrice per occurrence)
      expected_execution_count = 6
    caller(np.array([4, 5, 6, 7, 8]))
    self.assertEqual(execution_counter_box[0], expected_execution_count)

    # Running the batched version in dry-run mode should increment the box once,
    # because that should mimic the original function.
    execution_counter_box[0] = 0
    # pylint: disable=unexpected-keyword-arg
    # The `dry_run` keyword argument to `caller` is introduced by the
    # `@ctx.batch` decorator, confusing pylint.
    caller(np.array([4]), dry_run=True)
    self.assertEqual(execution_counter_box[0], 1)

  def testDryRunIf(self):
    def id_type(args):
      return args[0]
    truthy = frontend.truthy
    def batch_abs(x):
      if truthy(x > 0):
        return x
      else:
        return -x
    batched = frontend.Context().batch_uncurried(batch_abs, id_type)
    inputs = np.array([12, -13, 14], dtype=np.int64)
    self.assertAllEqual(
        [12, 13, 14],
        self.evaluate(batched(inputs, max_stack_depth=15, backend=TF_BACKEND)))

    # Note: trying to dry-run control flow will only work in Eager mode, because
    # Graph-mode Tensors cannot be used as `if` conditions at all.
    if tf.executing_eagerly():
      self.assertEqual([12], self.evaluate(
          batched(tf.constant([12]), dry_run=True)))
      self.assertEqual([13], self.evaluate(
          batched(tf.constant([-13]), dry_run=True)))

  def testConsumeEmitMultipleValues(self):
    def dup_type(args):
      return args[0]
    def function(inp):
      x, y = inp
      return x + 1, y + 2
    batched = frontend.Context().batch_uncurried(function, dup_type)
    inputs = (np.array([12, -13, 14], dtype=np.int32),
              np.array([[4, 3], [3, 2], [2, 1]], dtype=np.int32))
    expected_outputs = ([13, -12, 15], [[6, 5], [5, 4], [4, 3]])
    got_outputs = self.evaluate(
        batched(inputs, max_stack_depth=15, backend=TF_BACKEND))
    self.assertEqual(len(expected_outputs), len(got_outputs))
    for exp, got in zip(expected_outputs, got_outputs):
      self.assertAllEqual(exp, got)

  def testConsumeEmitMultipleValuesNested(self):
    def dup_type(args):
      return args[0]
    ctx = frontend.Context()
    @ctx.batch(type_inference=dup_type)
    def function(inp):
      x, y = inp
      return x + 1, y + 2
    @ctx.batch(type_inference=dup_type)
    def caller(inp):
      ans1, ans2 = function(inp)
      return ans1, ans2
    inputs = (np.array([12, -13, 14], dtype=np.int32),
              np.array([[4, 8], [3, 6], [2, 4]], dtype=np.int32))
    expected_outputs = ([13, -12, 15], [[6, 10], [5, 8], [4, 6]])
    # pylint: disable=unexpected-keyword-arg
    # The `max_stack_depth` and `backend` keyword arguments to `caller`
    # are introduced by the `@ctx.batch` decorator, confusing pylint.
    got_outputs = self.evaluate(
        caller(inputs, max_stack_depth=15, backend=TF_BACKEND))
    self.assertEqual(len(expected_outputs), len(got_outputs))
    for exp, got in zip(expected_outputs, got_outputs):
      self.assertAllEqual(exp, got)

  def testRestructureOnFunctionReturn(self):
    def quad_type(args):
      return ((args[0], args[0]), (args[0], args[0]))
    ctx = frontend.Context()
    @ctx.batch(type_inference=quad_type)
    def function(x):
      left = x + 1, x + 2
      right_1 = x + 3
      right_2 = x + 4
      return left, (right_1, right_2)
    def id_type(args):
      return args[0]
    @ctx.batch(type_inference=id_type)
    def caller(x):
      (left_1, left_2), right = function(x)
      right_1, right_2 = right
      return left_1 + left_2 + right_1 + right_2
    inputs = np.array([12, -13, 14], dtype=np.int32)
    # pylint: disable=unexpected-keyword-arg
    # The `max_stack_depth` and `backend` keyword arguments to `caller`
    # are introduced by the `@ctx.batch` decorator, confusing pylint.
    self.assertAllEqual(
        [58, -42, 66],
        self.evaluate(caller(inputs, max_stack_depth=5, backend=TF_BACKEND)))

  def testBatchDimensionSensitivePrimop(self):
    def batchwise_reduce_sum(x):
      return tf.reduce_sum(input_tensor=x, axis=tf.range(1, tf.rank(x)))

    def my_type(args):
      return instructions.TensorType(args[0].dtype, ())
    def function(x):
      y = batchwise_reduce_sum(x)
      return y + y
    batched = frontend.Context().batch_uncurried(function, my_type)
    inputs = np.array([[12, 13], [-13, 10], [14, 1]], dtype=np.int32)
    self.assertAllEqual(
        [50, -6, 30],
        self.evaluate(batched(inputs, max_stack_depth=5, backend=TF_BACKEND)))

  def testUseDummyVariableForPrimop(self):
    def my_type(args):
      return args[0], args[0]
    def callee(x):
      return x, x + 1
    def function(x):
      _, y = callee(x)
      return y, y + 1
    ctx = frontend.Context()
    batched = ctx.batch_uncurried(function, my_type)
    inputs = np.array([12, 13, 14], dtype=np.int32)
    self.assertAllEqual(
        [[13, 14, 15], [14, 15, 16]],
        self.evaluate(batched(inputs, max_stack_depth=5, backend=TF_BACKEND)))

  def testUseDummyVariableForCall(self):
    def my_type(args):
      return args[0], args[0]
    def callee(x):
      return x, x + 1
    def function(x):
      _, y = callee(x)
      return y, y + 1
    ctx = frontend.Context()
    ctx.batch_uncurried(callee, my_type)
    batched = ctx.batch_uncurried(function, my_type)
    inputs = np.array([12, 13, 14], dtype=np.int32)
    self.assertAllEqual(
        [[13, 14, 15], [14, 15, 16]],
        self.evaluate(batched(inputs, max_stack_depth=5, backend=TF_BACKEND)))

  def testBroadcastInputsToBatchSize(self):
    # The desired batch size is inferred from the inputs.  This test is checking
    # that the system supports broadcasting (other) inputs across the batch
    # dimension, which requires not accidentally inferring a batch size of 1.
    def my_type(args):
      return args[0]
    def function(a, b, c, d, e, f, g):
      return a + b + c + d + e + f + g
    a_in = np.array([12, 13, 14], dtype=np.int32)
    b_in = np.array([2], dtype=np.int32)
    c_in = np.array([3], dtype=np.int32)
    d_in = np.array([4], dtype=np.int32)
    ctx = frontend.Context()
    batched = ctx.batch_uncurried(function, my_type)
    # Repeat several times, because the buggy behavior this is catching
    # depended on Python dict order traversal.
    for _ in range(10):
      self.assertAllEqual(
          [39, 40, 41],
          batched(a_in, b_in, c_in, d_in, 5, 6, 7,
                  max_stack_depth=5, backend=NP_BACKEND))

  def testCompileCache(self):
    ctx = frontend.Context()
    def my_type(args):
      return args[0]
    def function(x):
      return x + x
    ctx.batch_uncurried(function, my_type)
    sig = [instructions.TensorType(np.int64, ())]
    prog1 = ctx.program_compiled('function', sig, NP_BACKEND)
    prog2 = ctx.program_compiled('function', sig, NP_BACKEND)
    self.assertEqual(prog1, prog2)
    sig2 = [instructions.TensorType(np.int32, ())]
    prog3 = ctx.program_compiled('function', sig2, NP_BACKEND)
    self.assertNotEqual(prog1, prog3)

  def testLoweringCache(self):
    ctx = frontend.Context()
    def my_type(args):
      return args[0]
    def function(x):
      return x + x
    ctx.batch_uncurried(function, my_type)
    sig = [instructions.TensorType(np.int64, ())]
    prog1 = ctx.program_lowered('function', sig, NP_BACKEND)
    prog2 = ctx.program_lowered('function', sig, NP_BACKEND)
    self.assertEqual(prog1, prog2)
    sig2 = [instructions.TensorType(np.int32, ())]
    prog3 = ctx.program_lowered('function', sig2, NP_BACKEND)
    self.assertNotEqual(prog1, prog3)

  def testSelfTailCallOptimization(self):
    def my_type(args):
      return args[0]
    ab = frontend.Context()

    @ab.batch(type_inference=my_type)
    def iota_sum(n, acc):
      if n <= 0:
        return acc
      else:
        return iota_sum(n - 1, acc + n)

    inputs = [np.array([12, -13, 100], dtype=np.int32),
              np.array([0, 0, 0], dtype=np.int32)]
    result = ab.lowered_for_args('iota_sum', inputs, backend=TF_BACKEND)
    # Check no pushes
    for i in range(result.graph.exit_index()):
      block = result.graph.block(i)
      if isinstance(block.terminator, instructions.PushGotoOp):
        assert False, 'iota_sum is tail-recursive: should not push the PC'
    for var, alloc in result.var_alloc.items():
      if alloc == instructions.VariableAllocation.FULL:
        if var != instructions.pc_var:
          assert False, 'iota_sum should not push any data'
    # pylint: disable=unexpected-keyword-arg
    # The `max_stack_depth` and `backend` keyword arguments to `iota_sum`
    # are introduced by the `@ctx.batch` decorator, confusing pylint.
    self.assertAllEqual(
        [78, 0, 5050],
        self.evaluate(iota_sum(*inputs, max_stack_depth=3, backend=TF_BACKEND)))

  def testParameterSwapping(self):
    # The thing that's interesting about this test program is that it contains a
    # self-call that writes variables to themselves, but misaligned.  To
    # implement this correctly, it is necessary to introduce a temporary
    # variable so that `a` and `b` can be swapped (b/135275883).
    def trace(x):
      logging.info(x)
      return x
    ab = frontend.Context()

    def gcd_type(args):
      return args[0]

    @ab.batch(type_inference=gcd_type)
    def gcd(a, b):
      if trace(a) == 0:
        return b
      elif a <= b:
        return gcd(a, b - a)
      else:
        # TODO(b/135275883): Remove these temporaries and check that this
        # program still works.
        a_tmp = a
        b_tmp = b
        return gcd(b_tmp, a_tmp)

    input_a = np.array([7, 12, 49], dtype=np.int32)
    input_b = np.array([9, 9, 56], dtype=np.int32)
    # pylint: disable=unexpected-keyword-arg
    # The `max_stack_depth` and `backend` keyword arguments to `gcd`
    # are introduced by the `@ab.batch` decorator, confusing pylint.
    self.assertAllEqual(
        [1, 3, 7],
        gcd(input_a, input_b, max_stack_depth=3, backend=NP_BACKEND))

  def testSyncingAsExpected(self):
    execution_counter_box = [0]
    def count_executions(x):
      execution_counter_box[0] += 1
      return x
    # Instrumented test program
    def fibonacci_inst(n):
      if n <= 1:
        return 1
      else:
        left = fibonacci_inst(n - 2)
        right = fibonacci_inst(n - 1)
        return count_executions(left + right)
    batch_fibo = frontend.Context().batch_uncurried(
        fibonacci_inst,
        lambda *args: instructions.TensorType(np.int64, ()))
    self.assertEqual(
        [3, 21, 5, 8],
        list(batch_fibo(np.array([3, 7, 4, 5], dtype=np.int64),
                        max_stack_depth=15, backend=NP_BACKEND)))
    # Expect 22 executions
    # - 2 for type checking, plus
    # - 20 because that's how many times 1 needs to be added to
    #   1 to get 21.
    self.assertEqual(22, execution_counter_box[0])

  def testCalmerAnf(self):
    # This test has two syntactic features that used to ANF badly:
    # - Literal True `if` argument used to not get replaced
    # - Trying to use a reference in a function call, like `np.array(foo)`, used
    #   to replace `np.array`, causing problems.
    def int_type(args):
      return args[0]
    ctx = frontend.Context()
    @ctx.batch(type_inference=int_type)
    def function(x):
      # pylint: disable=using-constant-test
      if True:
        return x + np.array(1)
      else:
        return x + np.array(1)

    inputs = np.array([1, 5, 60, 3, 7], dtype=np.int32)
    outputs = np.array([2, 6, 61, 4, 8], dtype=np.int32)
    # pylint: disable=unexpected-keyword-arg
    # The `max_stack_depth` and `backend` keyword arguments to `function`
    # are introduced by the `@ctx.batch` decorator, confusing pylint.
    self.assertAllEqual(
        outputs,
        function(inputs, max_stack_depth=3, backend=NP_BACKEND))

  def testFieldReference(self):
    # Test that we can extract fields from named tuples, whether those tuples
    # were defined inside or outside the autobatched function.
    foo_cls = collections.namedtuple('foo_cls', ['bar', 'baz'])
    foo1 = foo_cls(5, 20)
    def int_type(args):
      return args[0]
    ctx = frontend.Context()
    @ctx.batch(type_inference=int_type)
    def function(x):
      y = x + foo1.bar
      foo2 = foo_cls(y, y + 5)
      return foo2.baz

    inputs = np.array([1, 5, 60, 3, 7], dtype=np.int32)
    outputs = np.array([11, 15, 70, 13, 17], dtype=np.int32)
    # pylint: disable=unexpected-keyword-arg
    # The `max_stack_depth` and `backend` keyword arguments to `function`
    # are introduced by the `@ctx.batch` decorator, confusing pylint.
    self.assertAllEqual(
        outputs,
        function(inputs, max_stack_depth=3, backend=NP_BACKEND))


class _TestHidingTFBatchSize(object):

  def _build_tensor(self, ndarray):
    if self.use_static_batch_size:
      shape = ndarray.shape
    else:
      shape = [None] + list(ndarray.shape[1:])
    return tf1.placeholder_with_default(input=ndarray, shape=shape)

  def _check_batch_size(self, tensor, expected):
    if self.use_static_batch_size or tf.executing_eagerly():
      self.assertEqual(expected, tensor.shape[0])
    # TODO(b/259749542): Re-enable once we fix why `tensor` is `None`.
    # else:
    #   self.assertEqual(None, tensor.shape[0].value)

  def testFibonacciTF(self):
    batch_fibo = frontend.Context().batch_uncurried(
        fibonacci,
        lambda *args: instructions.TensorType(np.int64, ()))
    input_2 = self._build_tensor(np.array([6, 7, 8], dtype=np.int64))
    answer = batch_fibo(input_2, max_stack_depth=15, backend=TF_BACKEND)
    self._check_batch_size(answer, 3)
    self.assertAllEqual([13, 21, 34], self.evaluate(answer))

  def testOneArmedAndNestedIf(self):
    def int_type(_):
      return instructions.TensorType(np.int32, ())
    ctx = frontend.Context()
    @ctx.batch(type_inference=int_type)
    def function(x):
      ans = 1
      if x > 4:
        if x > 10:
          ans = 5
        else:
          ans = 3
      return ans

    inputs = self._build_tensor(np.array([1, 5, 60, 3, 7], dtype=np.int32))
    outputs = np.array([1, 3, 5, 1, 3], dtype=np.int32)
    # pylint: disable=unexpected-keyword-arg
    # The `max_stack_depth` and `backend` keyword arguments to `function`
    # are introduced by the `@ctx.batch` decorator, confusing pylint.
    answer = function(inputs, max_stack_depth=15, backend=TF_BACKEND)
    self._check_batch_size(answer, 5)
    self.assertAllEqual(outputs, self.evaluate(answer))

  def testCallingUntaggedFunctions(self):
    def id_type(args):
      return args[0]
    def subroutine(batched_x):
      return tf.reduce_sum(input_tensor=batched_x, axis=-1, keepdims=True)

    ctx = frontend.Context()
    @ctx.batch(type_inference=id_type)
    def function(x):
      y = subroutine(x)
      return x + y

    inputs = self._build_tensor(np.array([[1, 2],
                                          [5, 6],
                                          [60, 61]], dtype=np.int32))
    outputs = np.array([[4, 5],
                        [16, 17],
                        [181, 182]], dtype=np.int32)
    # pylint: disable=unexpected-keyword-arg
    # The `max_stack_depth` and `backend` keyword arguments to `function`
    # are introduced by the `@ctx.batch` decorator, confusing pylint.
    answer = function(inputs, max_stack_depth=15, backend=TF_BACKEND)
    self._check_batch_size(answer, 3)
    self.assertAllEqual(outputs, self.evaluate(answer))

  def testReferToEnclosingScope(self):
    an_object = 'four'
    def an_op_on_objects(obj):
      return len(obj)
    def id_type(args):
      return args[0]
    def an_autobatch_function(x):
      # Expect the object to be pulled in from the enclosing scope, not
      # converted to an auto-batch variable.
      return x + an_op_on_objects(an_object)
    batched = frontend.Context().batch_uncurried(an_autobatch_function, id_type)
    inputs = self._build_tensor(np.array([12, -13, 14], dtype=np.int32))
    answer = batched(inputs, max_stack_depth=15, backend=TF_BACKEND)
    self._check_batch_size(answer, 3)
    self.assertAllEqual([16, -9, 18], self.evaluate(answer))


@test_util.test_all_tf_execution_regimes
class TestTFStaticBatchSize(test_util.TestCase, _TestHidingTFBatchSize):
  use_static_batch_size = True


@test_util.test_all_tf_execution_regimes
class TestTFDynamicBatchSize(test_util.TestCase, _TestHidingTFBatchSize):
  use_static_batch_size = False

if __name__ == '__main__':
  tf.test.main()
