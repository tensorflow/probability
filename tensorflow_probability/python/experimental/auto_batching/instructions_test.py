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
"""Tests of the instruction language (and definitional interpreter)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.auto_batching import instructions
from tensorflow_probability.python.experimental.auto_batching import test_programs
from tensorflow_probability.python.internal import test_util as tfp_test_util


class InstructionsTest(tfp_test_util.TestCase):

  def testConstant(self):
    # This program always returns 2.
    def interpret(n):
      return instructions.interpret(test_programs.constant_program(), n)

    self.assertEqual(2, interpret(5))

  def testSingleIf(self):
    # This program returns n > 1 ? 2 : 0.
    def interpret(n):
      return instructions.interpret(test_programs.single_if_program(), n)

    self.assertEqual(0, interpret(1))
    self.assertEqual(2, interpret(3))

  def testPatternVariables(self):
    def interpret(n):
      return instructions.interpret(
          test_programs.synthetic_pattern_variable_program(), n)
    self.assertEqual((7, 8), interpret(5))

  def testFibonacci(self):
    # This program returns fib(n), where fib(0) = fib(1) = 1.
    def interpret(n):
      return instructions.interpret(test_programs.fibonacci_program(), n)

    self.assertEqual(8, interpret(5))

  def testFibonacciFunctions(self):
    # This program returns fib(n), where fib(0) = fib(1) = 1.
    def interpret(n):
      return instructions.interpret(
          test_programs.fibonacci_function_calls(), n)

    self.assertEqual(8, interpret(5))

  def testPatternMatching(self):
    self.assertEqual(
        ((1, 3), 4, (5, 6)),
        instructions.interpret(test_programs.synthetic_pattern_program()))


fibonacci_pretty_print = """
__program_counter__ :: full, Type(tensors=TensorType(int32()))
ans :: full, Type(tensors=TensorType(int64()))
cond :: full, Type(tensors=TensorType(bool()))
fibm1 :: full, Type(tensors=TensorType(int64()))
fibm2 :: full, Type(tensors=TensorType(int64()))
n :: full, Type(tensors=TensorType(int64()))
nm1 :: full, Type(tensors=TensorType(int64()))
nm2 :: full, Type(tensors=TensorType(int64()))

def main(n):
  returns ans
  entry:
    push_goto(exit, enter_fib)
  enter_fib:
    cond = prim(n) by
      lambda n: n > 1),                      # cond = n > 1
    branch(cond, recur1, finish)
  recur1:
    pop(cond)
    nm1 = prim(n) by
      lambda n: n - 1),                      #   nm1 = n - 1
    n = prim(nm1) by
      def assign(*x):
        return x
    pop(nm1)
    push_goto(recur2, enter_fib)
  recur2:
    fibm1 = prim(ans) by
      def assign(*x):
        return x
    pop(ans)
    nm2 = prim(n) by
      lambda n: n - 2),                      #   nm2 = n - 2
    pop(n)
    n = prim(nm2) by
      def assign(*x):
        return x
    pop(nm2)
    push_goto(recur3, enter_fib)
  recur3:
    fibm2 = prim(ans) by
      def assign(*x):
        return x
    pop(ans)
    ans = prim(fibm1, fibm2) by
      lambda x, y: x + y),                   #   ans = fibm1 + fibm2
    pop(fibm1, fibm2)
    indirect_jump
  finish:
    pop(n, cond)
    ans = prim() by
      lambda : 1),                           #   ans = 1
    indirect_jump
""".strip()

fibonacci_functions_pretty_print = """
__program_counter__ :: full, Type(tensors=TensorType(int32()))
ans :: full, Type(tensors=TensorType(int64()))
cond :: full, Type(tensors=TensorType(bool()))
fibm1 :: full, Type(tensors=TensorType(int64()))
fibm2 :: full, Type(tensors=TensorType(int64()))
n :: full, Type(tensors=TensorType(int64()))
n1 :: full, Type(tensors=TensorType(int64()))
nm1 :: full, Type(tensors=TensorType(int64()))
nm2 :: full, Type(tensors=TensorType(int64()))

def fibonacci(n):
  returns ans
  enter_fib:
    cond = prim(n) by
      lambda n: n > 1),                      # cond = n > 1
    branch(cond, recur, finish)
  recur:
    nm1 = prim(n) by
      lambda n: n - 1),                      #   nm1 = n - 1
    fibm1 = call(fibonacci, nm1)
    nm2 = prim(n) by
      lambda n: n - 2),                      #   nm2 = n - 2
    fibm2 = call(fibonacci, nm2)
    ans = prim(fibm1, fibm2) by
      lambda x, y: x + y),                   #   ans = fibm1 + fibm2
    return
  finish:
    ans = prim() by
      lambda : 1),                           #   ans = 1
    return

def main(n1):
  returns ans
  main_entry:
    ans = call(fibonacci, n1)
    return
""".strip()

fibonacci_functions_narrow_pretty_print = """
__program_counter__ :: full, Type(tensors=TensorType(int32()))
ans :: full, Type(tensors=TensorType(int64()))
cond :: full, Type(tensors=TensorType(bool()))
fibm1 :: full, Type(tensors=TensorType(int64()))
fibm2 :: full, Type(tensors=TensorType(int64()))
n :: full, Type(tensors=TensorType(int64()))
n1 :: full, Type(tensors=TensorType(int64()))
nm1 :: full, Type(tensors=TensorType(int64()))
nm2 :: full, Type(tensors=TensorType(int64()))

def fibonacci(
  n):
  returns ans
  enter_fib:
    cond = prim(
      n) by
      lambda n: n > 1),                      # cond = n > 1
    branch(
      cond,
      recur,
      finish)
  recur:
    nm1 = prim(
      n) by
      lambda n: n - 1),                      #   nm1 = n - 1
    fibm1 = call(
      fibonacci,
      nm1)
    nm2 = prim(
      n) by
      lambda n: n - 2),                      #   nm2 = n - 2
    fibm2 = call(
      fibonacci,
      nm2)
    ans = prim(
      fibm1,
      fibm2) by
      lambda x, y: x + y),                   #   ans = fibm1 + fibm2
    return
  finish:
    ans = prim(
      ) by
      lambda : 1),                           #   ans = 1
    return

def main(n1):
  returns ans
  main_entry:
    ans = call(
      fibonacci,
      n1)
    return
""".strip()


class PrettyPrintingTest(tfp_test_util.TestCase):

  def verify_program_pretty_print(self, expected_text, program, **kwargs):
    actual_text = str(program.__str__(**kwargs))
    if expected_text != actual_text:
      print(expected_text)
      print(actual_text)
    self.assertEqual(expected_text, actual_text)

  def testFibonacci(self):
    self.verify_program_pretty_print(
        fibonacci_pretty_print, test_programs.fibonacci_program())

  def testFibonacciFunctions(self):
    self.verify_program_pretty_print(
        fibonacci_functions_pretty_print,
        test_programs.fibonacci_function_calls())

  def testFibonacciFunctionsNarrow(self):
    self.verify_program_pretty_print(
        fibonacci_functions_narrow_pretty_print,
        test_programs.fibonacci_function_calls(),
        width=16)

if __name__ == '__main__':
  tf.test.main()
