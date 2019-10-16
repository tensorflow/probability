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
"""Tests of the allocation strategy optimization pass."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.auto_batching import allocation_strategy
from tensorflow_probability.python.experimental.auto_batching import instructions as inst
from tensorflow_probability.python.experimental.auto_batching import test_programs
from tensorflow_probability.python.internal import test_util as tfp_test_util


def strip_pop_ops(program):
  # Why might this be useful?  Because a variable waiting to be popped by an
  # explicit PopOp registers as "live" to the liveness analysis, which causes it
  # to demand a heavier allocation strategy than it actually deserves.  For an
  # example of the difference this can make, compare the answers in
  # `testAllocatingIsEvenProgram` and `testAllocatingIsEvenProgramNoPops`.
  def walk_graph(graph):
    for i in range(graph.exit_index()):
      block = graph.block(i)
      block.instructions = [op for op in block.instructions
                            if not isinstance(op, inst.PopOp)]
  walk_graph(program.graph)
  for func in program.functions:
    walk_graph(func.graph)


class AllocationStrategyTest(tfp_test_util.TestCase):

  def assertAllocates(self, expected, prog):
    allocated = allocation_strategy.optimize(prog)
    self.assertEqual(expected, allocated.var_alloc)

  def testAllocatingConstantProgram(self):
    prog = test_programs.constant_program()
    answer = {inst.pc_var: inst.VariableAllocation.REGISTER,
              'answer': inst.VariableAllocation.REGISTER}
    self.assertAllocates(answer, prog)

  def testAllocatingIfProgram(self):
    prog = test_programs.single_if_program()
    answer = {inst.pc_var: inst.VariableAllocation.REGISTER,
              'answer': inst.VariableAllocation.REGISTER,
              'cond': inst.VariableAllocation.REGISTER,
              'input': inst.VariableAllocation.REGISTER}
    self.assertAllocates(answer, prog)

  def testAllocatingIsEvenProgram(self):
    prog = test_programs.is_even_function_calls()
    answer = {inst.pc_var: inst.VariableAllocation.FULL,
              'ans': inst.VariableAllocation.REGISTER,
              'cond': inst.VariableAllocation.REGISTER,
              'n': inst.VariableAllocation.REGISTER,
              'n1': inst.VariableAllocation.REGISTER,
              'nm1': inst.VariableAllocation.FULL}
    self.assertAllocates(answer, prog)

  def testAllocatingIsEvenProgramNoPops(self):
    prog = test_programs.is_even_function_calls()
    strip_pop_ops(prog)
    answer = {inst.pc_var: inst.VariableAllocation.FULL,
              'ans': inst.VariableAllocation.REGISTER,
              'cond': inst.VariableAllocation.REGISTER,
              'n': inst.VariableAllocation.REGISTER,
              'n1': inst.VariableAllocation.REGISTER,
              'nm1': inst.VariableAllocation.TEMPORARY}
    self.assertAllocates(answer, prog)

  def testAllocatingFibonacciProgram(self):
    prog = test_programs.fibonacci_function_calls()
    answer = {inst.pc_var: inst.VariableAllocation.FULL,
              'ans': inst.VariableAllocation.REGISTER,
              'cond': inst.VariableAllocation.REGISTER,
              'fibm1': inst.VariableAllocation.FULL,
              'fibm2': inst.VariableAllocation.TEMPORARY,
              'n': inst.VariableAllocation.FULL,
              'n1': inst.VariableAllocation.REGISTER,
              'nm1': inst.VariableAllocation.TEMPORARY,
              'nm2': inst.VariableAllocation.TEMPORARY}
    self.assertAllocates(answer, prog)

if __name__ == '__main__':
  tf.test.main()
