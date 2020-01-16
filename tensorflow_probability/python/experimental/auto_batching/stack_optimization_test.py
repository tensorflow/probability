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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.auto_batching import stack_optimization as stack
from tensorflow_probability.python.experimental.auto_batching import test_programs
from tensorflow_probability.python.internal import test_util as tfp_test_util


class StackOptimizationTest(tfp_test_util.TestCase):

  def testPopPushFusionPrettyPrint(self):
    # Testing two things: That pop-push fusion does the expected thing, and that
    # push-skipping PrimOps print with exclamation marks as expected.  This test
    # is likely to be brittle, and may want to be rearranged later.
    prog = test_programs.fibonacci_program()
    fused = stack.fuse_pop_push(prog)
    self.verify_program_pretty_print(fib_fused_pretty, fused)

  def verify_program_pretty_print(self, expected_text, program, **kwargs):
    actual_text = str(program.__str__(**kwargs))
    if expected_text != actual_text:
      print(expected_text)
      print(actual_text)
    self.assertEqual(expected_text, actual_text)

fib_fused_pretty = """
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
    nm1 = prim(n) by
      lambda n: n - 1),                      #   nm1 = n - 1
    n = prim(nm1) by
      def assign(*x):
        return x
    pop(cond, nm1)
    push_goto(recur2, enter_fib)
  recur2:
    fibm1 = prim(ans) by
      def assign(*x):
        return x
    nm2 = prim(n) by
      lambda n: n - 2),                      #   nm2 = n - 2
    n! = prim(nm2) by
      def assign(*x):
        return x
    pop(ans, nm2)
    push_goto(recur3, enter_fib)
  recur3:
    fibm2 = prim(ans) by
      def assign(*x):
        return x
    ans! = prim(fibm1, fibm2) by
      lambda x, y: x + y),                   #   ans = fibm1 + fibm2
    pop(fibm1, fibm2)
    indirect_jump
  finish:
    ans = prim() by
      lambda : 1),                           #   ans = 1
    pop(n, cond)
    indirect_jump
""".strip()

if __name__ == '__main__':
  tf.test.main()
