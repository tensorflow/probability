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

import numpy as np

from tensorflow_probability.python.experimental.auto_batching import instructions


def constant_program():
  """Constant program: 'ans=1; ans=2; return ans;'.

  Returns:
    program: `instructions.Program` which returns a constant value.
  """
  constant_block = instructions.Block(
      [
          instructions.prim_op([], "answer", lambda: 1),
          instructions.prim_op([], "answer", lambda: 2),
      ],
      instructions.halt_op())

  constant_vars = {
      "answer": instructions.single_type(np.int64, ()),
  }

  return instructions.Program(
      instructions.ControlFlowGraph([constant_block]), [],
      constant_vars, ["answer"], "answer")


def _strip_types(the_vars):
  for k in the_vars:
    if k != instructions.pc_var:
      the_vars[k] = instructions.Type(None)


def single_if_program():
  """Single if program: 'if (input > 1) ans = 2; else ans = 0; return ans;'.

  Returns:
    program: `instructions.Program` with a simple conditional.
  """
  entry = instructions.Block()
  then_ = instructions.Block()
  else_ = instructions.Block()
  entry.assign_instructions([
      instructions.prim_op(["input"], "cond", lambda n: n > 1),
      instructions.BranchOp("cond", then_, else_),
  ])
  then_.assign_instructions([
      instructions.prim_op([], "answer", lambda: 2),
      instructions.halt_op(),
  ])
  else_.assign_instructions([
      instructions.prim_op([], "answer", lambda: 0),
      instructions.halt_op(),
  ])

  single_if_blocks = [entry, then_, else_]
  # pylint: disable=bad-whitespace
  single_if_vars = {
      "input"               : instructions.single_type(np.int64, ()),
      "cond"                : instructions.single_type(np.bool_,  ()),
      "answer"              : instructions.single_type(np.int64, ()),
  }

  return instructions.Program(
      instructions.ControlFlowGraph(single_if_blocks), [],
      single_if_vars, ["input"], "answer")


def synthetic_pattern_program():
  """A program that tests pattern matching of `PrimOp` outputs.

  Returns:
    program: `instructions.Program`.
  """
  block = instructions.Block(
      [
          instructions.prim_op(
              [], ("one", ("five", "three")), lambda: (1, (2, 3))),
          instructions.prim_op(
              [], (("four", "five"), "six"), lambda: ((4, 5), 6)),
      ],
      instructions.halt_op())

  the_vars = {
      "one": instructions.single_type(np.int64, ()),
      "three": instructions.single_type(np.int64, ()),
      "four": instructions.single_type(np.int64, ()),
      "five": instructions.single_type(np.int64, ()),
      "six": instructions.single_type(np.int64, ()),
  }

  return instructions.Program(
      instructions.ControlFlowGraph([block]), [],
      the_vars, [], (("one", "three"), "four", ("five", "six")))


def synthetic_pattern_variable_program(include_types=True):
  """A program that tests product types.

  Args:
    include_types: If False, we omit types on the variables, requiring a type
        inference pass.

  Returns:
    program: `instructions.Program`.
  """
  block = instructions.Block(
      [
          instructions.prim_op(
              ["inp"], "many", lambda x: (x + 1, (x + 2, x + 3))),
          instructions.prim_op(["many"], ["one", "two"], lambda x: x),
      ],
      instructions.halt_op())

  leaf = instructions.TensorType(np.int64, ())
  the_vars = {
      "inp": instructions.Type(leaf),
      "many": instructions.Type((leaf, (leaf, leaf))),
      "one": instructions.Type(leaf),
      "two": instructions.Type((leaf, leaf)),
  }

  if not include_types:
    _strip_types(the_vars)
  return instructions.Program(
      instructions.ControlFlowGraph([block]), [],
      the_vars, ["inp"], "two")


def fibonacci_program():
  """More complicated, fibonacci program: computes fib(n): fib(0) = fib(1) = 1.

  Returns:
    program: Full-powered `instructions.Program` that computes fib(n).
  """
  entry = instructions.Block(name="entry")
  enter_fib = instructions.Block(name="enter_fib")
  recur1 = instructions.Block(name="recur1")
  recur2 = instructions.Block(name="recur2")
  recur3 = instructions.Block(name="recur3")
  finish = instructions.Block(name="finish")
  # pylint: disable=bad-whitespace
  entry.assign_instructions([
      instructions.PushGotoOp(instructions.halt(), enter_fib),
  ])
  # Definition of fibonacci function starts here
  enter_fib.assign_instructions([
      instructions.prim_op(
          ["n"], "cond",
          lambda n: n > 1),                      # cond = n > 1
      instructions.BranchOp(
          "cond", recur1, finish),               # if cond
  ])
  recur1.assign_instructions([
      instructions.PopOp(["cond"]),              #   done with cond now
      instructions.prim_op(
          ["n"], "nm1",
          lambda n: n - 1),                      #   nm1 = n - 1
      instructions.push_op(["nm1"], ["n"]),      #   fibm1 = fibonacci(nm1)
      instructions.PopOp(["nm1"]),               #     done with nm1
      instructions.PushGotoOp(recur2, enter_fib),
  ])
  recur2.assign_instructions([
      instructions.push_op(["ans"], ["fibm1"]),  #     ...
      instructions.PopOp(["ans"]),               #     pop callee's "ans"
      instructions.prim_op(
          ["n"], "nm2",
          lambda n: n - 2),                      #   nm2 = n - 2
      instructions.PopOp(["n"]),                 #   done with n
      instructions.push_op(["nm2"], ["n"]),      #   fibm2 = fibonacci(nm2)
      instructions.PopOp(["nm2"]),               #     done with nm2
      instructions.PushGotoOp(recur3, enter_fib),
  ])
  recur3.assign_instructions([
      instructions.push_op(["ans"], ["fibm2"]),  #     ...
      instructions.PopOp(["ans"]),               #     pop callee's "ans"
      instructions.prim_op(
          ["fibm1", "fibm2"], "ans",
          lambda x, y: x + y),                   #   ans = fibm1 + fibm2
      instructions.PopOp(["fibm1", "fibm2"]),    #   done with fibm1, fibm2
      instructions.IndirectGotoOp(),             #   return ans
  ])
  finish.assign_instructions([                   # else:
      instructions.PopOp(["n", "cond"]),         #   done with n, cond
      instructions.prim_op(
          [], "ans",
          lambda : 1),                           #   ans = 1
      instructions.IndirectGotoOp(),             #   return ans
  ])

  fibonacci_blocks = [
      entry,
      enter_fib,
      recur1, recur2, recur3,
      finish
  ]

  # pylint: disable=bad-whitespace
  fibonacci_vars = {
      "n"                   : instructions.single_type(np.int64, ()),
      "cond"                : instructions.single_type(np.bool_,  ()),
      "nm1"                 : instructions.single_type(np.int64, ()),
      "fibm1"               : instructions.single_type(np.int64, ()),
      "nm2"                 : instructions.single_type(np.int64, ()),
      "fibm2"               : instructions.single_type(np.int64, ()),
      "ans"                 : instructions.single_type(np.int64, ()),
  }

  return instructions.Program(
      instructions.ControlFlowGraph(fibonacci_blocks), [],
      fibonacci_vars, ["n"], "ans")


def is_even_function_calls(include_types=True, dtype=np.int64):
  """The is-even program, via "even-odd" recursion.

  Computes True if the input is even, False if the input is odd, by a pair of
  mutually recursive functions is_even and is_odd, which return True and False
  respectively for <1-valued inputs.

  Tests out mutual recursion.

  Args:
    include_types: If False, we omit types on the variables, requiring a type
        inference pass.
    dtype: The dtype to use for `n`-like internal state variables.

  Returns:
    program: Full-powered `instructions.Program` that computes is_even(n).
  """
  def pred_type(t):
    return instructions.TensorType(np.bool_, t[0].shape)
  # Forward declaration of is_odd.
  is_odd_func = instructions.Function(None, ["n"], "ans", pred_type)

  enter_is_even = instructions.Block()
  finish_is_even = instructions.Block()
  recur_is_even = instructions.Block()
  is_even_func = instructions.Function(None, ["n"], "ans", pred_type)
  # pylint: disable=bad-whitespace
  # Definition of is_even function
  enter_is_even.assign_instructions([
      instructions.prim_op(
          ["n"], "cond", lambda n: n < 1),         # cond = n < 1
      instructions.BranchOp(
          "cond", finish_is_even, recur_is_even),  # if cond
  ])
  finish_is_even.assign_instructions([
      instructions.PopOp(["n", "cond"]),           #   done with n, cond
      instructions.prim_op(
          [], "ans", lambda : True),               #   ans = True
      instructions.halt_op(),                      #   return ans
  ])
  recur_is_even.assign_instructions([              # else
      instructions.PopOp(["cond"]),                #   done with cond now
      instructions.prim_op(
          ["n"], "nm1", lambda n: n - 1),          #   nm1 = n - 1
      instructions.PopOp(["n"]),                   #   done with n
      instructions.FunctionCallOp(
          is_odd_func, ["nm1"], "ans"),            #   ans = is_odd(nm1)
      instructions.PopOp(["nm1"]),                 #   done with nm1
      instructions.halt_op(),                      #   return ans
  ])
  is_even_blocks = [enter_is_even, finish_is_even, recur_is_even]
  is_even_func.graph = instructions.ControlFlowGraph(is_even_blocks)

  enter_is_odd = instructions.Block()
  finish_is_odd = instructions.Block()
  recur_is_odd = instructions.Block()
  # pylint: disable=bad-whitespace
  # Definition of is_odd function
  enter_is_odd.assign_instructions([
      instructions.prim_op(
          ["n"], "cond", lambda n: n < 1),         # cond = n < 1
      instructions.BranchOp(
          "cond", finish_is_odd, recur_is_odd),    # if cond
  ])
  finish_is_odd.assign_instructions([
      instructions.PopOp(["n", "cond"]),           #   done with n, cond
      instructions.prim_op(
          [], "ans", lambda : False),              #   ans = False
      instructions.halt_op(),                      #   return ans
  ])
  recur_is_odd.assign_instructions([               # else
      instructions.PopOp(["cond"]),                #   done with cond now
      instructions.prim_op(
          ["n"], "nm1", lambda n: n - 1),          #   nm1 = n - 1
      instructions.PopOp(["n"]),                   #   done with n
      instructions.FunctionCallOp(
          is_even_func, ["nm1"], "ans"),           #   ans = is_even(nm1)
      instructions.PopOp(["nm1"]),                 #   done with nm1
      instructions.halt_op(),                      #   return ans
  ])
  is_odd_blocks = [enter_is_odd, finish_is_odd, recur_is_odd]
  is_odd_func.graph = instructions.ControlFlowGraph(is_odd_blocks)

  is_even_main_blocks = [
      instructions.Block(
          [
              instructions.FunctionCallOp(is_even_func, ["n1"], "ans"),
          ],
          instructions.halt_op()),
  ]
  # pylint: disable=bad-whitespace
  is_even_vars = {
      "n"                   : instructions.single_type(dtype, ()),
      "n1"                  : instructions.single_type(dtype, ()),
      "cond"                : instructions.single_type(np.bool_, ()),
      "nm1"                 : instructions.single_type(dtype, ()),
      "ans"                 : instructions.single_type(np.bool_, ()),
  }
  if not include_types:
    _strip_types(is_even_vars)

  return instructions.Program(
      instructions.ControlFlowGraph(is_even_main_blocks),
      [is_even_func, is_odd_func],
      is_even_vars, ["n1"], "ans")


def fibonacci_function_calls(include_types=True, dtype=np.int64):
  """The Fibonacci program again, but with `instructions.FunctionCallOp`.

  Computes fib(n): fib(0) = fib(1) = 1.

  Args:
    include_types: If False, we omit types on the variables, requiring a type
        inference pass.
    dtype: The dtype to use for `n`-like internal state variables.

  Returns:
    program: Full-powered `instructions.Program` that computes fib(n).
  """
  enter_fib = instructions.Block(name="enter_fib")
  recur = instructions.Block(name="recur")
  finish = instructions.Block(name="finish")

  fibonacci_type = lambda types: types[0]
  fibonacci_func = instructions.Function(
      None, ["n"], "ans", fibonacci_type, name="fibonacci")
  # pylint: disable=bad-whitespace
  # Definition of fibonacci function
  enter_fib.assign_instructions([
      instructions.prim_op(
          ["n"], "cond",
          lambda n: n > 1),                      # cond = n > 1
      instructions.BranchOp(
          "cond", recur, finish),                # if cond
  ])
  recur.assign_instructions([
      instructions.prim_op(
          ["n"], "nm1",
          lambda n: n - 1),                      #   nm1 = n - 1
      instructions.FunctionCallOp(
          fibonacci_func, ["nm1"], "fibm1"),     #   fibm1 = fibonacci(nm1)
      instructions.prim_op(
          ["n"], "nm2",
          lambda n: n - 2),                      #   nm2 = n - 2
      instructions.FunctionCallOp(
          fibonacci_func, ["nm2"], "fibm2"),     #   fibm2 = fibonacci(nm2)
      instructions.prim_op(
          ["fibm1", "fibm2"], "ans",
          lambda x, y: x + y),                   #   ans = fibm1 + fibm2
      instructions.halt_op(),                    #   return ans
  ])
  finish.assign_instructions([                   # else:
      instructions.prim_op(
          [], "ans",
          lambda : 1),                           #   ans = 1
      instructions.halt_op(),                    #   return ans
  ])
  fibonacci_blocks = [enter_fib, recur, finish]
  fibonacci_func.graph = instructions.ControlFlowGraph(fibonacci_blocks)

  fibonacci_main_blocks = [
      instructions.Block(
          [
              instructions.FunctionCallOp(fibonacci_func, ["n1"], "ans"),
          ],
          instructions.halt_op(),
          name="main_entry"),
  ]

  # pylint: disable=bad-whitespace
  fibonacci_vars = {
      "n"                   : instructions.single_type(dtype, ()),
      "n1"                  : instructions.single_type(dtype, ()),
      "cond"                : instructions.single_type(np.bool_, ()),
      "nm1"                 : instructions.single_type(dtype, ()),
      "fibm1"               : instructions.single_type(dtype, ()),
      "nm2"                 : instructions.single_type(dtype, ()),
      "fibm2"               : instructions.single_type(dtype, ()),
      "ans"                 : instructions.single_type(dtype, ()),
  }
  if not include_types:
    _strip_types(fibonacci_vars)

  return instructions.Program(
      instructions.ControlFlowGraph(fibonacci_main_blocks),
      [fibonacci_func], fibonacci_vars, ["n1"], "ans")


def pea_nuts_program(latent_shape, choose_depth, step_state):
  """Synthetic program usable for benchmarking VM performance.

  This program is intended to resemble the control flow and scaling
  parameters of the NUTS algorithm, without any of the complexity.
  Hence the name.

  Each batch member looks like:

    state = ... # shape latent_shape

    def recur(depth, state):
      if depth > 1:
        state1 = recur(depth - 1, state)
        state2 = state1 + 1
        state3 = recur(depth - 1, state2)
        ans = state3 + 1
      else:
        ans = step_state(state)  # To simulate NUTS, something heavy
      return ans

    while count > 0:
      count = count - 1
      depth = choose_depth(count)
      state = recur(depth, state)

  Args:
    latent_shape: Python `tuple` of `int` giving the event shape of the
      latent state.
    choose_depth: Python `Tensor -> Tensor` callable.  The input
      `Tensor` will have shape `[batch_size]` (i.e., scalar event
      shape), and give the iteration of the outer while loop the
      thread is in.  The `choose_depth` function must return a `Tensor`
      of shape `[batch_size]` giving the depth, for each thread,
      to which to call `recur` in this iteration.
    step_state: Python `Tensor -> Tensor` callable.  The input and
      output `Tensor`s will have shape `[batch_size] + latent_shape`.
      This function is expected to update the state, and represents
      the "real work" versus which the VM overhead is being measured.

  Returns:
    program: `instructions.Program` that runs the above benchmark.
  """
  entry = instructions.Block()
  top_body = instructions.Block()
  finish_body = instructions.Block()
  enter_recur = instructions.Block()
  recur_body_1 = instructions.Block()
  recur_body_2 = instructions.Block()
  recur_body_3 = instructions.Block()
  recur_base_case = instructions.Block()
  # pylint: disable=bad-whitespace
  entry.assign_instructions([
      instructions.prim_op(
          ["count"], "cond",
          lambda count: count > 0),          # cond = count > 0
      instructions.BranchOp(
          "cond", top_body,
          instructions.halt()),              # if cond
  ])
  top_body.assign_instructions([
      instructions.PopOp(["cond"]),          #   done with cond now
      instructions.prim_op(
          ["count"], "ctm1",
          lambda count: count - 1),          #   ctm1 = count - 1
      instructions.PopOp(["count"]),         #   done with count now
      instructions.push_op(
          ["ctm1"], ["count"]),              #   count = ctm1
      instructions.PopOp(["ctm1"]),          #   done with ctm1
      instructions.prim_op(
          ["count"], "depth",
          choose_depth),                     #   depth = choose_depth(count)
      instructions.push_op(
          ["depth", "state"],
          ["depth", "state"]),               #   state = recur(depth, state)
      instructions.PopOp(
          ["depth", "state"]),               #     done with depth, state
      instructions.PushGotoOp(
          finish_body, enter_recur),
  ])
  finish_body.assign_instructions([
      instructions.push_op(
          ["ans"], ["state"]),               #     ...
      instructions.PopOp(["ans"]),           #     pop callee's "ans"
      instructions.GotoOp(entry),            # end of while body
  ])
  # Definition of recur begins here
  enter_recur.assign_instructions([
      instructions.prim_op(
          ["depth"], "cond1",
          lambda depth: depth > 0),          # cond1 = depth > 0
      instructions.BranchOp(
          "cond1", recur_body_1,
          recur_base_case),                  # if cond1
  ])
  recur_body_1.assign_instructions([
      instructions.PopOp(["cond1"]),         #   done with cond1 now
      instructions.prim_op(
          ["depth"], "dm1",
          lambda depth: depth - 1),          #   dm1 = depth - 1
      instructions.PopOp(["depth"]),         #   done with depth
      instructions.push_op(
          ["dm1", "state"],
          ["depth", "state"]),               #   state1 = recur(dm1, state)
      instructions.PopOp(["state"]),         #     done with state
      instructions.PushGotoOp(
          recur_body_2, enter_recur),
  ])
  recur_body_2.assign_instructions([
      instructions.push_op(
          ["ans"], ["state1"]),              #     ...
      instructions.PopOp(["ans"]),           #     pop callee's "ans"
      instructions.prim_op(
          ["state1"], "state2",
          lambda state: state + 1),          #   state2 = state1 + 1
      instructions.PopOp(["state1"]),        #   done with state1
      instructions.push_op(
          ["dm1", "state2"],
          ["depth", "state"]),               #   state3 = recur(dm1, state2)
      instructions.PopOp(
          ["dm1", "state2"]),                #     done with dm1, state2
      instructions.PushGotoOp(
          recur_body_3, enter_recur),
  ])
  recur_body_3.assign_instructions([
      instructions.push_op(
          ["ans"], ["state3"]),              #     ...
      instructions.PopOp(["ans"]),           #     pop callee's "ans"
      instructions.prim_op(
          ["state3"], "ans",
          lambda state: state + 1),          #   ans = state3 + 1
      instructions.PopOp(["state3"]),        #   done with state3
      instructions.IndirectGotoOp(),         #   return ans
  ])
  recur_base_case.assign_instructions([
      instructions.PopOp(
          ["cond1", "depth"]),               #   done with cond1, depth
      instructions.prim_op(
          ["state"], "ans", step_state),     #   ans = step_state(state)
      instructions.PopOp(["state"]),         #   done with state
      instructions.IndirectGotoOp(),         #   return ans
  ])

  pea_nuts_graph = instructions.ControlFlowGraph([
      entry,
      top_body,
      finish_body,
      enter_recur,
      recur_body_1,
      recur_body_2,
      recur_body_3,
      recur_base_case,
  ])

  # pylint: disable=bad-whitespace
  pea_nuts_vars = {
      "count"            : instructions.single_type(np.int64, ()),
      "cond"             : instructions.single_type(np.bool_,  ()),
      "cond1"            : instructions.single_type(np.bool_,  ()),
      "ctm1"             : instructions.single_type(np.int64, ()),
      "depth"            : instructions.single_type(np.int64, ()),
      "dm1"              : instructions.single_type(np.int64, ()),
      "state"            : instructions.single_type(np.float32, latent_shape),
      "state1"           : instructions.single_type(np.float32, latent_shape),
      "state2"           : instructions.single_type(np.float32, latent_shape),
      "state3"           : instructions.single_type(np.float32, latent_shape),
      "ans"              : instructions.single_type(np.float32, latent_shape),
  }

  return instructions.Program(
      pea_nuts_graph, [], pea_nuts_vars, ["count", "state"], "state")
