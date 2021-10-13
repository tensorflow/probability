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
"""XLA utilities."""

import tensorflow.compat.v2 as tf

__all__ = ['compile_nested_output']


def compile_nested_output(f, compile_fn=None):
  """Wraps f with a `tpu.rewrite` or `xla.compile`, propagates output structure.

  `xla.compile` insists `f` output a flat list of `Tensor`s or `Op`s, but
  tolerates nested input arguments. Here, we capture the output structure in
  order to propagate it.

  Args:
    f: Callable to compile, may accept/return nested inputs/outputs.
    compile_fn: The function to use to compile, i.e. `xla.compile` or
      `tpu.rewrite`. Accepts two args, `f` and `inputs`.

  Returns:
    g: Callable wrapping `f` which returns XLA-compiled, nested outputs.
  """
  def _wrapper(*inputs):  # pylint:disable=missing-docstring
    nest = tf.nest
    struct = [None]
    def _flattened(*inputs):
      result = f(*inputs)
      flat = nest.flatten(result)
      # Ick: Side-effect. Ideally we could push output nest support into
      # tpu.rewrite / xla.compile. b/121383831
      struct[0] = nest.pack_sequence_as(result, [1] * len(flat))
      return flat
    res = compile_fn(_flattened, inputs)
    if struct[0] is None:
      raise ValueError('Expected nest structure in struct[0]')
    return nest.pack_sequence_as(struct[0], res)
  return _wrapper
