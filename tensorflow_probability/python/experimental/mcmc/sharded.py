# Copyright 2021 The TensorFlow Probability Authors.
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
"""Module for sharding MCMC chains."""

from tensorflow_probability.python.internal import distribute_lib
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.mcmc import kernel


class Sharded(kernel.TransitionKernel):
  """Shards a transition kernel across a named axis.

  Ordinarily, one can produce independent Markov chains from a single kernel by
  proving a batch of states but when using named axes inside of a map (say
  in the case of using JAX's `pmap`, `vmap`, or `xmap`), the kernel is provided
  with state without batch dimensions. In order to sample independently across
  the named axis, the PRNG seed across the named axis must be different. This
  can be accomplished by folding the named axis index into the random seed.
  A `Sharded` kernel does exactly this, creating independent chains across a
  named axis.
  """

  def __init__(self,
               inner_kernel,
               chain_axis_names,
               validate_args=False,
               name=None):
    """Constructs a `Sharded` transition kernel.

    Args:
      inner_kernel: A `TransitionKernel` to be sharded.
      chain_axis_names: A `str` or list of `str`s that determine the named axes
        that independent Markov chains will be sharded across.
      validate_args: Python `bool`. When `True` kernel parameters are checked
        for validity. When `False` invalid inputs may silently render incorrect
        outputs.
      name: Python `str` name prefixed to Ops created by this class.
    """
    chain_axis_names = distribute_lib.canonicalize_named_axis(chain_axis_names)
    self._parameters = dict(
        inner_kernel=inner_kernel,
        chain_axis_names=chain_axis_names,
        validate_args=validate_args,
        name=name)

  def bootstrap_results(self, init_state):
    return self.inner_kernel.bootstrap_results(init_state)

  def one_step(self, current_state, previous_kernel_results, seed=None):
    seed = samplers.sanitize_seed(seed, salt='sharded_kernel')
    seed = distribute_lib.fold_in_axis_index(seed, self.chain_axis_names)
    return self.inner_kernel.one_step(
        current_state, previous_kernel_results, seed=seed)

  @property
  def inner_kernel(self):
    return self._parameters['inner_kernel']

  @property
  def chain_axis_names(self):
    return self._parameters['chain_axis_names']

  @property
  def is_calibrated(self):
    return self.inner_kernel.is_calibrated

  @property
  def experimental_shard_axis_names(self):
    return self.kernel.experimental_shard_axis_names

  def experimental_with_shard_axes(self, shard_axis_names):
    return self.copy(
        kernel=kernel.experimental_with_shard_axes(shard_axis_names))
