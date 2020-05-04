# Copyright 2020 The TensorFlow Probability Authors.
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
"""FunMCMC utilities implemented via TensorFlow Probability."""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function

import functools

from discussion.fun_mcmc import backend
from discussion.fun_mcmc import fun_mcmc_lib
from typing import Any, Optional, Tuple

tf = backend.tf
tfp = backend.tfp
util = backend.util


def bijector_to_transform_fn(
    bijector: 'fun_mcmc_lib.BijectorNest',
    state_structure: 'Any',
    batch_ndims: 'fun_mcmc_lib.IntTensor' = 0
) -> 'fun_mcmc_lib.TransitionOperator':
  """Creates a TransitionOperator that transforms the state using a bijector.

  The returned operator has the following signature:

  ```none
    (*args, **kwargs) ->
      transformed_state, [(), forward_ldj]
  ```

  It also has an `inverse` property that contains the inverse transformation.

  By default, the returned operator is assumed to operate on single events, i.e.
  the returned `forward_ldj` is a scalar. This can be configured via the
  `batch_ndims` argument, which indicates how many leading dimensions of state
  are batch dimensions. This adjusts the shape of `forward_ldj` accordingly.

  The `bijector` argument must be a tree-prefix to the `state_structure`, which
  allows for multi-part bijectors.

  Args:
    bijector: A nest of bijectors.
    state_structure: Structure of the state that the returned transformation
      operates on.
    batch_ndims: How many leading dimensions of state are treated as batch
      dimensions.

  Returns:
    transform_fn: The created transformation.
  """

  def transform_fn(bijector, state_structure, *args, **kwargs):
    """Transport map implemented via the bijector."""
    state = fun_mcmc_lib.recover_state_from_args(args, kwargs, state_structure)

    value = util.map_tree_up_to(bijector, lambda b, x: b(x), bijector, state)
    ldj_parts = util.map_tree_up_to(
        bijector,
        lambda b, x: b.forward_log_det_jacobian(  # pylint: disable=g-long-lambda
            x,
            event_ndims=util.map_tree(lambda x: tf.rank(x) - batch_ndims, x)),
        bijector,
        state)
    ldj = sum(util.flatten_tree(ldj_parts))

    return value, ((), ldj)

  inverse_bijector = util.map_tree(tfp.bijectors.Invert, bijector)

  forward_transform_fn = functools.partial(transform_fn, bijector,
                                           state_structure)
  inverse_transform_fn = functools.partial(
      transform_fn, inverse_bijector,
      util.map_tree_up_to(bijector, lambda b, s: b.forward_dtype(s), bijector,
                          state_structure))

  forward_transform_fn.inverse = inverse_transform_fn
  inverse_transform_fn.inverse = forward_transform_fn

  return forward_transform_fn


def transition_kernel_wrapper(
    current_state: 'fun_mcmc_lib.FloatNest', kernel_results: 'Optional[Any]',
    kernel: 'tfp.mcmc.TransitionKernel'
) -> 'Tuple[fun_mcmc_lib.FloatNest, Any]':
  """Wraps a `tfp.mcmc.TransitionKernel` as a `TransitionOperator`.

  Args:
    current_state: Current state passed to the transition kernel.
    kernel_results: Kernel results passed to the transition kernel. Can be
      `None`.
    kernel: The transition kernel.

  Returns:
    state: A tuple of:
      current_state: Current state returned by the transition kernel.
      kernel_results: Kernel results returned by the transition kernel.
    extra: An empty tuple.
  """
  flat_current_state = util.flatten_tree(current_state)
  flat_current_state, kernel_results = kernel.one_step(flat_current_state,
                                                       kernel_results)
  return (util.unflatten_tree(current_state,
                              flat_current_state), kernel_results), ()
