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
"""CovarianceReducer for pre-packaged streaming covariance."""

import collections

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.mcmc import reducer as reducer_base
from tensorflow_probability.python.experimental.stats import sample_stats
from tensorflow_probability.python.internal import nest_util
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'CovarianceReducer',
    'VarianceReducer'
]


CovarianceReducerState = collections.namedtuple(
    'CovarianceReducerState', 'cov_state')


def _get_sample(current_state, kernel_results):
  del kernel_results
  return current_state


class CovarianceReducer(reducer_base.Reducer):
  """`Reducer` that computes a running covariance.

  By default, `CovarianceReducer` computes covariance directly over the
  latent state, ignoring kernel results; however, it can also compute
  a running covariance on samples evaluated on some arbitrary structure of
  `transform_fn`s. A `trasform_fn` is defined as any function that accepts
  the chain state and kernel results, and ouptuts a `Tensor` or nested
  structure of `Tensor`s to compute the covariance over. To be explicit,
  if we denote a `transform_fn` as f(x, y), then `CovarianceReducer` will
  compute Cov(f(x, y)). If some structure of `transform_fn`s are given,
  the final statistic (as returned by the `finalize` method) will mimic
  the results of that structure.

  As with all reducers, CovarianceReducer does not hold state information;
  rather, it stores supplied metadata. Intermediate calculations are held in
  a state object, which is returned via `initialize` and `one_step` method
  calls.

  `CovarianceReducer` is meant to fit into the larger Streaming MCMC
  framework. `RunningCovariance` in `tfp.experimental.stats` is better suited
  for more generic streaming covariance needs.

  There are two ways to stream over MCMC samples. The first is to manually
  wrap `CovarianceReducer` in a `tfp.experimental.mcmc.WithReductions`
  `TransitionKernel`. Doing so enables a wide variety of possible compositions.
  For example, to perform covariance calculation on samples that survive
  thinning and burn-in, `WithReductions` should wrap around an appropriate
  `SampleDiscardingKernel`.

  If the sole objective is to estimate covariance, it may be more convenient to
  use a pre-defined driver like `tfp.experimental.mcmc.sample_fold`.
  `sample_fold` will automatically create the Transition Kernel onion, apply
  `kernel` samples to update reducer states, and `finalize` calculations.

  ```
  python

  kernel = ...
  reducer = tfp.experimental.mcmc.CovarianceReducer()
  covariance_estimate, _, _ = tfp.experimental.mcmc.sample_fold(
      num_steps=...,
      current_state=...,
      previous_kernel_results=...,
      kernel=kernel,
      reducers=reducer,
  )
  covariance_estimate  # estimate of covariance (not `CovarianceReducerState`)
  ```
  """

  def __init__(
      self, event_ndims=None, transform_fn=_get_sample, ddof=0, name=None):
    """Instantiates this object.

    The running covariance computation supports batching. The `event_ndims`
    parameter indicates the number of trailing dimensions to treat as part of
    the event, and to compute covariance across. The leading dimensions, if
    any, are treated as batch shape, and no cross terms are computed.

    For example, if the incoming samples have shape `[5, 7]`, the `event_ndims`
    selects among three different covariance computations:
    - `event_ndims=0` treats the samples as a `[5, 7]` batch of scalar random
      variables, and computes their variances in batch.  The shape of the result
      is `[5, 7]`.
    - `event_ndims=1` treats the samples as a `[5]` batch of vector random
      variables of shape `[7]`, and computes their covariances in batch.  The
      shape of the result is `[5, 7, 7]`.
    - `event_ndims=2` treats the samples as a single random variable of
      shape `[5, 7]` and computes its covariance.  The shape of the result
      is `[5, 7, 5, 7]`.

    For nested latent samples, both `event_ndims` and `ddof` must be either
    a single value or of identical structure to the chain state. For example,
    if the chain state is the tuple of dictionaries ({pairA, pairB}, {pairC}),
    one could use scalar values for `event_ndims` (integer or `None`) and
    `ddof` (integer) that apply to all states in the latent. If not,
    `event_ndims` and `ddof` must be structures that exactly mimic
    ({keyA: argA, keyB: argB}, {keyC: argC}) in composition.

    Args:
      event_ndims: A (possibly nested) structure of integers, or `None`. Defines
        the number of inner-most dimensions that represent the event shape.
        Specifying `None` returns all cross product terms (no batching)
        and is the default.
      transform_fn: A (possibly nested) structure of functions to evaluate the
        incoming chain states on before covariance calculation. The default is
        a single function that returns the chain state.
      ddof: A (possibly nested) structure of integers that represent the
        requested dynamic degrees of freedom. For example, use `ddof=0`
        for population covariance and `ddof=1` for sample covariance. Defaults
        to the population covariance.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'covariance_reducer').
    """
    self._parameters = dict(
        event_ndims=event_ndims,
        transform_fn=transform_fn,
        ddof=ddof,
        name=name or 'covariance_reducer'
    )

  def initialize(self, initial_chain_state, initial_kernel_results=None):
    """Initializes a `CovarianceReducerState` using previously defined metadata.

    For calculation purposes, the `initial_chain_state` does not count as a
    sample. This is a deliberate decision that ensures consistency across
    sampling procedures (i.e. `tfp.mcmc.sample_chain` follows similar
    semantics).

    Args:
      initial_chain_state: A (possibly nested) structure of `Tensor`s or Python
        `list`s of `Tensor`s representing the current state(s) of the Markov
        chain(s). It is used to infer the shape and dtype of future samples.
      initial_kernel_results: A (possibly nested) structure of `Tensor`s
        representing internal calculations made in a related `TransitionKernel`.

    Returns:
      state: `CovarianceReducerState` with `cov_state` field representing
        a stream of no inputs.
    """
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'covariance_reducer', 'initialize')):
      initial_chain_state = tf.nest.map_structure(
          tf.convert_to_tensor,
          initial_chain_state)
      if initial_kernel_results is not None:
        initial_kernel_results = tf.nest.map_structure(
            tf.convert_to_tensor,
            initial_kernel_results,
            expand_composites=True)
      initial_fn_result = tf.nest.map_structure(
          lambda fn: fn(initial_chain_state, initial_kernel_results),
          self.transform_fn)
      event_ndims = _canonicalize_event_ndims(
          initial_fn_result, self.event_ndims)
      running_covariances = tf.nest.map_structure(
          sample_stats.RunningCovariance.from_example,
          initial_fn_result, event_ndims)
      return CovarianceReducerState(running_covariances)

  def one_step(
      self,
      new_chain_state,
      current_reducer_state,
      previous_kernel_results=None,
      axis=None):
    """Update the `current_reducer_state` with a new chain state.

    Chunking semantics are similar to those of batching and are specified by the
    `axis` parameter. If chunking is enabled (axis is not `None`), all elements
    along the specified `axis` will be treated as separate samples. If a
    single scalar value is provided for a non-scalar sample structure, that
    value will be used for all elements in the structure. If not, an identical
    structure must be provided.

    Args:
      new_chain_state: A (possibly nested) structure of incoming chain state(s)
        with shape and dtype compatible with those used to initialize the
        `current_reducer_state`.
      current_reducer_state: `CovarianceReducerState`s representing the current
        state of the running covariance.
      previous_kernel_results: A (possibly nested) structure of `Tensor`s
        representing internal calculations made in a related
        `TransitionKernel`.
      axis: If chunking is desired, this is a (possibly nested) structure of
        integers that specifies the axis with chunked samples. For individual
        samples, set this to `None`. By default, samples are not chunked
        (`axis` is None).

    Returns:
      new_reducer_state: `CovarianceReducerState` with updated running
        statistics. Its `cov_state` field has an identical structure to the
        results of `self.transform_fn`. Each of the individual values in that
        structure subsequently mimics the structure of `current_reducer_state`.
    """
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'covariance_reducer', 'one_step')):
      new_chain_state = tf.nest.map_structure(
          tf.convert_to_tensor,
          new_chain_state)
      if previous_kernel_results is not None:
        previous_kernel_results = tf.nest.map_structure(
            tf.convert_to_tensor,
            previous_kernel_results,
            expand_composites=True)
      fn_results = tf.nest.map_structure(
          lambda fn: fn(new_chain_state, previous_kernel_results),
          self.transform_fn)
      if not nest.is_nested(axis):
        axis = nest_util.broadcast_structure(fn_results, axis)
      running_covariances = tf.nest.map_structure(
          lambda cov, *args: cov.update(*args),
          current_reducer_state.cov_state,
          fn_results,
          axis,
          check_types=False)
      return CovarianceReducerState(running_covariances)

  def finalize(self, final_reducer_state):
    """Finalizes covariance calculation from the `final_reducer_state`.

    Args:
      final_reducer_state: `CovarianceReducerState`s that represent the
        final running covariance state.

    Returns:
      covariance: an estimate of the covariance with identical structure to
        the results of `self.transform_fn`.
    """
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'covariance_reducer', 'finalize')):
      def finalize(cov):
        return cov.covariance(ddof=self.ddof)
      return nest.map_structure(finalize, final_reducer_state.cov_state)

  @property
  def event_ndims(self):
    return self._parameters['event_ndims']

  @property
  def transform_fn(self):
    return self._parameters['transform_fn']

  @property
  def ddof(self):
    return self._parameters['ddof']

  @property
  def name(self):
    return self._parameters['name']

  @property
  def parameters(self):
    return self._parameters


class VarianceReducer(CovarianceReducer):
  """`Reducer` that computes running variance.

  This is a special case of `CovarianceReducer` with `event_ndims=0`, provided
  for convenience. See `CovarianceReducer` for more information.

  `VarianceReducer` is also meant to fit into the larger streaming MCMC
  framework. For more generic streaming variance needs, see
  `RunningVariance` in `tfp.experimental.stats`.
  """

  def __init__(self, transform_fn=_get_sample, ddof=0, name=None):
    super(VarianceReducer, self).__init__(
        event_ndims=0,
        transform_fn=transform_fn,
        ddof=ddof,
        name=name or 'variance_reducer',
    )


def _canonicalize_event_ndims(target, event_ndims):
  """Returns `event_ndims` shaped parallel to `target`, repeating as needed."""
  # This is only here to support the possibility of different event_ndims across
  # different Tensors in the target structure.  Otherwise, event_ndims could
  # just be an integer (or None) and wouldn't need to be canonicalized to a
  # structure.
  if not nest.is_nested(event_ndims):
    return nest_util.broadcast_structure(target, event_ndims)
  else:
    return event_ndims
