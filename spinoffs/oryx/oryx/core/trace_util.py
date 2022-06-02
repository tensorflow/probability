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
"""Module for JAX tracing utility functions."""
import contextlib
import threading
from typing import Any, Dict, Generator, List

import jax
from jax import abstract_arrays
from jax import api_util
from jax import core as jax_core
from jax import linear_util as lu
from jax import tree_util
from jax import util as jax_util
from jax._src import dtypes
from jax.interpreters import partial_eval as pe

__all__ = [
    'get_shaped_aval',
    'pv_like',
    'stage',
    'trees',
    'new_dynamic_context',
    'get_dynamic_context'
]

safe_map = jax_util.safe_map


def get_shaped_aval(x):
  """Converts a JAX value type into a shaped abstract value."""
  if hasattr(x, 'dtype') and hasattr(x, 'shape'):
    return abstract_arrays.ShapedArray(
        x.shape, dtypes.canonicalize_dtype(x.dtype))
  return abstract_arrays.raise_to_shaped(jax_core.get_aval(x))


def pv_like(x, abstract=True):
  """Converts a JAX value type into a JAX `PartialVal`."""
  if abstract:
    return pe.PartialVal.unknown(get_shaped_aval(x))
  else:
    return pe.PartialVal((None, x))  # pytype: disable=wrong-arg-types


def stage(f, dynamic=True):
  """Returns a function that stages a function to a ClosedJaxper."""

  def wrapped(*args, **kwargs):
    fun = lu.wrap_init(f, kwargs)
    flat_args, in_tree = tree_util.tree_flatten(args)
    flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
    flat_avals = safe_map(get_shaped_aval, flat_args)
    if not jax.config.omnistaging_enabled:
      raise ValueError('Oryx must be used with JAX omnistaging enabled.')
    if dynamic:
      jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
          flat_fun,
          flat_avals)
    else:
      pvals = [pe.PartialVal.unknown(aval) for aval in flat_avals]
      jaxpr, _, consts = pe.trace_to_jaxpr(
          flat_fun,
          pvals,
          instantiate=True)
    typed_jaxpr = jax_core.ClosedJaxpr(jaxpr, consts)
    return typed_jaxpr, (in_tree, out_tree())

  return wrapped


def trees(f):
  """Returns a function that determines input and output pytrees from inputs."""

  def wrapped(*args, **kwargs):
    return stage(f)(*args, **kwargs)[1]

  return wrapped


class _ThreadLocalState(threading.local):

  def __init__(self):
    super().__init__()
    self.dynamic_contexts: Dict[jax_core.MainTrace, List[Any]] = {}

_thread_local_state = _ThreadLocalState()


@contextlib.contextmanager
def new_dynamic_context(master: jax_core.MainTrace,
                        context: Any) -> Generator[None, None, None]:
  """Creates a dynamic context for a trace."""
  if master not in _thread_local_state.dynamic_contexts:
    _thread_local_state.dynamic_contexts[master] = []
  _thread_local_state.dynamic_contexts[master].append(context)
  try:
    yield
  finally:
    _thread_local_state.dynamic_contexts[master].pop()
    if not _thread_local_state.dynamic_contexts[master]:
      del _thread_local_state.dynamic_contexts[master]


def get_dynamic_context(trace: jax_core.Trace) -> Any:
  """Returns the current active dynamic context for a trace."""
  if trace.main not in _thread_local_state.dynamic_contexts:
    raise ValueError(f'No dynamic context registered for trace: {trace}')
  return _thread_local_state.dynamic_contexts[trace.main][-1]


def extract_call_jaxpr(primitive, params):
  if not (primitive.call_primitive or primitive.map_primitive):
    return None, params
  else:
    params = dict(params)
    return params.pop('call_jaxpr'), params
