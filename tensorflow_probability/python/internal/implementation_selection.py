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
"""Utilities for selecting a device-specific implementation."""

import functools
import uuid

import tensorflow.compat.v2 as tf


__all__ = [
    'implementation_selecting',
    'is_xla',
    'never_runs_functions_eagerly',
]


# The following string constants support a secret handshake with the TF function
# library and Grappler's implementation_selector.cc.
# This tf.function implements the API named ...
_FUNCTION_API_NAME_ATTRIBUTE = 'api_implements'
# This tf.function is the implementation of the API specialized for ...
_FUNCTION_DEVICE_ATTRIBUTE = 'api_preferred_device'
_CPU_DEVICE_NAME = 'CPU'

# The following number constants are used to represent the runtime of the defun
# backend function. Since the CPU and default implementations are otherwise
# indistinguishable, we need some signal for the function to indicate which
# function was executed. This is for testing, to verify the backend function
# actually swapped.
_RUNTIME_UNKNOWN = 0
_RUNTIME_DEFAULT = 1
_RUNTIME_CPU = 2

JAX_MODE = False
NUMPY_MODE = False


def is_xla():
  """Returns `True` when we are tracing a function for XLA compilation."""
  if JAX_MODE:
    return True
  # Import locally to avoid TF dependency for TFP-on-JAX.
  from tensorflow.python.framework import ops  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  from tensorflow.python.ops import control_flow_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  return (not tf.executing_eagerly() and
          control_flow_util.GraphOrParentsInXlaContext(ops.get_default_graph()))


def _function_with_backend(func, unique_api_name, preferred_device=None):
  function_attributes = {_FUNCTION_API_NAME_ATTRIBUTE: unique_api_name}
  if preferred_device:
    function_attributes[_FUNCTION_DEVICE_ATTRIBUTE] = preferred_device
  return tf.function(
      func, autograph=False, experimental_attributes=function_attributes
  )


def never_runs_functions_eagerly(f):
  @functools.wraps(f)
  def f_wrapped(*args, **kwargs):
    orig = tf.config.functions_run_eagerly()
    try:
      tf.config.run_functions_eagerly(False)
      return f(*args, **kwargs)
    finally:
      tf.config.run_functions_eagerly(orig)
  return f_wrapped


def implementation_selecting(fn_name, default_fn, cpu_fn):
  """Secret handshake: Grappler replaces default_fn w/ cpu_fn when run on CPU.

  Note that this should generally be called in the scope of a `tf.function`.

  Example:
  ```python
  @tf.function(autograph=False)
  def cumsum(x):

    def _xla_friendly(x):
      return tf.math.cumsum(x)

    def _xla_hostile(x):
      return tf.while_loop(
          cond=lambda x, _: tf.size(x) > 0,
          body=lambda x, cx: (
              x[:-1], cx + tf.pad(x, [[tf.size(cx) - tf.size(x), 0]])),
          loop_vars=[x, tf.zeros_like(x)],
          shape_invariants=[tf.TensorShape([None]), x.shape])[1]

    return implementation_selecting(
        'cumsum', default_fn=_xla_friendly, cpu_fn=_xla_hostile)(x=x)

  retval, which_runtime = cumsum(tf.constant([1, 2, 3]))
  ```

  Args:
    fn_name: A `str` prefix used to specify the "API" being implemented by these
      functions. Mostly only interesting for debugging / graph inspection.
    default_fn: The default implementation. This is used when we are in an XLA
      tracing context, in TFP-on-JAX, or running on devices such as GPU or TPU.
    cpu_fn: The CPU-specialized implementation. This impl is used when Grappler
      sees that the function execution has been placed on the CPU. (Also used
      for TFP-on-numpy.)

  Returns:
    impl_selecting_fn: A function which takes kwargs only, forwards them to the
      underlying functions, and returns a tuple:
      `(selected_fn(**kwargs), runtime_that_was_selected)`.
  """
  def wrap_fn(fn, runtime):
    def stub_fn(**kwargs):
      return fn(**kwargs), tf.constant(runtime, dtype=tf.float32)
    stub_fn.__name__ = fn.__name__
    return stub_fn

  default_fn = wrap_fn(default_fn, _RUNTIME_DEFAULT)
  cpu_fn = wrap_fn(cpu_fn, _RUNTIME_CPU)

  def impl_selecting_fn(**kwargs):
    """The wrapper function to be returned."""
    if is_xla():  # JAX, XLA breakout.
      return default_fn(**kwargs)
    if NUMPY_MODE:  # Numpy breakout.
      return cpu_fn(**kwargs)

    # Each time a `tf.function` is called, we will give it a unique
    # identifiable API name, so that Grappler won't get confused when it
    # sees multiple samplers in same graph, and it will be able
    # to pair up the different implementations across them.
    api_name = '{}_{}'.format(fn_name, str(uuid.uuid4()))
    default_fn_with_backend = _function_with_backend(
        default_fn, api_name)
    cpu_fn_with_backend = _function_with_backend(
        cpu_fn, api_name, preferred_device=_CPU_DEVICE_NAME)

    # Call the default sampling impl and register the CPU-specialized impl.
    # Grappler will kick in during session execution to optimize the graph.
    result, runtime = default_fn_with_backend(**kwargs)
    concrete_func = cpu_fn_with_backend.get_concrete_function(**kwargs)
    concrete_func.add_to_graph()
    concrete_func.add_gradient_functions_to_graph()
    return result, runtime
  return impl_selecting_fn
