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
"""Functions for minimizing losses."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

_trace_loss = lambda loss, grads, variables: loss


def minimize(loss_fn,
             num_steps,
             optimizer,
             trainable_variables=None,
             trace_fn=_trace_loss,
             name='minimize'):
  """Minimize a loss function using a provided optimizer.

  Args:
    loss_fn: Python callable with signature `loss = loss_fn()`, where `loss`
      is a `Tensor` loss to be minimized.
    num_steps: Python `int` number of steps to run the optimizer.
    optimizer: Optimizer instance to use. This may be a TF1-style
      `tf.train.Optimizer`, TF2-style `tf.optimizers.Optimizer`, or any Python
      object that implements `optimizer.apply_gradients(grads_and_vars)`.
    trainable_variables: list of `tf.Variable` instances to optimize with
      respect to. If `None`, defaults to the set of all variables accessed
      during the execution of `loss_fn()`.
      Default value: `None`.
    trace_fn: Python callable with signature `state = trace_fn(
      loss, grads, variables)`, where `state` may be a `Tensor` or nested
      structure of `Tensor`s. The state values are accumulated (by `tf.scan`)
      and returned. The default `trace_fn` simply returns the loss, but in
      general can depend on the gradients and variables (if
      `trainable_variables` is not `None` then `variables==trainable_variables`;
      otherwise it is the list of all variables accessed during execution of
      `loss_fn()`), as well as any other quantities captured in the closure of
      `trace_fn`, for example, statistics of a variational distribution.
      Default value: `lambda loss, grads, variables: loss`.
    name: Python `str` name prefixed to ops created by this function.
      Default value: 'minimize'.

  Returns:
    trace: `Tensor` or nested structure of `Tensor`s, according to the
      return type of `trace_fn`. Each `Tensor` has an added leading dimension
      of size `num_steps`, packing the trajectory of the result over the course
      of the optimization.

  ### Examples

  To minimize the scalar function `(x - 5)**2`:

  ```python
  x = tf.Variable(0.)
  loss_fn = lambda: (x - 5.)**2
  losses = tfp.math.minimize(loss_fn,
                             num_steps=100,
                             optimizer=tf.optimizers.Adam(learning_rate=0.1))

  # In TF2/eager mode, the optimization runs immediately.
  print("optimized value is {} with loss {}".format(x, losses[-1]))
  ```

  In graph mode (e.g., inside of `tf.function` wrapping), retrieving any Tensor
  that depends on the minimization op will trigger the optimization:

  ```python
  with tf.control_dependencies([losses]):
    optimized_x = tf.identity(x)  # Use a dummy op to attach the dependency.
  ```

  In some cases, we may want to track additional context inside the
  optimization. We can do this by defining a custom `trace_fn`. Note that
  the `trace_fn` is passed the loss and gradients, but it may also report the
  values of trainable variables or other derived quantities by capturing them in
  its closure. For example, we can capture `x` and track its value over the
  optimization:

  ```python
  # `x` is the tf.Variable instance defined above.
  trace_fn = lambda loss, grads, variables: {'loss': loss, 'x': x}
  trace = tfp.vi.minimize(loss_fn, num_steps=100,
                          optimizer=tf.optimizers.Adam(0.1),
                          trace_fn=trace_fn)
  print(trace['loss'].shape,   # => [100]
        trace['x'].shape)      # => [100]
  ```
  """

  @tf.function(autograph=False)
  def train_loop_body(old_result, step):  # pylint: disable=unused-argument
    """Run a single optimization step."""
    with tf.GradientTape(
        watch_accessed_variables=trainable_variables is None) as tape:
      for v in trainable_variables or []:
        tape.watch(v)
      loss = loss_fn()
    watched_variables = tape.watched_variables()
    grads = tape.gradient(loss, watched_variables)
    train_op = optimizer.apply_gradients(zip(grads, watched_variables))
    with tf.control_dependencies([train_op]):
      state = trace_fn(tf.identity(loss),
                       [tf.identity(g) for g in grads],
                       [tf.identity(v) for v in watched_variables])
    return state

  with tf.name_scope(name) as name:

    # Compute the shape of the trace without executing the graph, if possible.
    concrete_loop_body = train_loop_body.get_concrete_function(
        tf.TensorSpec([]), tf.TensorSpec([]))  # Inputs ignored.
    if all([shape.is_fully_defined()
            for shape in tf.nest.flatten(concrete_loop_body.output_shapes)]):
      state_initializer = tf.nest.map_structure(
          lambda shape, dtype: tf.zeros(shape, dtype=dtype),
          concrete_loop_body.output_shapes,
          concrete_loop_body.output_dtypes)
      initial_trace_step = None
    else:
      state_initializer = concrete_loop_body(
          tf.convert_to_tensor(0.), tf.convert_to_tensor(0.))  # Inputs ignored.
      num_steps = num_steps - 1
      initial_trace_step = state_initializer

    # TODO(b/136103064): Rewrite as explicit `while_loop` to support custom
    # convergence criteria and Tensor-valued `num_steps`, and avoid
    # re-tracing the train loop body.
    trace = tf.scan(train_loop_body,
                    elems=np.arange(num_steps),
                    initializer=state_initializer)
    if initial_trace_step is not None:
      trace = tf.nest.map_structure(
          lambda a, b: tf.concat([a[tf.newaxis, ...], b], axis=0),
          initial_trace_step, trace)
    return trace

