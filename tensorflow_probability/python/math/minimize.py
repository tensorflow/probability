# Copyright 2019 The TensorFlow Probability Authors.
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

import collections

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import samplers


class MinimizeTraceableQuantities(collections.namedtuple(
    'MinimizeTraceableQuantities',
    ('step', 'loss', 'gradients', 'parameters', 'has_converged',
     'convergence_criterion_state'))):
  """Namedtuple of quantities that may be traced from `tfp.math.minimize`.

  These are (in order):

  - `step`: int `Tensor` index (starting from zero) of the current optimization
     step.
  - `loss`: float `Tensor` value returned from the user-provided `loss_fn`.
  - `gradients`: list of `Tensor` gradients of `loss` with respect to the
     parameters.
  - `parameters`: list of `Tensor` values of parameters being optimized. This
     corresponds to the `trainable_variables` passed in to `minimize`.
  - `has_converged`: boolean `Tensor` of the same shape as `loss_fn`, with
    `True` values corresponding to loss entries that have converged according
    to the user-provided convergence criterion. If no convergence criterion
    was specified, this is `None`.
  - `convergence_criterion_state`: structure of `Tensor`s containing any
    auxiliary state (e.g., moving averages of loss or other quantities)
    maintained by the user-provided convergence criterion.

  """


def _sanitize_traced_values(traced_values):
  """Represents Python values and `None` as Tensors."""
  return tf.nest.map_structure(
      lambda x: (tf.zeros([0], dtype=tf.int32) if x is None  # pylint: disable=g-long-lambda
                 else tf.convert_to_tensor(x)),
      traced_values)


def _tile_last_written_value(trace_array, last_written_idx):
  last_written_value = trace_array.read(last_written_idx)
  _, tiled_trace_array = tf.while_loop(
      cond=lambda n, ta: n < ta.size(),
      body=lambda n, ta: (n + 1, ta.write(n, last_written_value)),
      loop_vars=(last_written_idx + 1, trace_array))
  return tiled_trace_array

_trace_loss = lambda traceable_quantities: traceable_quantities.loss


def _make_optimizer_step_fn(loss_fn, optimizer, trainable_variables):
  """Construct a single optimization step."""

  @tf.function(autograph=False)
  def optimizer_step(seed=None):
    """Run a single optimization step."""
    with tf.GradientTape(
        watch_accessed_variables=trainable_variables is None) as tape:
      for v in trainable_variables or []:
        tape.watch(v)
      try:
        loss = loss_fn(seed=seed)
      except TypeError:
        loss = loss_fn()
    watched_variables = tape.watched_variables()
    grads = tape.gradient(loss, watched_variables)
    train_op = optimizer.apply_gradients(zip(grads, watched_variables))
    with tf.control_dependencies([train_op]):
      return (tf.identity(loss),
              [tf.identity(g) for g in grads],
              [tf.identity(v) for v in watched_variables])
  return optimizer_step


def _make_training_loop_body(optimizer_step_fn,
                             convergence_criterion,
                             trace_fn):
  """Construct the training loop body."""

  def training_loop_body(step, seed, trace_arrays, has_converged=None,
                         convergence_criterion_state=None):
    """Invokes the convergence criterion and trace fn and writes the result."""
    seed, next_seed = samplers.split_seed(seed, n=2)
    loss, grads, parameters = optimizer_step_fn(seed=seed)
    if convergence_criterion is not None:
      (has_converged,
       convergence_criterion_state) = convergence_criterion.one_step(
           step, loss, grads, parameters, convergence_criterion_state)
    traceable_quantities = MinimizeTraceableQuantities(
        loss=loss, gradients=grads, parameters=parameters, step=step,
        has_converged=has_converged,
        convergence_criterion_state=convergence_criterion_state)
    traced_values = _sanitize_traced_values(trace_fn(traceable_quantities))
    trace_arrays = tf.nest.map_structure(
        lambda ta, x: ta.write(step, x), trace_arrays, traced_values)
    potential_new_loop_vars = (
        step + 1, next_seed, trace_arrays, has_converged,
        convergence_criterion_state)
    return [x for x in potential_new_loop_vars if x is not None]

  return training_loop_body


def _initialize_arrays(initial_values,
                       num_steps,
                       truncate_at_convergence):
  """Construct a structure of `TraceArray`s from initial values."""
  initial_values = _sanitize_traced_values(initial_values)
  num_steps_ = tf.get_static_value(tf.convert_to_tensor(num_steps))
  size_is_dynamic = (num_steps_ is None or truncate_at_convergence)
  trace_arrays = tf.nest.map_structure(
      lambda t: tf.TensorArray(  # pylint: disable=g-long-lambda
          dtype=t.dtype,
          size=1 if size_is_dynamic else num_steps_,  # Initial size.
          dynamic_size=size_is_dynamic,
          clear_after_read=False,  # Allow reading->tiling final value.
          element_shape=t.shape),
      initial_values)
  return tf.nest.map_structure(
      lambda ta, t: ta.write(0, t), trace_arrays, initial_values)


def minimize(loss_fn,
             num_steps,
             optimizer,
             convergence_criterion=None,
             batch_convergence_reduce_fn=tf.reduce_all,
             trainable_variables=None,
             trace_fn=_trace_loss,
             return_full_length_trace=True,
             jit_compile=False,
             seed=None,
             name='minimize'):
  """Minimize a loss function using a provided optimizer.

  Args:
    loss_fn: Python callable with signature `loss = loss_fn()`, where `loss`
      is a `Tensor` loss to be minimized. This may optionally take a `seed`
      keyword argument, used to specify a per-iteration seed for stochastic
      loss functions (a stateless `Tensor` seed will be passed; see
      `tfp.random.sanitize_seed`).
    num_steps: Python `int` maximum number of steps to run the optimizer.
    optimizer: Optimizer instance to use. This may be a TF1-style
      `tf.train.Optimizer`, TF2-style `tf.optimizers.Optimizer`, or any Python
      object that implements `optimizer.apply_gradients(grads_and_vars)`.
    convergence_criterion: Optional instance of
      `tfp.optimizer.convergence_criteria.ConvergenceCriterion`
      representing a criterion for detecting convergence. If `None`,
      the optimization will run for `num_steps` steps, otherwise, it will run
      for at *most* `num_steps` steps, as determined by the provided criterion.
      Default value: `None`.
    batch_convergence_reduce_fn: Python `callable` of signature
      `has_converged = batch_convergence_reduce_fn(batch_has_converged)`
      whose input is a `Tensor` of boolean values of the same shape as the
      `loss` returned by `loss_fn`, and output is a scalar
      boolean `Tensor`. This determines the behavior of batched
      optimization loops when `loss_fn`'s return value is non-scalar.
      For example, `tf.reduce_all` will stop the optimization
      once all members of the batch have converged, `tf.reduce_any` once *any*
      member has converged,
      `lambda x: tf.reduce_mean(tf.cast(x, tf.float32)) > 0.5` once more than
      half have converged, etc.
      Default value: `tf.reduce_all`.
    trainable_variables: list of `tf.Variable` instances to optimize with
      respect to. If `None`, defaults to the set of all variables accessed
      during the execution of `loss_fn()`.
      Default value: `None`.
    trace_fn: Python callable with signature `traced_values = trace_fn(
      traceable_quantities)`, where the argument is an instance of
      `tfp.math.MinimizeTraceableQuantities` and the returned `traced_values`
      may be a `Tensor` or nested structure of `Tensor`s. The traced values are
      stacked across steps and returned.
      The default `trace_fn` simply returns the loss. In general, trace
      functions may also examine the gradients, values of parameters,
      the state propagated by the specified `convergence_criterion`, if any (if
      no convergence criterion is specified, this will be `None`),
      as well as any other quantities captured in the closure of `trace_fn`,
      for example, statistics of a variational distribution.
      Default value: `lambda traceable_quantities: traceable_quantities.loss`.
    return_full_length_trace: Python `bool` indicating whether to return a trace
      of the full length `num_steps`, even if a convergence criterion stopped
      the optimization early, by tiling the value(s) traced at the final
      optimization step. This enables use in contexts such as XLA that require
      shapes to be known statically.
      Default value: `True`.
    jit_compile: If True, compiles the minimization loop using
      XLA. XLA performs compiler optimizations, such as fusion, and attempts to
      emit more efficient code. This may drastically improve the performance.
      See the docs for `tf.function`. (In JAX, this will apply `jax.jit`).
      Default value: `False`.
    seed: PRNG seed for stochastic losses; see `tfp.random.sanitize_seed.`
      Default value: `None`.
    name: Python `str` name prefixed to ops created by this function.
      Default value: 'minimize'.

  Returns:
    trace: `Tensor` or nested structure of `Tensor`s, according to the
      return type of `trace_fn`. Each `Tensor` has an added leading dimension
      stacking the trajectory of the traced values over the course of the
      optimization. The size of this dimension is equal to `num_steps` if
      a convergence criterion was not specified and/or
      `return_full_length_trace=True`, and otherwise it is equal
      equal to the number of optimization steps taken.

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

  We can attempt to automatically detect convergence and stop the optimization
  by passing an instance of
  `tfp.optimize.convergence_criteria.ConvergenceCriterion`. For example, to
  stop the optimization once a moving average of the per-step decrease in loss
  drops below `0.01`:

  ```python
  losses = tfp.math.minimize(
    loss_fn, num_steps=1000, optimizer=tf.optimizers.Adam(learning_rate=0.1),
    convergence_criterion=(
      tfp.optimizers.convergence_criteria.LossNotDecreasing(atol=0.01)))
  ```

  Here `num_steps=1000` defines an upper bound: the optimization will be
  stopped after 1000 steps even if no convergence is detected.

  In some cases, we may want to track additional context inside the
  optimization. We can do this by defining a custom `trace_fn`. Note that
  the `trace_fn` is passed the loss and gradients, as well as any auxiliary
  state maintained by the convergence criterion (if any), for example, moving
  averages of the loss or gradients, but it may also report the
  values of trainable parameters or other derived quantities by capturing them
  in its closure. For example, we can capture `x` and track its value over the
  optimization:

  ```python
  # `x` is the tf.Variable instance defined above.
  trace_fn = lambda traceable_quantities: {
    'loss': traceable_quantities.loss, 'x': x}
  trace = tfp.math.minimize(loss_fn, num_steps=100,
                            optimizer=tf.optimizers.Adam(0.1),
                            trace_fn=trace_fn)
  print(trace['loss'].shape,   # => [100]
        trace['x'].shape)      # => [100]
  ```

  When optimizing a batch of losses, some batch members will converge before
  others. The optimization will continue until the condition defined by the
  `batch_convergence_reduce_fn` becomes `True`. During these additional steps,
  converged elements will continue to be updated and may become unconverged.
  The convergence status of batch members can be diagnosed by tracing
  `has_converged`:

  ```python
  batch_size = 10
  x = tf.Variable([0.] * batch_size)
  trace_fn = lambda traceable_quantities: {
    'loss': traceable_quantities.loss,
    'has_converged': traceable_quantities.has_converged}
  trace = tfp.math.minimize(loss_fn, num_steps=100,
                            optimizer=tf.optimizers.Adam(0.1),,
                            trace_fn=trace_fn,
                            convergence_criterion=(
      tfp.optimizers.convergence_criteria.LossNotDecreasing(atol=0.01)))

  for i in range(batch_size):
    print('Batch element {} final state is {}converged.'
          ' It first converged at step {}.'.format(
          i, '' if has_converged[-1, i] else 'not ',
          np.argmax(trace.has_converged[:, i])))
  ```

  """

  if jit_compile:
    # Run the entire minimization inside a jit-compiled function. This is
    # typically faster than jit-compiling the individual steps.
    parameters = dict(locals())
    parameters['jit_compile'] = False
    @tf.function(autograph=False, jit_compile=True)
    def run_jitted_minimize():
      return minimize(**parameters)
    return run_jitted_minimize()

  def convergence_detected(step, seed, trace_arrays,
                           has_converged=None,
                           convergence_criterion_state=None):
    del step
    del seed
    del trace_arrays
    del convergence_criterion_state
    return (has_converged is not None  # Convergence criterion in use.
            and batch_convergence_reduce_fn(has_converged))

  # Main optimization routine.
  with tf.name_scope(name) as name:
    seed = samplers.sanitize_seed(seed, salt='minimize')

    # Take an initial training step to obtain the initial loss and values, which
    # will define the shape(s) of the `TensorArray`(s) that we create to hold
    # the results, and are used to initialize the convergence criterion.
    # This will trigger tf.function tracing of `optimizer_step_fn`, which is
    # then reused inside the training loop (i.e., it is only traced once).
    optimizer_step_fn = _make_optimizer_step_fn(
        loss_fn=loss_fn, optimizer=optimizer,
        trainable_variables=trainable_variables)
    initial_loss, initial_grads, initial_parameters = optimizer_step_fn(
        seed=seed)
    has_converged = None
    initial_convergence_criterion_state = None
    if convergence_criterion is not None:
      has_converged = tf.zeros(tf.shape(initial_loss), dtype=tf.bool)
      initial_convergence_criterion_state = convergence_criterion.bootstrap(
          initial_loss, initial_grads, initial_parameters)
    initial_traced_values = trace_fn(
        MinimizeTraceableQuantities(
            loss=initial_loss,
            gradients=initial_grads,
            parameters=initial_parameters,
            step=0,
            has_converged=has_converged,
            convergence_criterion_state=initial_convergence_criterion_state))

    trace_arrays = _initialize_arrays(
        initial_values=initial_traced_values,
        num_steps=num_steps,
        truncate_at_convergence=(convergence_criterion is not None
                                 and not return_full_length_trace))

    # Run the optimization loop.
    with tf.control_dependencies([initial_loss]):
      potential_loop_vars = (1, seed, trace_arrays, has_converged,
                             initial_convergence_criterion_state)
      results = tf.while_loop(
          cond=lambda *args: tf.logical_not(convergence_detected(*args)),  # pylint: disable=no-value-for-parameter
          body=_make_training_loop_body(
              optimizer_step_fn=optimizer_step_fn,
              convergence_criterion=convergence_criterion,
              trace_fn=trace_fn),
          loop_vars=[x for x in potential_loop_vars if x is not None],
          parallel_iterations=1,
          maximum_iterations=num_steps - 1)
      indices, _, trace_arrays = results[:3]  # Guaranteed to be present.

      if convergence_criterion is not None and return_full_length_trace:
        # Fill out the trace by tiling the last written values.
        last_written_idx = tf.reduce_max(indices) - 1
        trace_arrays = tf.nest.map_structure(
            lambda ta: _tile_last_written_value(ta, last_written_idx),
            trace_arrays)

    return tf.nest.map_structure(lambda array: array.stack(), trace_arrays)

