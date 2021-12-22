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

import collections
import functools

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import loop_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.math import value_and_gradient

JAX_MODE = False


class MinimizeTraceableQuantities(collections.namedtuple(
    'MinimizeTraceableQuantities',
    ('step', 'loss', 'gradients', 'parameters', 'has_converged',
     'convergence_criterion_state', 'optimizer_state', 'seed'))):
  """Namedtuple of quantities that may be traced from `tfp.math.minimize`.

  These are (in order):

  - `step`: int `Tensor` index (starting from zero) of the current optimization
     step.
  - `loss`: float `Tensor` value returned from the user-provided `loss_fn`.
  - `gradients`: list of `Tensor` gradients of `loss` with respect to the
     parameters.
  - `parameters`: list of `Tensor` values of parameters being optimized. This
     corresponds to `trainable_variables` passed to `minimize`, or
     `init` passed to `minimize_stateless`.
  - `has_converged`: boolean `Tensor` of the same shape as `loss_fn`, with
    `True` values corresponding to loss entries that have converged according
    to the user-provided convergence criterion. If no convergence criterion
    was specified, this is `None`.
  - `convergence_criterion_state`: structure of `Tensor`s containing any
    auxiliary state (e.g., moving averages of loss or other quantities)
    maintained by the user-provided convergence criterion.
  - `optimizer_state`: structure of `Tensor`s containing optional state from
    a user-provided pure optimizer.

  """

_trace_loss = lambda traceable_quantities: traceable_quantities.loss


def _trace_has_converged(trace_fn, batch_convergence_reduce_fn):
  """Augments a trace_fn to record whether the optimization has converged."""
  return (
      lambda tq: (trace_fn(tq), batch_convergence_reduce_fn(tq.has_converged)))


def _truncate_at_has_converged(augmented_traced_values):
  """Truncates trace from `trace_fn` wrapped by `_also_trace_has_converged`."""
  traced_values, global_has_converged = augmented_traced_values
  num_steps = ps.argmax(ps.concat([global_has_converged, [True]], axis=0))
  return tf.nest.map_structure(lambda x: x[:num_steps + 1], traced_values)


def _make_training_loop_body(optimizer_step_fn,
                             convergence_criterion,
                             batch_convergence_reduce_fn,
                             seed_is_none=False):
  """Constructs the training loop body."""

  def unconverged_training_loop_body(previous, step):
    """Invokes the optimizer step and convergence criterion."""
    if seed_is_none:
      seed, next_seed = None, previous.seed
    else:
      seed, next_seed = samplers.split_seed(previous.seed, n=2)
    loss, grads, parameters, optimizer_state = optimizer_step_fn(
        previous.parameters, previous.optimizer_state, seed=seed)
    if convergence_criterion is None:
      has_converged = previous.has_converged
      convergence_criterion_state = previous.convergence_criterion_state
    else:
      (has_converged,
       convergence_criterion_state) = convergence_criterion.one_step(
           step, loss, grads, parameters, previous.convergence_criterion_state)
    return MinimizeTraceableQuantities(
        loss=loss,
        gradients=grads,
        parameters=parameters,
        step=step,
        has_converged=has_converged,
        convergence_criterion_state=convergence_criterion_state,
        optimizer_state=optimizer_state,
        seed=next_seed)

  def training_loop_body(previous_traced_values, step):
    return ps.cond(
        batch_convergence_reduce_fn(previous_traced_values.has_converged),
        lambda: previous_traced_values,
        lambda: unconverged_training_loop_body(previous_traced_values, step))

  return training_loop_body


def _minimize_common(num_steps,
                     optimizer_step_fn,
                     initial_parameters,
                     initial_optimizer_state,
                     convergence_criterion,
                     batch_convergence_reduce_fn,
                     trace_fn,
                     return_full_length_trace=True,
                     jit_compile=False,
                     seed=None,
                     name='minimize'):
  """General-purpose optimization loop."""
  if jit_compile:
    # Run the entire minimization inside a jit-compiled function. This is
    # typically faster than jit-compiling the individual steps.
    kwargs = dict(locals())
    kwargs['jit_compile'] = False
    @tf.function(autograph=False, jit_compile=True)
    def run_jitted_minimize():
      return _minimize_common(**kwargs)
    return run_jitted_minimize()

  # Main optimization routine.
  with tf.name_scope(name) as name:
    seed_is_none = seed is None
    if not seed_is_none:
      seed = samplers.sanitize_seed(seed, salt='minimize')

    if not return_full_length_trace:
      # Augment trace to record convergence info, so we can truncate it later.
      trace_fn = _trace_has_converged(trace_fn, batch_convergence_reduce_fn)

    # Take an initial training step to obtain the initial loss and values, which
    # are used to initialize the convergence criterion. This will trigger
    # tf.function tracing of `optimizer_step_fn`, which is
    # then reused inside the training loop (i.e., it is only traced once).
    (initial_loss,
     initial_grads,
     initial_parameters,
     initial_optimizer_state) = optimizer_step_fn(
         parameters=initial_parameters,
         optimizer_state=initial_optimizer_state,
         seed=seed)

    initial_convergence_criterion_state = ()
    if convergence_criterion is not None:
      initial_convergence_criterion_state = convergence_criterion.bootstrap(
          initial_loss, initial_grads, initial_parameters)

    with tf.control_dependencies([initial_loss]):
      has_converged = tf.zeros(tf.shape(initial_loss), dtype=tf.bool)
      initial_state = MinimizeTraceableQuantities(
          loss=initial_loss,
          gradients=initial_grads,
          parameters=initial_parameters,
          step=0,
          has_converged=has_converged,
          convergence_criterion_state=initial_convergence_criterion_state,
          optimizer_state=initial_optimizer_state,
          seed=-1 if seed_is_none else seed)
      initial_traced_values = tf.nest.map_structure(
          lambda x: tf.convert_to_tensor(x)[tf.newaxis, ...],
          trace_fn(initial_state))

    def run_optimization_loop():
      final_step_traceable_values, loop_traced_values = loop_util.trace_scan(
          loop_fn=_make_training_loop_body(
              optimizer_step_fn=optimizer_step_fn,
              convergence_criterion=convergence_criterion,
              batch_convergence_reduce_fn=batch_convergence_reduce_fn,
              seed_is_none=seed_is_none),
          initial_state=initial_state,
          elems=tf.range(1, num_steps),
          trace_fn=trace_fn)
      traced_values = tf.nest.map_structure(
          lambda t0, t: tf.concat([t0, t], axis=0),
          initial_traced_values,
          loop_traced_values)
      return final_step_traceable_values.parameters, traced_values

    final_parameters, traced_values = ps.cond(
        num_steps > 1,
        run_optimization_loop,
        lambda: (initial_parameters, initial_traced_values))

    if not return_full_length_trace:
      traced_values = _truncate_at_has_converged(traced_values)
    return final_parameters, traced_values


def _make_stateless_optimizer_step_fn(loss_fn, optimizer):
  """Constructs a single step of a pure functional optimizer."""

  @tf.function(autograph=False)
  def optimizer_step(parameters, optimizer_state, seed=None):
    """Runs a single optimization step."""
    try:
      loss, grads = value_and_gradient(functools.partial(loss_fn, seed=seed),
                                       parameters)
    except TypeError:
      loss, grads = value_and_gradient(loss_fn, parameters)
    # Coerce grads to the same sequence type (e.g., namedtuple) as parameters.
    grads = tf.nest.pack_sequence_as(parameters, tf.nest.flatten(grads))
    updates, optimizer_state = optimizer.update(grads,
                                                optimizer_state,
                                                parameters)
    # Apply updates.
    parameters = tf.nest.map_structure(lambda a, b: a + b, parameters, updates)
    return loss, grads, parameters, optimizer_state
  return optimizer_step


def minimize_stateless(loss_fn,
                       init,
                       num_steps,
                       optimizer,
                       convergence_criterion=None,
                       batch_convergence_reduce_fn=tf.reduce_all,
                       trace_fn=_trace_loss,
                       return_full_length_trace=True,
                       jit_compile=False,
                       seed=None,
                       name='minimize_stateless'):
  """Minimize a loss expressed as a pure function of its parameters.

  Args:
    loss_fn: Python callable with signature
      `loss = loss_fn(*init, seed=None)`. The loss function may
      optionally take a `seed` keyword argument, used to specify a per-iteration
      seed for stochastic loss functions (a stateless `Tensor` seed will be
      passed; see `tfp.random.sanitize_seed`).
    init: Tuple of `Tensor` initial parameter values (or nested structures
      of `Tensor` values) passed to the loss function.
    num_steps: Python `int` maximum number of steps to run the optimizer.
    optimizer: Pure functional optimizer to use. This may be an
      `optax.GradientTransformation` instance (in JAX), or any similar object
      that implements methods
      `optimizer_state = optimizer.init(parameters)` and
      `updates, optimizer_state = optimizer.update(grads, optimizer_state,
      parameters)`.
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
      Default value: 'minimize_stateless'.

  Returns:
    final_parameters: Tuple of final parameter values, with the same structure
      and `Tensor` shapes as `init`.
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
  import optax  # Assume JAX backend.

  loss_fn = lambda x: (x - 5.)**2
  final_x, losses = tfp.math.minimize_stateless(
    loss_fn,
    init=0.,
    num_steps=100,
    optimizer=optax.adam(0.1))
  print("optimized value is {} with loss {}".format(final_x, losses[-1]))
  ```

  We can attempt to automatically detect convergence and stop the optimization
  by passing an instance of
  `tfp.optimize.convergence_criteria.ConvergenceCriterion`. For example, to
  stop the optimization once a moving average of the per-step decrease in loss
  drops below `0.01`:

  ```python
  _, losses = tfp.math.minimize_stateless(
    loss_fn,
    init=0.,
    num_steps=1000,
    optimizer=optax.adam(0.1),
    convergence_criterion=(
      tfp.optimizers.convergence_criteria.LossNotDecreasing(atol=0.01)))
  ```

  Here `num_steps=1000` defines an upper bound: the optimization will be
  stopped after 1000 steps even if no convergence is detected.

  In some cases, we may want to track additional context inside the
  optimization. We can do this by defining a custom `trace_fn`. This accepts
  a `tfp.math.MinimizeTraceableQuantities` tuple and returns a structure
  values to trace; these may include the loss, gradients, parameter values,
  or any auxiliary state maintained by the convergence criterion (if any).

  ```python
  trace_fn = lambda traceable_quantities: {
    'loss': traceable_quantities.loss,
    'x': traceable_quantities.parameters}
  _, trace = tfp.math.minimize_stateless(loss_fn,
                                    init=0.,
                                    num_steps=100,
                                    optimizer=optax.adam(0.1),
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
  trace_fn = lambda traceable_quantities: {
    'loss': traceable_quantities.loss,
    'has_converged': traceable_quantities.has_converged}
  _, trace = tfp.math.minimize_stateless(
    loss_fn,
    init=tf.zeros([batch_size]),
    num_steps=100,
    optimizer=optax.adam(0.1),
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
  return _minimize_common(
      num_steps=num_steps,
      optimizer_step_fn=_make_stateless_optimizer_step_fn(
          loss_fn=loss_fn, optimizer=optimizer),
      initial_parameters=init,
      initial_optimizer_state=optimizer.init(init),
      convergence_criterion=convergence_criterion,
      batch_convergence_reduce_fn=batch_convergence_reduce_fn,
      trace_fn=trace_fn,
      return_full_length_trace=return_full_length_trace,
      jit_compile=jit_compile,
      seed=seed,
      name=name)


def _make_stateful_optimizer_step_fn(loss_fn, optimizer, trainable_variables):
  """Constructs a single step of a stateful (`tf.optimizers`) optimizer."""

  @tf.function(autograph=False)
  def optimizer_step(parameters,
                     optimizer_state=None,
                     seed=None):
    """Run a single optimization step."""
    del parameters  # Unused.
    del optimizer_state  # Unused.
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
              [tf.identity(v) for v in watched_variables],
              ())

  return optimizer_step


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
  _, traced_values = _minimize_common(
      num_steps=num_steps,
      optimizer_step_fn=_make_stateful_optimizer_step_fn(
          loss_fn=loss_fn,
          optimizer=optimizer,
          trainable_variables=trainable_variables),
      initial_parameters=(),
      initial_optimizer_state=(),
      convergence_criterion=convergence_criterion,
      batch_convergence_reduce_fn=batch_convergence_reduce_fn,
      trace_fn=trace_fn,
      return_full_length_trace=return_full_length_trace,
      jit_compile=jit_compile,
      seed=seed,
      name=name)
  return traced_values


