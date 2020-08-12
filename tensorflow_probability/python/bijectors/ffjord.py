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
"""FFJORD bijector class."""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import cache_util
from tensorflow_probability.python.internal import prefer_static


# TODO(b/144156734) Consider moving trace estimators to stand alone module.
def trace_jacobian_hutchinson(
    ode_fn,
    state_shape,
    dtype,
    sample_fn=tf.random.normal,
    num_samples=1,
    seed=None):
  """Generates a function that computes `ode_fn` and expectation of the trace.

  Uses Hutchinson's trick to estimate the trace of the Jacobian using automatic
  differentiation. This is the approach used in the original FFJORD paper [1].
  This method computes unreduced trace, as reduction is performed inside of the
  bijector class.

  The trace estimate is obtained by computing
  ```None
  Tr[A] approx_equal sum_{i} r[i]^{T} @ A @ r[i]; r[i] - gaussian sample.
  ```

  For details on the original work see [2].

  Args:
    ode_fn: `Callable(time, state)` that computes time derivative.
    state_shape: `TensorShape` representing the shape of the state.
    dtype: ``tf.DType` object representing the dtype of `state` tensor.
    sample_fn: `Callable(shape, dtype, seed)` that generates random samples with
      zero mean and covariance of an identity matrix.
      Default value: `tf.random.normal`
    num_samples: `Integer` number of random samples to use for trace estimation.
      Default value: '1'
    seed: 'Integer' seed for random number generator.

  Returns:
    augmented_ode_fn: `Callable(time, (state, log_det_jac))` that computes
      augmented time derivative `(state_time_derivative, trace_estimation)`.

  #### References

  [1]:  Grathwohl, W., Chen, R. T., Betterncourt, J., Sutskever, I.,
        & Duvenaud, D. (2018). Ffjord: Free-form continuous dynamics for
        scalable reversible generative models. arXiv preprint arXiv:1810.01367.
        http://arxiv.org.abs/1810.01367

  [2]:  Hutchinson, M. F. (1989). A stochastic estimator of the trace of the
        influence matrix for Laplacian smoothing splines. Communications in
        Statistics-Simulation and Computation, 18(3), 1059-1076.
  """

  random_samples = sample_fn(
      prefer_static.concat([[num_samples], state_shape], axis=0),
      dtype=dtype, seed=seed)

  def augmented_ode_fn(time, state_log_det_jac):
    """Computes both time derivative and trace of the jacobian."""
    state, _ = state_log_det_jac
    with tf.GradientTape(persistent=True,
                         watch_accessed_variables=False) as tape:
      tape.watch(state)
      state_time_derivative = ode_fn(time, state)

    def estimate_trace(random_sample):
      """Computes stochastic trace estimate based on a single random_sample."""
      #  We use first use gradient with `output_gradients` to compute the
      #  jacobian-value-product and then take a dot product with the random
      #  sample to obtain the trace estimate as formula above.
      jvp = tape.gradient(state_time_derivative, state, random_sample)
      return random_sample * jvp

    # TODO(dkochkov) switch to vectorized_map once more features are supported.
    results = tf.map_fn(estimate_trace, random_samples)
    trace_estimates = tf.reduce_mean(results, axis=0)
    return state_time_derivative, trace_estimates

  return augmented_ode_fn


def trace_jacobian_exact(ode_fn, state_shape, dtype):
  """Generates a function that computes `ode_fn` and trace of the jacobian.

  Augments provided `ode_fn` with explicit computation of the trace of the
  jacobian. This approach scales quadratically with the number of dimensions.
  This method computes unreduced trace, as reduction is performed inside of the
  bijector class.

  Args:
    ode_fn: `Callable(time, state)` that computes time derivative.
    state_shape: `TensorShape` representing the shape of the state.
    dtype: ``tf.DType` object representing the dtype of `state` tensor.

  Returns:
    augmented_ode_fn: `Callable(time, (state, log_det_jac))` that computes
      augmented time derivative `(state_time_derivative, trace_estimation)`.
  """
  del state_shape, dtype  # Not used by trace_jacobian_exact

  def augmented_ode_fn(time, state_log_det_jac):
    """Computes both time derivative and trace of the jacobian."""
    state, _ = state_log_det_jac
    ode_fn_with_time = lambda x: ode_fn(time, x)
    batch_shape = [prefer_static.size0(state)]
    state_time_derivative, diag_jac = tfp_math.diag_jacobian(
        xs=state, fn=ode_fn_with_time, sample_shape=batch_shape)
    # tfp_math.diag_jacobian returns lists
    if isinstance(state_time_derivative, list):
      state_time_derivative = state_time_derivative[0]
    if isinstance(diag_jac, list):
      diag_jac = diag_jac[0]
    trace_value = diag_jac
    return state_time_derivative, trace_value

  return augmented_ode_fn

# TODO(b/142901683) Add Mutually Unbiased Bases for trace estimation.


class FFJORD(bijector.Bijector):
  """Implements a continuous normalizing flow X->Y defined via an ODE.


  This bijector implements a continuous dynamics transformation
  parameterized by a differential equation, where initial and terminal
  conditions correspond to domain (X) and image (Y) i.e.

  ```None
  d/dt[state(t)]=state_time_derivative_fn(t, state(t))
  state(initial_time) = X
  state(final_time) = Y
  ```

  For this transformation the value of `log_det_jacobian` follows another
  differential equation, reducing it to computation of the trace of the jacbian
  along the trajectory

  ```None
  state_time_derivative = state_time_derivative_fn(t, state(t))
  d/dt[log_det_jac(t)] = Tr(jacobian(state_time_derivative, state(t)))
  ```

  FFJORD constructor takes two functions `ode_solve_fn` and
  `trace_augmentation_fn` arguments that customize integration of the
  differential equation and trace estimation.

  Differential equation integration is performed by a call to `ode_solve_fn`.
  Custom `ode_solve_fn` must accept the following arguments:
  * ode_fn(time, state): Differential equation to be solved.
  * initial_time: Scalar float or floating Tensor representing the initial time.
  * initial_state: Floating Tensor representing the initial state.
  * solution_times: 1D floating Tensor of solution times.

  And return a Tensor of shape [solution_times.shape, initial_state.shape]
  representing state values evaluated at `solution_times`. In addition
  `ode_solve_fn` must support nested structures. For more details see the
  interface of `tfp.math.ode.Solver.solve()`.

  Trace estimation is computed simultaneously with `state_time_derivative`
  using `augmented_state_time_derivative_fn` that is generated by
  `trace_augmentation_fn`. `trace_augmentation_fn` takes
  `state_time_derivative_fn`, `state.shape` and `state.dtype` arguments and
  returns a `augmented_state_time_derivative_fn` callable that computes both
  `state_time_derivative` and unreduced `trace_estimation`.

  #### Custom `ode_solve_fn` and `trace_augmentation_fn` examples:
  ```python
  # custom_solver_fn: `callable(f, t_initial, t_solutions, y_initial, ...)`
  # custom_solver_kwargs: Additional arguments to pass to custom_solver_fn.
  def ode_solve_fn(ode_fn, initial_time, initial_state, solution_times):
    results = custom_solver_fn(ode_fn, initial_time, solution_times,
                               initial_state, **custom_solver_kwargs)
    return results

  ffjord = tfb.FFJORD(state_time_derivative_fn, ode_solve_fn=ode_solve_fn)
  ```

  ```python
  # state_time_derivative_fn: `callable(time, state)`
  # trace_jac_fn: `callable(time, state)` unreduced jacobian trace function

  def trace_augmentation_fn(ode_fn, state_shape, state_dtype):
    def augmented_ode_fn(time, state):
      return ode_fn(time, state), trace_jac_fn(time, state)
    return augmented_ode_fn

  ffjord = tfb.FFJORD(state_time_derivative_fn,
                      trace_augmentation_fn=trace_augmentation_fn)
  ```

  For more details on FFJORD and continous normalizing flows see [1], [2].

  #### Usage example:
  ```python
  tfd = tfp.distributions
  tfb = tfp.bijectors
  # state_time_derivative_fn: `Callable(time, state)` -> state_time_derivative
  # e.g. Neural network with inputs and outputs of the same shapes and dtypes.

  bijector = tfb.FFJORD(state_time_derivative_fn=state_time_derivative_fn)
  y = bijector.forward(x)  # forward mapping
  x = bijector.inverse(y)  # inverse mapping
  base = tfd.Normal(tf.zeros_like(x), tf.ones_like(x))  # Base distribution
  transformed_distribution = tfd.TransformedDistribution(base, bijector)
  ```

  #### References

  [1]:  Chen, T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018).
        Neural ordinary differential equations. In Advances in neural
        information processing systems (pp. 6571-6583)

  [2]:  Grathwohl, W., Chen, R. T., Betterncourt, J., Sutskever, I.,
        & Duvenaud, D. (2018). Ffjord: Free-form continuous dynamics for
        scalable reversible generative models. arXiv preprint
        arXiv:1810.01367.
        http://arxiv.org.abs/1810.01367
  """

  def __init__(
      self,
      state_time_derivative_fn,
      ode_solve_fn=None,
      trace_augmentation_fn=trace_jacobian_hutchinson,
      initial_time=0.,
      final_time=1.,
      validate_args=False,
      dtype=tf.float32,
      name='ffjord'):
    """Constructs a FFJORD bijector.

    Args:
      state_time_derivative_fn: Python `callable` taking arguments `time`
        (a scalar representing time) and `state` (a Tensor representing the
        state at given `time`) returning the time derivative of the `state` at
        given `time`.
      ode_solve_fn: Python `callable` taking arguments `ode_fn` (same as
        `state_time_derivative_fn` above), `initial_time` (a scalar representing
        the initial time of integration), `initial_state` (a Tensor of floating
        dtype represents the initial state) and `solution_times` (1D Tensor of
        floating dtype representing time at which to obtain the solution)
        returning a Tensor of shape [time_axis, initial_state.shape]. Will take
        `[final_time]` as the `solution_times` argument and
        `state_time_derivative_fn` as `ode_fn` argument. For details on
        providing custom `ode_solve_fn` see class docstring.
        If `None` a DormandPrince solver from `tfp.math.ode` is used.
        Default value: None
      trace_augmentation_fn: Python `callable` taking arguments `ode_fn` (
        python `callable` same as `state_time_derivative_fn` above),
        `state_shape` (TensorShape of a the state), `dtype` (same as dtype of
        the state) and returning a python `callable` taking arguments `time`
        (a scalar representing the time at which the function is evaluted),
        `state` (a Tensor representing the state at given `time`) that computes
        a tuple (`ode_fn(time, state)`, `jacobian_trace_estimation`).
        `jacobian_trace_estimation` should represent trace of the jacobian of
        `ode_fn` with respect to `state`. `state_time_derivative_fn` will be
        passed as `ode_fn` argument. For details on providing custom
        `trace_augmentation_fn` see class docstring.
        Default value: tfp.bijectors.ffjord.trace_jacobian_hutchinson
      initial_time: Scalar float representing time to which the `x` value of the
        bijector corresponds to. Passed as `initial_time` to `ode_solve_fn`.
        For default solver can be Python `float` or floating scalar `Tensor`.
        Default value: 0.
      final_time: Scalar float representing time to which the `y` value of the
        bijector corresponds to. Passed as `solution_times` to `ode_solve_fn`.
        For default solver can be Python `float` or floating scalar `Tensor`.
        Default value: 1.
      validate_args: Python 'bool' indicating whether to validate input.
        Default value: False
      dtype: `tf.DType` to prefer when converting args to `Tensor`s. Else, we
        fall back to a common dtype inferred from the args, finally falling
        back to float32.
      name: Python `str` name prefixed to Ops created by this function.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      self._initial_time = initial_time
      self._final_time = final_time
      self._ode_solve_fn = ode_solve_fn
      if self._ode_solve_fn is None:
        self._ode_solver = tfp_math.ode.DormandPrince()
        self._ode_solve_fn = self._ode_solver.solve
      self._trace_augmentation_fn = trace_augmentation_fn
      self._state_time_derivative_fn = state_time_derivative_fn

      def inverse_state_time_derivative(time, state):
        return -state_time_derivative_fn(self._final_time - time, state)

      self._inv_state_time_derivative_fn = inverse_state_time_derivative
      super(FFJORD, self).__init__(
          forward_min_event_ndims=0,
          dtype=dtype,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  def _setup_cache(self):
    """Overrides the bijector cache to update attrs on forward/inverse."""
    return cache_util.BijectorCache(
        forward_impl=self._augmented_forward,
        inverse_impl=self._augmented_inverse,
        cache_type=cache_util.CachedDirectedFunctionWithGreedyAttrs)

  def _solve_ode(self, ode_fn, state):
    """Solves the initial value problem defined by `ode_fn`.

    Args:
      ode_fn: `Callable(time, state)` that represents state time derivative.
      state: A `Tensor` representing initial state.

    Returns:
      final_state: `Tensor` of the same shape and dtype as `state` representing
        the solution of ODE defined by `ode_fn` at `self._final_time`.
    """
    integration_result = self._ode_solve_fn(
        ode_fn=ode_fn,
        initial_time=self._initial_time,
        initial_state=state,
        solution_times=[self._final_time])
    final_state = tf.nest.map_structure(
        lambda x: x[-1], integration_result.states)
    return final_state

  def _augmented_forward(self, x):
    """Computes forward and forward_log_det_jacobian transformations."""
    augmented_ode_fn = self._trace_augmentation_fn(
        self._state_time_derivative_fn, x.shape, x.dtype)
    augmented_x = (x, tf.zeros(shape=x.shape, dtype=x.dtype))
    y, fldj = self._solve_ode(augmented_ode_fn, augmented_x)
    return y, {'ildj': -fldj, 'fldj': fldj}

  def _augmented_inverse(self, y):
    """Computes inverse and inverse_log_det_jacobian transformations."""
    augmented_inv_ode_fn = self._trace_augmentation_fn(
        self._inv_state_time_derivative_fn, y.shape, y.dtype)
    augmented_y = (y, tf.zeros(shape=y.shape, dtype=y.dtype))
    x, ildj = self._solve_ode(augmented_inv_ode_fn, augmented_y)
    return x, {'ildj': ildj, 'fldj': -ildj}

  def _forward(self, x):
    y, _ = self._augmented_forward(x)
    return y

  def _inverse(self, y):
    x, _ = self._augmented_inverse(y)
    return x

  def _forward_log_det_jacobian(self, x):
    cached = self._cache.forward.attributes(x)
    # If LDJ isn't in the cache, call forward once.
    if 'fldj' not in cached:
      _, attrs = self._augmented_forward(x)
      cached.update(attrs)
    return cached['fldj']

  def _inverse_log_det_jacobian(self, y):
    cached = self._cache.inverse.attributes(y)
    # If LDJ isn't in the cache, call inverse once.
    if 'ildj' not in cached:
      _, attrs = self._augmented_inverse(y)
      cached.update(attrs)
    return cached['ildj']
