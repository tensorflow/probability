<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.ode.BDF" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="solve"/>
</div>

# tfp.math.ode.BDF

## Class `BDF`

Backward differentiation formula (BDF) solver for stiff ODEs.

Inherits From: [`Solver`](../../../tfp/math/ode/Solver.md)



Defined in [`python/math/ode/bdf.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/math/ode/bdf.py).

<!-- Placeholder for "Used in" -->

Implements the solver described in [Shampine and Reichelt (1997)][1], a
variable step size, variable order (VSVO) BDF integrator with order varying
between 1 and 5.

### Algorithm details

Each step involves solving the following nonlinear equation by Newton's
method:
```none
0 = 1/1 * BDF(1, y)[n+1] + ... + 1/k * BDF(k, y)[n+1]
  - h ode_fn(t[n+1], y[n+1])
  - bdf_coefficients[k-1] * (1/1 + ... + 1/k) * (y[n+1] - y[n] - BDF(1, y)[n]
                                                        -  ... - BDF(k, y)[n])
```
where `k >= 1` is the current order of the integrator, `h` is the current step
size, `bdf_coefficients` is a list of numbers that parameterizes the method,
and `BDF(m, y)` is the `m`-th order backward difference of the vector `y`. In
particular, `BDF(0, y)[n] = y[n]` and
`BDF(m + 1, y)[n] = BDF(m, y)[n] - BDF(m, y)[n - 1]` for `m >= 0`.

Newton's method can fail because
* the method has exceeded the maximum number of iterations,
* the method is converging too slowly, or
* the method is not expected to converge
(the last two conditions are determined by approximating the Lipschitz
constant associated with the iteration).

When `evaluate_jacobian_lazily` is `True`, the solver avoids evaluating the
Jacobian of the dynamics function as much as possible. In particular, Newton's
method will try to use the Jacobian from a previous integration step. If
Newton's method fails with an out-of-date Jacobian, the Jacobian is
re-evaluated and Newton's method is restarted. If Newton's method fails and
the Jacobian is already up-to-date, then the step size is decreased and
Newton's method is restarted.

Even if Newton's method converges, the solution it generates can still be
rejected if it exceeds the specified error tolerance due to truncation error.
In this case, the step size is decreased and Newton's method is restarted.

#### References

[1]: Lawrence F. Shampine and Mark W. Reichelt. The MATLAB ODE Suite. _SIAM
     Journal on Scientific Computing_ 18(1):1-22, 1997.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    rtol=0.001,
    atol=1e-06,
    first_step_size=None,
    safety_factor=0.9,
    min_step_size_factor=0.1,
    max_step_size_factor=10.0,
    max_num_steps=None,
    max_order=bdf_util.MAX_ORDER,
    max_num_newton_iters=4,
    newton_tol_factor=0.1,
    newton_step_size_factor=0.5,
    bdf_coefficients=(-0.185, -1.0 / 9.0, -0.0823, -0.0415, 0.0),
    evaluate_jacobian_lazily=False,
    use_pfor_to_compute_jacobian=True,
    validate_args=False,
    name='bdf'
)
```

Initializes the solver.

#### Args:

* <b>`rtol`</b>: Optional float `Tensor` specifying an upper bound on relative error,
  per element of the dependent variable. The error tolerance for the next
  step is `tol = atol + rtol * abs(state)` where `state` is the computed
  state at the current step (see also `atol`). The next step is rejected
  if it incurs a local truncation error larger than `tol`.
  Default value: `1e-3`.
* <b>`atol`</b>: Optional float `Tensor` specifying an upper bound on absolute error,
  per element of the dependent variable (see also `rtol`).
  Default value: `1e-6`.
* <b>`first_step_size`</b>: Optional scalar float `Tensor` specifying the size of the
  first step. If unspecified, the size is chosen automatically.
  Default value: `None`.
* <b>`safety_factor`</b>: Scalar positive float `Tensor`. When Newton's method
  converges, the solver may choose to update the step size by applying a
  multiplicative factor to the current step size. This factor is `factor =
  clamp(factor_unclamped, min_step_size_factor, max_step_size_factor)`
  where `factor_unclamped = error_ratio**(-1. / (order + 1)) *
  safety_factor` (see also `min_step_size_factor` and
  `max_step_size_factor`). A small (respectively, large) value for the
  safety factor causes the solver to take smaller (respectively, larger)
  step sizes. A value larger than one, though not explicitly prohibited,
  is discouraged.
  Default value: `0.9`.
* <b>`min_step_size_factor`</b>: Scalar float `Tensor` (see `safety_factor`).
  Default value: `0.1`.
* <b>`max_step_size_factor`</b>: Scalar float `Tensor` (see `safety_factor`).
  Default value: `10.`.
* <b>`max_num_steps`</b>: Optional scalar integer `Tensor` specifying the maximum
  number of steps allowed (including rejected steps). If unspecified,
  there is no upper bound on the number of steps.
  Default value: `None`.
* <b>`max_order`</b>: Scalar integer `Tensor` taking values between 1 and 5
  (inclusive) specifying the maximum BDF order.
  Default value: `5`.
* <b>`max_num_newton_iters`</b>: Optional scalar integer `Tensor` specifying the
  maximum number of iterations per invocation of Newton's method. If
  unspecified, there is no upper bound on the number iterations.
  Default value: `4`.
* <b>`newton_tol_factor`</b>: Scalar float `Tensor` used to determine the stopping
  condition for Newton's method. In particular, Newton's method terminates
  when the distance to the root is estimated to be less than
  `newton_tol_factor * norm(atol + rtol * abs(state))` where `state` is
  the computed state at the current step.
  Default value: `0.1`.
* <b>`newton_step_size_factor`</b>: Scalar float `Tensor` specifying a multiplicative
  factor applied to the size of the integration step when Newton's method
  fails.
  Default value: `0.5`.
* <b>`bdf_coefficients`</b>: 1-D float `Tensor` with 5 entries that parameterize the
  solver. The default values are those proposed in [1].
  Default value: `(-0.1850, -1. / 9., -0.0823, -0.0415, 0.)`.
* <b>`evaluate_jacobian_lazily`</b>: Optional boolean specifying whether the Jacobian
  should be evaluated at each point in time or as needed (i.e., lazily).
  Default value: `True`.
* <b>`use_pfor_to_compute_jacobian`</b>: Boolean specifying whether or not to use
  parallel for in computing the Jacobian when `jacobian_fn` is not
  specified.
  Default value: `True`.
* <b>`validate_args`</b>: Whether to validate input with asserts. If `validate_args`
  is `False` and the inputs are invalid, correct behavior is not
  guaranteed.
  Default value: `False`.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
  Default value: `None` (i.e., 'bdf').



## Methods

<h3 id="solve"><code>solve</code></h3>

``` python
solve(
    ode_fn,
    initial_time,
    initial_state,
    solution_times,
    jacobian_fn=None,
    jacobian_sparsity=None,
    batch_ndims=None,
    previous_solver_internal_state=None
)
```

See <a href="../../../tfp/math/ode/Solver.md#solve"><code>tfp.math.ode.Solver.solve</code></a>.



