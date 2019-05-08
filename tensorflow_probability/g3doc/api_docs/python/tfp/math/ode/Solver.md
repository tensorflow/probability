<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.ode.Solver" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="solve"/>
</div>

# tfp.math.ode.Solver

## Class `Solver`

Base class for an ODE solver.





Defined in [`python/math/ode/base.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/math/ode/base.py).

<!-- Placeholder for "Used in" -->


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

Solves an initial value problem.

An initial value problem consists of a system of ODEs and an initial
condition:

```none
dy/dt(t) = ode_fn(t, y(t))
y(initial_time) = initial_state
```

Here, `t` (also called time) is a scalar float `Tensor` and `y(t)` (also
called the state at time `t`) is an N-D float or complex `Tensor`.

#### Args:

* <b>`ode_fn`</b>: Function of the form `ode_fn(t, y)`. The input `t` is a scalar
  float `Tensor`. The input `y` and output are both `Tensor`s with the
  same shape and `dtype` as `initial_state`.
* <b>`initial_time`</b>: Scalar float `Tensor` specifying the initial time.
* <b>`initial_state`</b>: N-D float or complex `Tensor` specifying the initial state.
  The `dtype` of `initial_state` must be complex for problems with
  complex-valued states (even if the initial state is real).
* <b>`solution_times`</b>: 1-D float `Tensor` specifying a list of times. The solver
  stores the computed state at each of these times in the returned
  `Results` object. Must satisfy `initial_time <= solution_times[0]` and
  `solution_times[i] < solution_times[i+1]`. Alternatively, the user can
  pass `tfp.math.ode.ChosenBySolver(final_time)` where `final_time` is a
  scalar float `Tensor` satisfying `initial_time < final_time`. Doing so
  requests that the solver automatically choose suitable times up to and
  including `final_time` at which to store the computed state.
* <b>`jacobian_fn`</b>: Optional function of the form `jacobian_fn(t, y)`. The input
  `t` is a scalar float `Tensor`. The input `y` has the same shape and
  `dtype` as `initial_state`. The output is a 2N-D `Tensor` whose shape is
  `initial_state.shape + initial_state.shape` and whose `dtype` is the
  same as `initial_state`. In particular, the `(i1, ..., iN, j1, ...,
  jN)`-th entry of `jacobian_fn(t, y)` is the derivative of the `(i1, ...,
  iN)`-th entry of `ode_fn(t, y)` with respect to the `(j1, ..., jN)`-th
  entry of `y`. If this argument is left unspecified, the solver
  automatically computes the Jacobian if and when it is needed.
  Default value: `None`.
* <b>`jacobian_sparsity`</b>: Optional 2N-D boolean `Tensor` whose shape is
  `initial_state.shape + initial_state.shape` specifying the sparsity
  pattern of the Jacobian. This argument is ignored if `jacobian_fn` is
  specified.
  Default value: `None`.
* <b>`batch_ndims`</b>: Optional nonnegative integer. When specified, the first
  `batch_ndims` dimensions of `initial_state` are batch dimensions.
  Default value: `None`.
* <b>`previous_solver_internal_state`</b>: Optional solver-specific argument used to
  warm-start this invocation of `solve`.
  Default value: `None`.


#### Returns:

Object of type `Results`.



