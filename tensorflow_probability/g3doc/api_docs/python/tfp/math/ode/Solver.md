<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.ode.Solver" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="solve"/>
</div>

# tfp.math.ode.Solver


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/math/ode/base.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



## Class `Solver`

Base class for an ODE solver.



<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/math/ode/base.py">View source</a>

``` python
__init__(
    use_pfor_to_compute_jacobian,
    validate_args,
    name
)
```

Initialize self.  See help(type(self)) for accurate signature.




## Methods

<h3 id="solve"><code>solve</code></h3>

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/math/ode/base.py">View source</a>

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

### Example

The ODE `dy/dt(t) = dot(A, y(t))` is solved below.

```python
t_init, t0, t1 = 0., 0.5, 1.
y_init = tf.constant([1., 1.], dtype=tf.float64)
A = [[-1., -2.], [-3., -4.]]

def ode_fn(t, y):
  return tf.linalg.matvec(A, y)

results = tfp.math.ode.BDF.solve(ode_fn, t_init, y_init,
                                 solution_times=[t0, t1])
y0 = results.states[0]  # == dot(matrix_exp(A * t0), y_init)
y1 = results.states[1]  # == dot(matrix_exp(A * t1), y_init)
```

Using instead `solution_times=tfp.math.ode.ChosenBySolver(final_time=1.)`
yields the state at various times between `t_init` and `final_time` chosen
automatically by the solver. In this case, `results.states[i]` is the state
at time `results.times[i]`.

#### Gradient

The gradient of the result is computed using the adjoint sensitivity method
described in [Chen et al. (2018)][1].

```python
grad = tf.gradients(y1, y0) # == dot(e, J)
# J is the Jacobian of y1 with respect to y0. In this case, J = exp(A * t1).
# e = [1, ..., 1] is the row vector of ones.
```

#### References

[1]: Chen, Tian Qi, et al. "Neural ordinary differential equations."
     Advances in Neural Information Processing Systems. 2018.

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
  pass <a href="../../../tfp/math/ode/ChosenBySolver.md"><code>tfp.math.ode.ChosenBySolver(final_time)</code></a> where `final_time` is a
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




