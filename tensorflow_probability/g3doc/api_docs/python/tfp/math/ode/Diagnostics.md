<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.ode.Diagnostics" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="num_jacobian_evaluations"/>
<meta itemprop="property" content="num_matrix_factorizations"/>
<meta itemprop="property" content="num_ode_fn_evaluations"/>
<meta itemprop="property" content="status"/>
<meta itemprop="property" content="success"/>
</div>

# tfp.math.ode.Diagnostics

## Class `Diagnostics`

Diagnostics returned by a Solver.





Defined in [`python/math/ode/base.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/math/ode/base.py).

<!-- Placeholder for "Used in" -->


## Properties

<h3 id="num_jacobian_evaluations"><code>num_jacobian_evaluations</code></h3>

Number of Jacobian evaluations.


#### Returns:


* <b>`num_jacobian_evaluations`</b>: Scalar integer `Tensor` containing number of
  Jacobian evaluations.

<h3 id="num_matrix_factorizations"><code>num_matrix_factorizations</code></h3>

Number of matrix factorizations.


#### Returns:


* <b>`num_matrix_factorizations`</b>: Scalar integer `Tensor` containing the number
  of matrix factorizations.

<h3 id="num_ode_fn_evaluations"><code>num_ode_fn_evaluations</code></h3>

Number of function evaluations.


#### Returns:


* <b>`num_ode_fn_evaluations`</b>: Scalar integer `Tensor` containing the number of
  function evaluations.

<h3 id="status"><code>status</code></h3>

Completion status.


#### Returns:


* <b>`status`</b>: Scalar integer `Tensor` containing the reason for termination. -1
  on failure, 1 on termination by an event, and 0 otherwise.

<h3 id="success"><code>success</code></h3>

Boolean indicating whether or not the method succeeded.


#### Returns:


* <b>`success`</b>: Boolean `Tensor` equivalent to `status >= 0`.



