<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.ode.ChosenBySolver" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="final_time"/>
<meta itemprop="property" content="__new__"/>
</div>

# tfp.math.ode.ChosenBySolver

## Class `ChosenBySolver`

Sentinel used to modify the behaviour of the `solve` method of a solver.





Defined in [`python/math/ode/base.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/math/ode/base.py).

<!-- Placeholder for "Used in" -->

Can be passed as the `solution_times` argument in the `solve` method of a
solver. Doing so requests that the solver automatically choose suitable times
at which to store the computed state (see `tfp.math.ode.Base.solve`).

#### Properties:

* <b>`final_time`</b>: Scalar float `Tensor` specifying the largest time at which to
  store the computed state.

<h2 id="__new__"><code>__new__</code></h2>

``` python
__new__(
    _cls,
    final_time
)
```

Create new instance of ChosenBySolver(final_time,)



## Properties

<h3 id="final_time"><code>final_time</code></h3>





