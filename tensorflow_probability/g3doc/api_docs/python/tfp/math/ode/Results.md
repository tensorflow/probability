<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.ode.Results" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="times"/>
<meta itemprop="property" content="states"/>
<meta itemprop="property" content="diagnostics"/>
<meta itemprop="property" content="__new__"/>
</div>

# tfp.math.ode.Results


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/math/ode/base.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



## Class `Results`

Results returned by a Solver.



<!-- Placeholder for "Used in" -->


#### Properties:


* <b>`times`</b>: A 1-D float `Tensor` satisfying `times[i] < times[i+1]`.
* <b>`states`</b>: A (1+N)-D `Tensor` containing the state at each time. In particular,
  `states[i]` is the state at time `times[i]`.
* <b>`diagnostics`</b>: Object of type `Diagnostics` containing performance
  information.
* <b>`solver_internal_state`</b>: Solver-specific object which can be used to
warm-start the solver on a future invocation of `solve`.

<h2 id="__new__"><code>__new__</code></h2>

``` python
__new__(
    _cls,
    times,
    states,
    diagnostics,
    solver_internal_state
)
```

Create new instance of Results(times, states, diagnostics, solver_internal_state)




## Properties

<h3 id="times"><code>times</code></h3>




<h3 id="states"><code>states</code></h3>




<h3 id="diagnostics"><code>diagnostics</code></h3>






