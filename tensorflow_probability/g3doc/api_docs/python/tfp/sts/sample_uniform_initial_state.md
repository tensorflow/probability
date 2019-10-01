<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.sts.sample_uniform_initial_state" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.sts.sample_uniform_initial_state


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/sts/fitting.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Initialize from a uniform [-2, 2] distribution in unconstrained space.

``` python
tfp.sts.sample_uniform_initial_state(
    parameter,
    return_constrained=True,
    init_sample_shape=(),
    seed=None
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`parameter`</b>: `sts.Parameter` named tuple instance.
* <b>`return_constrained`</b>: if `True`, re-applies the constraining bijector
  to return initializations in the original domain. Otherwise, returns
  initializations in the unconstrained space.
  Default value: `True`.
* <b>`init_sample_shape`</b>: `sample_shape` of the sampled initializations.
  Default value: `[]`.
* <b>`seed`</b>: Python integer to seed the random number generator.


#### Returns:


* <b>`uniform_initializer`</b>: `Tensor` of shape `concat([init_sample_shape,
parameter.prior.batch_shape, transformed_event_shape])`, where
`transformed_event_shape` is `parameter.prior.event_shape`, if
`return_constrained=True`, and otherwise it is
`parameter.bijector.inverse_event_shape(parameteter.prior.event_shape)`.