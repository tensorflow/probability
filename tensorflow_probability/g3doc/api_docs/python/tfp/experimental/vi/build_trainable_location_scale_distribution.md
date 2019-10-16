<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.vi.build_trainable_location_scale_distribution" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.vi.build_trainable_location_scale_distribution


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/vi/surrogate_posteriors.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Builds a variational distribution from a location-scale family.

``` python
tfp.experimental.vi.build_trainable_location_scale_distribution(
    initial_loc,
    initial_scale,
    event_ndims,
    distribution_fn=tfp.distributions.Normal,
    validate_args=False,
    name=None
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`initial_loc`</b>: Float `Tensor` initial location.
* <b>`initial_scale`</b>: Float `Tensor` initial scale.
* <b>`event_ndims`</b>: Integer `Tensor` number of event dimensions in `initial_loc`.
* <b>`distribution_fn`</b>: Optional constructor for a `tfd.Distribution` instance
  in a location-scale family. This should have signature `dist =
  distribution_fn(loc, scale, validate_args)`.
  Default value: `tfd.Normal`.
* <b>`validate_args`</b>: Python `bool`. Whether to validate input with asserts. This
  imposes a runtime cost. If `validate_args` is `False`, and the inputs are
  invalid, correct behavior is not guaranteed.
  Default value: `False`.
* <b>`name`</b>: Python `str` name prefixed to ops created by this function.
  Default value: `None` (i.e.,
    'build_trainable_location_scale_distribution').

#### Returns:


* <b>`posterior_dist`</b>: A `tfd.Distribution` instance.