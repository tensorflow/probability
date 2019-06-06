<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.Mixture" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.Mixture

Create a random variable for Mixture.

``` python
tfp.edward2.Mixture(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See Mixture for more details.

#### Returns:

RandomVariable.


#### Original Docstring for Distribution

Initialize a Mixture distribution.

A `Mixture` is defined by a `Categorical` (`cat`, representing the
mixture probabilities) and a list of `Distribution` objects
all having matching dtype, batch shape, event shape, and continuity
properties (the components).

The `num_classes` of `cat` must be possible to infer at graph construction
time and match `len(components)`.

#### Args:


* <b>`cat`</b>: A `Categorical` distribution instance, representing the probabilities
    of `distributions`.
* <b>`components`</b>: A list or tuple of `Distribution` instances.
  Each instance must have the same type, be defined on the same domain,
  and have matching `event_shape` and `batch_shape`.
* <b>`validate_args`</b>: Python `bool`, default `False`. If `True`, raise a runtime
  error if batch or event ranks are inconsistent between cat and any of
  the distributions. This is only checked if the ranks cannot be
  determined statically at graph construction time.
* <b>`allow_nan_stats`</b>: Boolean, default `True`. If `False`, raise an
 exception if a statistic (e.g. mean/mode/etc...) is undefined for any
  batch member. If `True`, batch members with valid parameters leading to
  undefined statistics will return NaN for this statistic.
* <b>`use_static_graph`</b>: Calls to `sample` will not rely on dynamic tensor
  indexing, allowing for some static graph compilation optimizations, but
  at the expense of sampling all underlying distributions in the mixture.
  (Possibly useful when running on TPUs).
  Default value: `False` (i.e., use dynamic indexing).
* <b>`name`</b>: A name for this distribution (optional).


#### Raises:


* <b>`TypeError`</b>: If cat is not a `Categorical`, or `components` is not
  a list or tuple, or the elements of `components` are not
  instances of `Distribution`, or do not have matching `dtype`.
* <b>`ValueError`</b>: If `components` is an empty list or tuple, or its
  elements do not have a statically known event rank.
  If `cat.num_classes` cannot be inferred at graph creation time,
  or the constant value of `cat.num_classes` is not equal to
  `len(components)`, or all `components` and `cat` do not have
  matching static batch shapes, or all components do not
  have matching static event shapes.