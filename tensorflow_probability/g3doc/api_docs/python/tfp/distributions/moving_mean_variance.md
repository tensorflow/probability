<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.distributions.moving_mean_variance" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.distributions.moving_mean_variance

``` python
tfp.distributions.moving_mean_variance(
    value,
    decay,
    collections=None,
    name=None
)
```

Compute exponentially weighted moving {mean,variance} of a streaming value.

The exponentially-weighting moving `mean_var` and `variance_var` are updated
by `value` according to the following recurrence:

```python
variance_var = decay * (variance_var + (1-decay) * (value - mean_var)**2)
mean_var     = decay * mean_var + (1 - decay) * value
```

Note: `mean_var` is updated *after* `variance_var`, i.e., `variance_var` uses
the lag-`1` mean.

For derivation justification, see [Finch (2009; Eq. 143)][1].

Unlike `assign_moving_mean_variance`, this function handles
variable creation.

#### Args:

* <b>`value`</b>: `float`-like `Tensor`. Same shape as `mean_var` and `variance_var`.
* <b>`decay`</b>: A `float`-like `Tensor`. The moving mean decay. Typically close to
    `1.`, e.g., `0.999`.
* <b>`collections`</b>: Python list of graph-collections keys to which the internal
    variables `mean_var` and `variance_var` are added.
    Default value is `[GraphKeys.GLOBAL_VARIABLES]`.
* <b>`name`</b>: Optional name of the returned operation.


#### Returns:

* <b>`mean_var`</b>: `Variable` representing the `value`-updated exponentially weighted
    moving mean.
* <b>`variance_var`</b>: `Variable` representing the `value`-updated
    exponentially weighted moving variance.


#### Raises:

* <b>`TypeError`</b>: if `value_var` does not have float type `dtype`.
* <b>`TypeError`</b>: if `value`, `decay` have different `base_dtype`.

#### References

[1]: Tony Finch. Incremental calculation of weighted mean and variance.
     _Technical Report_, 2009.
     http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf