<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.distributions.assign_moving_mean_variance" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.distributions.assign_moving_mean_variance


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/distributions/internal/moving_stats.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Compute exponentially weighted moving {mean,variance} of a streaming value.

``` python
tfp.distributions.assign_moving_mean_variance(
    mean_var,
    variance_var,
    value,
    decay,
    name=None
)
```



<!-- Placeholder for "Used in" -->

The `value` updated exponentially weighted moving `mean_var` and
`variance_var` are given by the following recurrence relations:

```python
variance_var = decay * (variance_var + (1 - decay) * (value - mean_var)**2)
mean_var     = decay * mean_var + (1 - decay) * value
```

Note: `mean_var` is updated *after* `variance_var`, i.e., `variance_var` uses
the lag-1 mean.

For derivation justification, see [Finch (2009; Eq. 143)][1].
Parameterization: Finch's `alpha` is `1 - decay`.

#### Args:


* <b>`mean_var`</b>: `float`-like `Variable` representing the exponentially weighted
  moving mean. Same shape as `variance_var` and `value`.
* <b>`variance_var`</b>: `float`-like `Variable` representing the
  exponentially weighted moving variance. Same shape as `mean_var` and
  `value`.
* <b>`value`</b>: `float`-like `Tensor`. Same shape as `mean_var` and `variance_var`.
* <b>`decay`</b>: A `float`-like `Tensor`. The moving mean decay. Typically close to
  `1.`, e.g., `0.999`.
* <b>`name`</b>: Optional name of the returned operation.


#### Returns:


* <b>`mean_var`</b>: `Variable` representing the `value`-updated exponentially weighted
  moving mean.
* <b>`variance_var`</b>: `Variable` representing the `value`-updated
  exponentially weighted moving variance.


#### Raises:


* <b>`TypeError`</b>: if `mean_var` does not have float type `dtype`.
* <b>`TypeError`</b>: if `mean_var`, `variance_var`, `value`, `decay` have different
  `base_dtype`.

#### References

[1]: Tony Finch. Incremental calculation of weighted mean and variance.
     _Technical Report_, 2009.
     http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf