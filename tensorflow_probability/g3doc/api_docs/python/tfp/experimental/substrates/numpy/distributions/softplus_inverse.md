<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.substrates.numpy.distributions.softplus_inverse" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.substrates.numpy.distributions.softplus_inverse


<table class="tfo-notebook-buttons tfo-api" align="left">
</table>



Computes the inverse softplus, i.e., x = softplus_inverse(softplus(x)). (deprecated)

``` python
tfp.experimental.substrates.numpy.distributions.softplus_inverse(
    *args,
    **kwargs
)
```



<!-- Placeholder for "Used in" -->

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2019-10-01.
Instructions for updating:
This function has moved to <a href="../../../../../tfp/math.md"><code>tfp.math</code></a>.

Mathematically this op is equivalent to:

```none
softplus_inverse = log(exp(x) - 1.)
```

#### Args:


* <b>`x`</b>: `Tensor`. Non-negative (not enforced), floating-point.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

`Tensor`. Has the same type/shape as input `x`.
