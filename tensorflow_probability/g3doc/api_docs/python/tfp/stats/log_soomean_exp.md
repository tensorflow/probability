<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.stats.log_soomean_exp" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.stats.log_soomean_exp


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/stats/leave_one_out.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Computes the log-swap-one-out-mean of `exp(logx)`.

``` python
tfp.stats.log_soomean_exp(
    logx,
    axis,
    keepdims=False,
    name=None
)
```



<!-- Placeholder for "Used in" -->

The swapped out element `logx[i]` is replaced with the log-leave-`i`-out
geometric mean of `logx`.

#### Args:


* <b>`logx`</b>: Floating-type `Tensor` representing `log(x)` where `x` is some
  positive value.
* <b>`axis`</b>: The dimensions to sum across. If `None` (the default), reduces all
  dimensions. Must be in the range `[-rank(logx), rank(logx)]`.
  Default value: `None` (i.e., reduce over all dims).
* <b>`keepdims`</b>: If true, retains reduced dimensions with length 1.
  Default value: `False` (i.e., keep all dims in `log_mean_x`).
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
  Default value: `None` (i.e., `"log_soomean_exp"`).


#### Returns:


* <b>`log_soomean_x`</b>: ``Tensor` with same shape and dtype as `logx` representing
  the natural-log of the average of `x`` except that the element `logx[i]`
  is replaced with the log of the leave-`i`-out Geometric-average. The mean
  of the gradient of `log_soomean_x` is `1`. Mathematically `log_soomean_x`
  is,
  ```none
  log_soomean_x[i] = log(Avg{h[j ; i] : j=0, ..., m-1})
  h[j ; i] = { u[j]                              j!=i
             { GeometricAverage{u[k] : k != i}   j==i
  ```
* <b>`log_mean_x`</b>: `logx.dtype` `Tensor` corresponding to the natural-log of the
  average of `x`. The sum of the gradient of `log_mean_x` is `1`. Has
  reduced shape of `logx` (per `axis` and `keepdims`).