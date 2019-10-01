<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.stats.log_loosum_exp" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.stats.log_loosum_exp


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/stats/leave_one_out.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Computes the log-leave-one-out-sum of `exp(logx)`.

``` python
tfp.stats.log_loosum_exp(
    logx,
    axis=None,
    keepdims=False,
    name=None
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`logx`</b>: Floating-type `Tensor` representing `log(x)` where `x` is some
  positive value.
* <b>`axis`</b>: The dimensions to sum across. If `None` (the default), reduces all
  dimensions. Must be in the range `[-rank(logx), rank(logx)]`.
  Default value: `None` (i.e., reduce over all dims).
* <b>`keepdims`</b>: If true, retains reduced dimensions with length 1.
  Default value: `False` (i.e., keep all dims in `log_sum_x`).
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
  Default value: `None` (i.e., `"log_loosum_exp"`).


#### Returns:


* <b>`log_loosum_exp`</b>: `Tensor` with same shape and dtype as `logx` representing
  the natural-log of the sum of `exp(logx)` except that the element
  `logx[i]` is removed.
* <b>`log_sum_x`</b>: `logx.dtype` `Tensor` corresponding to the natural-log of the
  sum of `exp(logx)`. Has reduced shape of `logx` (per `axis` and
  `keepdims`).