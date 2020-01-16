<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.RelaxedOneHotCategorical" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.RelaxedOneHotCategorical


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/edward2/interceptor.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Create a random variable for RelaxedOneHotCategorical.

### Aliases:

* `tfp.experimental.edward2.RelaxedOneHotCategorical`


``` python
tfp.edward2.RelaxedOneHotCategorical(
    *args,
    **kwargs
)
```



<!-- Placeholder for "Used in" -->

See RelaxedOneHotCategorical for more details.

#### Returns:

RandomVariable.


#### Original Docstring for Distribution

Initialize RelaxedOneHotCategorical using class log-probabilities.

#### Args:


* <b>`temperature`</b>: An 0-D `Tensor`, representing the temperature
  of a set of RelaxedOneHotCategorical distributions. The temperature
  should be positive.
* <b>`logits`</b>: An N-D `Tensor`, `N >= 1`, representing the log probabilities
  of a set of RelaxedOneHotCategorical distributions. The first
  `N - 1` dimensions index into a batch of independent distributions and
  the last dimension represents a vector of logits for each class. Only
  one of `logits` or `probs` should be passed in.
* <b>`probs`</b>: An N-D `Tensor`, `N >= 1`, representing the probabilities
  of a set of RelaxedOneHotCategorical distributions. The first `N - 1`
  dimensions index into a batch of independent distributions and the last
  dimension represents a vector of probabilities for each class. Only one
  of `logits` or `probs` should be passed in.
* <b>`validate_args`</b>: Unused in this distribution.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. If `False`, raise an
  exception if a statistic (e.g. mean/mode/etc...) is undefined for any
  batch member. If `True`, batch members with valid parameters leading to
  undefined statistics will return NaN for this statistic.
* <b>`name`</b>: A name for this distribution (optional).