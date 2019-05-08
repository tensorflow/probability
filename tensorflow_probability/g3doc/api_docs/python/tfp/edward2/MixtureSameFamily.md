<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.MixtureSameFamily" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.MixtureSameFamily

Create a random variable for MixtureSameFamily.

``` python
tfp.edward2.MixtureSameFamily(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See MixtureSameFamily for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Construct a `MixtureSameFamily` distribution.


#### Args:

* <b>`mixture_distribution`</b>: <a href="../../tfp/distributions/Categorical.md"><code>tfp.distributions.Categorical</code></a>-like instance.
  Manages the probability of selecting components. The number of
  categories must match the rightmost batch dimension of the
  `components_distribution`. Must have either scalar `batch_shape` or
  `batch_shape` matching `components_distribution.batch_shape[:-1]`.
* <b>`components_distribution`</b>: <a href="../../tfp/distributions/Distribution.md"><code>tfp.distributions.Distribution</code></a>-like instance.
  Right-most batch dimension indexes components.
* <b>`reparameterize`</b>: Python `bool`, default `False`. Whether to reparameterize
  samples of the distribution using implicit reparameterization gradients
  [(Figurnov et al., 2018)][1]. The gradients for the mixture logits are
  equivalent to the ones described by [(Graves, 2016)][2]. The gradients
  for the components parameters are also computed using implicit
  reparameterization (as opposed to ancestral sampling), meaning that
  all components are updated every step.
  Only works when:
    (1) components_distribution is fully reparameterized;
    (2) components_distribution is either a scalar distribution or
    fully factorized (tfd.Independent applied to a scalar distribution);
    (3) batch shape has a known rank.
  Experimental, may be slow and produce infs/NaNs.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
  (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
  result is undefined. When `False`, an exception is raised if one or
  more of the statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.


#### Raises:

  ValueError: `if not dtype_util.is_integer(mixture_distribution.dtype)`.
  ValueError: if mixture_distribution does not have scalar `event_shape`.
  ValueError: if `mixture_distribution.batch_shape` and
    `components_distribution.batch_shape[:-1]` are both fully defined and
    the former is neither scalar nor equal to the latter.
  ValueError: if `mixture_distribution` categories does not equal
    `components_distribution` rightmost batch shape.

#### References

[1]: Michael Figurnov, Shakir Mohamed and Andriy Mnih. Implicit
     reparameterization gradients. In _Neural Information Processing
     Systems_, 2018. https://arxiv.org/abs/1805.08498

[2]: Alex Graves. Stochastic Backpropagation through Mixture Density
     Distributions. _arXiv_, 2016. https://arxiv.org/abs/1607.05690