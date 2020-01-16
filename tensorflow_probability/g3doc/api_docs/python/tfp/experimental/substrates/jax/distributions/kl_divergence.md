<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.substrates.jax.distributions.kl_divergence" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.substrates.jax.distributions.kl_divergence


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/substrates/jax/distributions/kullback_leibler.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Get the KL-divergence KL(distribution_a || distribution_b).

``` python
tfp.experimental.substrates.jax.distributions.kl_divergence(
    distribution_a,
    distribution_b,
    allow_nan_stats=True,
    name=None
)
```



<!-- Placeholder for "Used in" -->

If there is no KL method registered specifically for `type(distribution_a)`
and `type(distribution_b)`, then the class hierarchies of these types are
searched.

If one KL method is registered between any pairs of classes in these two
parent hierarchies, it is used.

If more than one such registered method exists, the method whose registered
classes have the shortest sum MRO paths to the input types is used.

If more than one such shortest path exists, the first method
identified in the search is used (favoring a shorter MRO distance to
`type(distribution_a)`).

#### Args:


* <b>`distribution_a`</b>: The first distribution.
* <b>`distribution_b`</b>: The second distribution.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`,
  statistics (e.g., mean, mode, variance) use the value "`NaN`" to
  indicate the result is undefined. When `False`, an exception is raised
  if one or more of the statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.


#### Returns:

A Tensor with the batchwise KL-divergence between `distribution_a`
and `distribution_b`.



#### Raises:


* <b>`NotImplementedError`</b>: If no KL method is defined for distribution types
  of `distribution_a` and `distribution_b`.

Built-in KL(distribution_a || distribution_b) registrations:

```text
                     distribution_a || distribution_b
==========================================================================
                          Bernoulli || Bernoulli
                               Beta || Beta
                          Blockwise || Blockwise
                        Categorical || Categorical
                                Chi || Chi
                               Chi2 || Chi2
                               Chi2 || Gamma
                                       + Exponential
                          Dirichlet || Dirichlet
                              Gamma || Chi2
                      Exponential +    
                              Gamma || Gamma
                      Exponential +    + Exponential
                             Gumbel || Gumbel
                         HalfNormal || HalfNormal
                        Independent || Independent
        JointDistributionSequential || JointDistributionSequential
                            Laplace || Laplace
   MultivariateNormalLinearOperator || MultivariateNormalLinearOperator
           MultivariateNormalDiag +    + MultivariateNormalDiag
MultivariateNormalDiagPlusLowRank +    + MultivariateNormalDiagPlusLowRank
 MultivariateNormalFullCovariance +    + MultivariateNormalFullCovariance
           MultivariateNormalTriL +    + MultivariateNormalTriL
                             Normal || Normal
                  OneHotCategorical || OneHotCategorical
                             Pareto || Pareto
                    ProbitBernoulli || ProbitBernoulli
                             Sample || Sample
                            Uniform || Uniform
                 _BaseDeterministic || Distribution
                    Deterministic +    + Autoregressive
              VectorDeterministic +    + BatchReshape
                                       + Bernoulli
                                       + Beta
                                       + Binomial
                                       + Blockwise
                                       + 66 more
                              _Cast || _Cast
```