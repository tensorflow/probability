<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.distributions.kl_divergence" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.distributions.kl_divergence

Get the KL-divergence KL(distribution_a || distribution_b).

### Aliases:

* `tfp.distributions.kl_divergence`
* `tfp.layers.dense_variational_v2.kullback_leibler.kl_divergence`

``` python
tfp.distributions.kl_divergence(
    distribution_a,
    distribution_b,
    allow_nan_stats=True,
    name=None
)
```



Defined in [`python/distributions/kullback_leibler.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/distributions/kullback_leibler.py).

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

  NotImplementedError: If no KL method is defined for distribution types
    of `distribution_a` and `distribution_b`.

Built-in KL(distribution_a || distribution_b) registrations:

```text
                           distribution_a || distribution_b
======================================================================================
                                Bernoulli || Bernoulli
                                     Beta || Beta
                                Blockwise || Blockwise
                              Categorical || Categorical
                                      Chi || Chi
                                Dirichlet || Dirichlet
                                    Gamma || Gamma
                                   Chi2 +    + Chi2
                          Chi2WithAbsDf +    + Chi2WithAbsDf
                            Exponential +    + Exponential
                          GaussianProcess || MultivariateNormalLinearOperator
         GaussianProcessRegressionModel +    + MultivariateNormalDiag
                                             + MultivariateNormalDiagPlusLowRank
                                             + MultivariateNormalDiagWithSoftplusScale
                                             + MultivariateNormalFullCovariance
                                             + MultivariateNormalTriL
                                             + VariationalGaussianProcess
                          GaussianProcess || Normal
         GaussianProcessRegressionModel +    
                                   Gumbel || Gumbel
                               HalfNormal || HalfNormal
                              Independent || Independent
              JointDistributionSequential || JointDistributionSequential
                 JointDistributionNamed +    + JointDistributionNamed
                                  Laplace || Laplace
         MultivariateNormalLinearOperator || GaussianProcess
                 MultivariateNormalDiag +    + GaussianProcessRegressionModel
      MultivariateNormalDiagPlusLowRank +    
MultivariateNormalDiagWithSoftplusScale +    
       MultivariateNormalFullCovariance +    
                 MultivariateNormalTriL +    
             VariationalGaussianProcess +    
         MultivariateNormalLinearOperator || MultivariateNormalLinearOperator
                 MultivariateNormalDiag +    + MultivariateNormalDiag
      MultivariateNormalDiagPlusLowRank +    + MultivariateNormalDiagPlusLowRank
MultivariateNormalDiagWithSoftplusScale +    + MultivariateNormalDiagWithSoftplusScale
       MultivariateNormalFullCovariance +    + MultivariateNormalFullCovariance
                 MultivariateNormalTriL +    + MultivariateNormalTriL
             VariationalGaussianProcess +    + VariationalGaussianProcess
                                   Normal || GaussianProcess
                                             + GaussianProcessRegressionModel
                                   Normal || Normal
                        OneHotCategorical || OneHotCategorical
                                   Pareto || Pareto
                                   Sample || Sample
                                  Uniform || Uniform
                                 VonMises || VonMises
                       _BaseDeterministic || Distribution
                          Deterministic +    + Autoregressive
                    VectorDeterministic +    + BatchReshape
                                             + Bernoulli
                                             + Beta
                                             + Binomial
                                             + Blockwise
                                             + 74 more
```