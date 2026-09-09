#### Copyright 2025 The TensorFlow Authors.

```none
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

# Discrete Distributions

## Discrete Distributions with Finite Support

### Distributions Supported on a Single Point

[Deterministic](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Deterministic),
[VectorDeterministic](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/VectorDeterministic)

### Distributions Supported on Two Points

[Bernoulli](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Bernoulli),
[ProbitBernoulli](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/ProbitBernoulli)

### Distributions Supported on N Points

#### Distributions Supported on {0, 1, .. N}

[OrderedLogistic](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/OrderedLogistic),
[StoppingRatioLogistic](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/StoppingRatioLogistic),
[Categorical](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Categorical),
[BetaBinomial](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/BetaBinomial),
[Binomial](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Binomial)

#### Distributions over [low, .., high]

[QuantizedDistribution](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/QuantizedDistribution)

#### Distributions on Arbitrary N Points

[FiniteDiscrete](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/FiniteDiscrete)

#### Distributions on Count Vectors

[DirichletMultinomial](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/DirichletMultinomial),
[Multinomial](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Multinomial)

#### Distributions on Finite Subsets

[DeterminantalPointProcess](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/DeterminantalPointProcess)

#### Distributions on Images

[PixelCNN](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/PixelCNN)

#### Distributions on One Hot Vectors

[OneHotCategorical](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/OneHotCategorical)

#### Distributions on Permutations

[PlackettLuce](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/PlackettLuce)

## Discrete Distributions with Infinite Support

### Distributions on all Integers

[Skellam](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Skellam)

### Distributions on Positive Integers

[Geometric](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Geometric),
[PoissonLogNormalQuadratureCompound](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/PoissonLogNormalQuadratureCompound)

### Distributions on Non-negative Integers

[NegativeBinomial](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/NegativeBinomial),
[Poisson](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Poisson),
ZeroInflatedNegativeBinomial,
[Zipf](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Zipf)

# Continuous Distributions

## Scalar Continous Distributions

### Distributions over the Line

[Cauchy](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Cauchy),
[DoublesidedMaxwell](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/DoublesidedMaxwell),
[ExpGamma](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/ExpGamma),
[ExpInverseGamma](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/ExpInverseGamma),
[Exponential](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Exponential),
[ExponentiallyModifiedGaussian](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/ExponentiallyModifiedGaussian),
[GeneralizedNormal](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/GeneralizedNormal),
[GeneralizedPareto](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/GeneralizedPareto),
[GeneralizedExtremeValue](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/GeneralizedExtremeValue),
[Gumbel](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Gumbel),
[Horseshoe](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Horseshoe),
[JohnsonSU](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/JohnsonSU),
[LambertWDistribution](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/LambertWDistribution),
[LambertWNormal](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/LambertWNormal),
[Laplace](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Laplace),
[Logistic](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Logistic),
[Moyal](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Moyal),
[Normal](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Normal),
[NormalInverseGaussian](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/NormalInverseGaussian),
[SigmoidBeta](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/SigmoidBeta),
[SinhArcsinh](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/SinhArcsinh),
[StudentT](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/StudentT),
[TwoPieceNormal](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/TwoPieceNormal),
[Weibull](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Weibull)

### Distributions over Half the Line

#### Distributions over Positive Numbers

[BetaQuotient](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/BetaQuotient),
[Chi](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Chi),
[Chi2](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Chi2),
[Gamma](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Gamma),
[GammaGamma](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/GammaGamma),
[HalfNormal](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/HalfNormal),
[InverseGamma](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/InverseGamma),
[InverseGaussian](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/InverseGaussian),
[LogLogistic](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/LogLogistic),
[LogNormal](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/LogNormal)

#### Distributions over [a, infinity)

[HalfCauchy](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/HalfCauchy),
[HalfStudentT](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/HalfStudentT),
[Pareto](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Pareto)

### Distributions over Intervals

#### Distributions over [0, 1]

[Beta](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Beta),
[ContinuousBernoulli](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/ContinuousBernoulli),
[Kumaraswamy](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Kumaraswamy),
[LogitNormal](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/LogitNormal),
[RelaxedBernoulli](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/RelaxedBernoulli)

#### Distributions over Angles

[VonMises](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/VonMises)

#### Distributions over [low, high]

[Bates](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Bates),
[PERT](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/PERT),
[Triangular](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Triangular),
[TruncatedCauchy](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/TruncatedCauchy),
[TruncatedNormal](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/TruncatedNormal),
[Uniform](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Uniform)

## Higher-Dimensional Continuous Distributions

### Distributions over R^n

[MultivariateStudentTLinearOperator](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MultivariateStudentTLinearOperator),
[MultivariateNormalDiag](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MultivariateNormalDiag),
[MultivariateNormalDiagPlusLowRank](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MultivariateNormalDiagPlusLowRank),
[MultivariateNormalDiagPlusLowRankCovariance](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MultivariateNormalDiagPlusLowRankCovariance),
[MultivariateNormalFullCovariance](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MultivariateNormalFullCovariance),
[MultivariateNormalLinearOperator](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MultivariateNormalLinearOperator),
[MultivariateNormalLowRankUpdateLinearOperatorCovariance](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MultivariateNormalLowRankUpdateLinearOperatorCovariance),
[MultivariateNormalTriL](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MultivariateNormalTriL),
[VectorExponentialLinearOperator](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/VectorExponentialLinearOperator),
[Autoregressive](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Autoregressive),
[Empirical](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Empirical),
[ExpRelaxedOneHotCategorical](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/ExpRelaxedOneHotCategorical)

### Distributions over Unit Simplex in R^n

[Dirichlet](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Dirichlet)
[FlatDirichlet](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/FlatDirichlet)
[RelaxedOneHotCategorical](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/RelaxedOneHotCategorical)

### Distributions over Matrices

#### Distributions over Unconstrained Matrices

[MatrixNormalLinearOperator](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MatrixNormalLinearOperator),
[MatrixTLinearOperator](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MatrixTLinearOperator)

#### Distributions over Lower Triangular Matrices

[CholeskyLKJ](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/CholeskyLKJ)

#### Distributions over Positive Semi-definite Matrices

[WishartLinearOperator](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/WishartLinearOperator),
[WishartTriL](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/WishartTriL)

#### Distributions over Positive Definite Correlation Matrices

[LKJ](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/LKJ)

### Distributions over Time Series

[HiddenMarkovModel](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/HiddenMarkovModel),
[LinearGaussianStateSpaceModel](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/LinearGaussianStateSpaceModel),
[MarkovChain](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MarkovChain)

### Distributions on a Sphere

[PowerSpherical](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/PowerSpherical),
[SphericalUniform](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/SphericalUniform),
[VonMisesFisher](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/VonMisesFisher)

### Distributions over Functions

[GaussianProcess](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/GaussianProcess),
[GaussianProcessRegressionModel](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/GaussianProcessRegressionModel),
[StudentTProcess](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/StudentTProcess),
[StudentTProcessRegressionModel](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/StudentTProcessRegressionModel),
[VariationalGaussianProcess](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/VariationalGaussianProcess)

# Wrapper Distributions

## Batching Wrappers

[BatchBroadcast](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/BatchBroadcast),
[BatchConcat](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/BatchConcat),
[BatchReshape](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/BatchReshape)

## Joint Distribution Wrappers

[JointDistributionCoroutineAutoBatched](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/JointDistributionCoroutineAutoBatched),
[JointDistributionNamedAutoBatched](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/JointDistributionNamedAutoBatched),
[JointDistributionSequentialAutoBatched](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/JointDistributionSequentialAutoBatched),
[JointDistributionCoroutine](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/JointDistributionCoroutine),
[JointDistributionNamed](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/JointDistributionNamed),
[JointDistributionSequential](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/JointDistributionSequential)

## Other Distribution Wrappers

[Blockwise](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Blockwise),
[Independent](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Independent),
Inflated,
[Masked](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Masked),
[Mixture](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Mixture),
[MixtureSameFamily](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MixtureSameFamily),
[Sample](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Sample),
[TransformedDistribution](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/TransformedDistribution)
