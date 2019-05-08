<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.distributions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="FULLY_REPARAMETERIZED"/>
<meta itemprop="property" content="NOT_REPARAMETERIZED"/>
</div>

# Module: tfp.distributions

Statistical distributions.



Defined in [`python/distributions/__init__.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/distributions/__init__.py).

<!-- Placeholder for "Used in" -->


## Classes

[`class Cauchy`](../tfp/distributions/Cauchy.md): The Cauchy distribution with location `loc` and scale `scale`.

[`class ConditionalDistribution`](../tfp/distributions/ConditionalDistribution.md): Distribution that supports intrinsic parameters (local latents).

[`class ConditionalTransformedDistribution`](../tfp/distributions/ConditionalTransformedDistribution.md): A TransformedDistribution that allows intrinsic conditioning.

[`class ReparameterizationType`](../tfp/distributions/ReparameterizationType.md): Instances of this class represent how sampling is reparameterized.

[`class Distribution`](../tfp/distributions/Distribution.md): A generic probability distribution base class.

[`class Autoregressive`](../tfp/distributions/Autoregressive.md): Autoregressive distributions.

[`class BatchReshape`](../tfp/distributions/BatchReshape.md): The Batch-Reshaping distribution.

[`class Bernoulli`](../tfp/distributions/Bernoulli.md): Bernoulli distribution.

[`class Beta`](../tfp/distributions/Beta.md): Beta distribution.

[`class Binomial`](../tfp/distributions/Binomial.md): Binomial distribution.

[`class Blockwise`](../tfp/distributions/Blockwise.md): Blockwise distribution.

[`class Categorical`](../tfp/distributions/Categorical.md): Categorical distribution over integers.

[`class Chi`](../tfp/distributions/Chi.md): Chi distribution.

[`class Chi2`](../tfp/distributions/Chi2.md): Chi2 distribution.

[`class Chi2WithAbsDf`](../tfp/distributions/Chi2WithAbsDf.md): Chi2 with parameter transform `df = floor(abs(df))`.

[`class Deterministic`](../tfp/distributions/Deterministic.md): Scalar `Deterministic` distribution on the real line.

[`class VectorDeterministic`](../tfp/distributions/VectorDeterministic.md): Vector `Deterministic` distribution on `R^k`.

[`class Empirical`](../tfp/distributions/Empirical.md): Empirical distribution.

[`class Exponential`](../tfp/distributions/Exponential.md): Exponential distribution.

[`class VectorExponentialDiag`](../tfp/distributions/VectorExponentialDiag.md): The vectorization of the Exponential distribution on `R^k`.

[`class Gamma`](../tfp/distributions/Gamma.md): Gamma distribution.

[`class GammaGamma`](../tfp/distributions/GammaGamma.md): Gamma-Gamma distribution.

[`class InverseGaussian`](../tfp/distributions/InverseGaussian.md): Inverse Gaussian distribution.

[`class Geometric`](../tfp/distributions/Geometric.md): Geometric distribution.

[`class GaussianProcess`](../tfp/distributions/GaussianProcess.md): Marginal distribution of a Gaussian process at finitely many points.

[`class GaussianProcessRegressionModel`](../tfp/distributions/GaussianProcessRegressionModel.md): Posterior predictive distribution in a conjugate GP regression model.

[`class VariationalGaussianProcess`](../tfp/distributions/VariationalGaussianProcess.md): Posterior predictive of a variational Gaussian process.

[`class Gumbel`](../tfp/distributions/Gumbel.md): The scalar Gumbel distribution with location `loc` and `scale` parameters.

[`class HalfCauchy`](../tfp/distributions/HalfCauchy.md): Half-Cauchy distribution.

[`class HalfNormal`](../tfp/distributions/HalfNormal.md): The Half Normal distribution with scale `scale`.

[`class HiddenMarkovModel`](../tfp/distributions/HiddenMarkovModel.md): Hidden Markov model distribution.

[`class Horseshoe`](../tfp/distributions/Horseshoe.md): Horseshoe distribution.

[`class Independent`](../tfp/distributions/Independent.md): Independent distribution from batch of distributions.

[`class InverseGamma`](../tfp/distributions/InverseGamma.md): InverseGamma distribution.

[`class JointDistribution`](../tfp/distributions/JointDistribution.md): Joint distribution over one or more component distributions.

[`class JointDistributionCoroutine`](../tfp/distributions/JointDistributionCoroutine.md): Joint distribution parameterized by a distribution-making generator.

[`class JointDistributionNamed`](../tfp/distributions/JointDistributionNamed.md): Joint distribution parameterized by named distribution-making functions.

[`class JointDistributionSequential`](../tfp/distributions/JointDistributionSequential.md): Joint distribution parameterized by distribution-making functions.

[`class Kumaraswamy`](../tfp/distributions/Kumaraswamy.md): Kumaraswamy distribution.

[`class LinearGaussianStateSpaceModel`](../tfp/distributions/LinearGaussianStateSpaceModel.md): Observation distribution from a linear Gaussian state space model.

[`class Laplace`](../tfp/distributions/Laplace.md): The Laplace distribution with location `loc` and `scale` parameters.

[`class LKJ`](../tfp/distributions/LKJ.md): The LKJ distribution on correlation matrices.

[`class Logistic`](../tfp/distributions/Logistic.md): The Logistic distribution with location `loc` and `scale` parameters.

[`class LogNormal`](../tfp/distributions/LogNormal.md): The log-normal distribution.

[`class NegativeBinomial`](../tfp/distributions/NegativeBinomial.md): NegativeBinomial distribution.

[`class Normal`](../tfp/distributions/Normal.md): The Normal distribution with location `loc` and `scale` parameters.

[`class Poisson`](../tfp/distributions/Poisson.md): Poisson distribution.

[`class PoissonLogNormalQuadratureCompound`](../tfp/distributions/PoissonLogNormalQuadratureCompound.md): `PoissonLogNormalQuadratureCompound` distribution.

[`class Sample`](../tfp/distributions/Sample.md): Sample distribution via independent draws.

[`class SeedStream`](../tfp/distributions/SeedStream.md): Local PRNG for amplifying seed entropy into seeds for base operations.

[`class SinhArcsinh`](../tfp/distributions/SinhArcsinh.md): The SinhArcsinh transformation of a distribution on `(-inf, inf)`.

[`class StudentT`](../tfp/distributions/StudentT.md): Student's t-distribution.

[`class StudentTProcess`](../tfp/distributions/StudentTProcess.md): Marginal distribution of a Student's T process at finitely many points.

[`class Triangular`](../tfp/distributions/Triangular.md): Triangular distribution with `low`, `high` and `peak` parameters.

[`class TruncatedNormal`](../tfp/distributions/TruncatedNormal.md): The Truncated Normal distribution.

[`class Uniform`](../tfp/distributions/Uniform.md): Uniform distribution with `low` and `high` parameters.

[`class MultivariateNormalDiag`](../tfp/distributions/MultivariateNormalDiag.md): The multivariate normal distribution on `R^k`.

[`class MultivariateNormalFullCovariance`](../tfp/distributions/MultivariateNormalFullCovariance.md): The multivariate normal distribution on `R^k`.

[`class MultivariateNormalLinearOperator`](../tfp/distributions/MultivariateNormalLinearOperator.md): The multivariate normal distribution on `R^k`.

[`class MultivariateNormalTriL`](../tfp/distributions/MultivariateNormalTriL.md): The multivariate normal distribution on `R^k`.

[`class MultivariateNormalDiagPlusLowRank`](../tfp/distributions/MultivariateNormalDiagPlusLowRank.md): The multivariate normal distribution on `R^k`.

[`class MultivariateNormalDiagWithSoftplusScale`](../tfp/distributions/MultivariateNormalDiagWithSoftplusScale.md): MultivariateNormalDiag with `diag_stddev = softplus(diag_stddev)`.

[`class MultivariateStudentTLinearOperator`](../tfp/distributions/MultivariateStudentTLinearOperator.md): The [Multivariate Student's t-distribution](

[`class Dirichlet`](../tfp/distributions/Dirichlet.md): Dirichlet distribution.

[`class DirichletMultinomial`](../tfp/distributions/DirichletMultinomial.md): Dirichlet-Multinomial compound distribution.

[`class Multinomial`](../tfp/distributions/Multinomial.md): Multinomial distribution.

[`class VectorDiffeomixture`](../tfp/distributions/VectorDiffeomixture.md): VectorDiffeomixture distribution.

[`class VectorLaplaceDiag`](../tfp/distributions/VectorLaplaceDiag.md): The vectorization of the Laplace distribution on `R^k`.

[`class VectorSinhArcsinhDiag`](../tfp/distributions/VectorSinhArcsinhDiag.md): The (diagonal) SinhArcsinh transformation of a distribution on `R^k`.

[`class VonMises`](../tfp/distributions/VonMises.md): The von Mises distribution over angles.

[`class VonMisesFisher`](../tfp/distributions/VonMisesFisher.md): The von Mises-Fisher distribution over unit vectors on `S^{n-1}`.

[`class Wishart`](../tfp/distributions/Wishart.md): The matrix Wishart distribution on positive definite matrices.

[`class TransformedDistribution`](../tfp/distributions/TransformedDistribution.md): A Transformed Distribution.

[`class QuantizedDistribution`](../tfp/distributions/QuantizedDistribution.md): Distribution representing the quantization `Y = ceiling(X)`.

[`class Mixture`](../tfp/distributions/Mixture.md): Mixture distribution.

[`class MixtureSameFamily`](../tfp/distributions/MixtureSameFamily.md): Mixture (same-family) distribution.

[`class ExpRelaxedOneHotCategorical`](../tfp/distributions/ExpRelaxedOneHotCategorical.md): ExpRelaxedOneHotCategorical distribution with temperature and logits.

[`class OneHotCategorical`](../tfp/distributions/OneHotCategorical.md): OneHotCategorical distribution.

[`class Pareto`](../tfp/distributions/Pareto.md): Pareto distribution.

[`class RelaxedBernoulli`](../tfp/distributions/RelaxedBernoulli.md): RelaxedBernoulli distribution with temperature and logits parameters.

[`class RelaxedOneHotCategorical`](../tfp/distributions/RelaxedOneHotCategorical.md): RelaxedOneHotCategorical distribution with temperature and logits.

[`class Zipf`](../tfp/distributions/Zipf.md): Zipf distribution.

[`class RegisterKL`](../tfp/distributions/RegisterKL.md): Decorator to register a KL divergence implementation function.

## Functions

[`InverseGammaWithSoftplusConcentrationRate(...)`](../tfp/distributions/InverseGammaWithSoftplusConcentrationRate.md): `InverseGamma` with softplus of `concentration` and `scale`. (deprecated)

[`InverseGammaWithSoftplusConcentrationScale(...)`](../tfp/distributions/InverseGammaWithSoftplusConcentrationScale.md): `InverseGamma` with softplus of `concentration` and `scale`. (deprecated)

[`kl_divergence(...)`](../tfp/distributions/kl_divergence.md): Get the KL-divergence KL(distribution_a || distribution_b).

[`fill_triangular(...)`](../tfp/distributions/fill_triangular.md): Creates a (batch of) triangular matrix from a vector of inputs.

[`fill_triangular_inverse(...)`](../tfp/distributions/fill_triangular_inverse.md): Creates a vector from a (batch of) triangular matrix.

[`matrix_diag_transform(...)`](../tfp/distributions/matrix_diag_transform.md): Transform diagonal of [batch-]matrix, leave rest of matrix unchanged.

[`reduce_weighted_logsumexp(...)`](../tfp/distributions/reduce_weighted_logsumexp.md): Computes `log(abs(sum(weight * exp(elements across tensor dimensions))))`.

[`softplus_inverse(...)`](../tfp/distributions/softplus_inverse.md): Computes the inverse softplus, i.e., x = softplus_inverse(softplus(x)).

[`tridiag(...)`](../tfp/distributions/tridiag.md): Creates a matrix with values set above, below, and on the diagonal.

[`normal_conjugates_known_scale_posterior(...)`](../tfp/distributions/normal_conjugates_known_scale_posterior.md): Posterior Normal distribution with conjugate prior on the mean.

[`normal_conjugates_known_scale_predictive(...)`](../tfp/distributions/normal_conjugates_known_scale_predictive.md): Posterior predictive Normal distribution w. conjugate prior on the mean.

[`assign_moving_mean_variance(...)`](../tfp/distributions/assign_moving_mean_variance.md): Compute exponentially weighted moving {mean,variance} of a streaming value.

[`assign_log_moving_mean_exp(...)`](../tfp/distributions/assign_log_moving_mean_exp.md): Compute the log of the exponentially weighted moving mean of the exp.

[`moving_mean_variance(...)`](../tfp/distributions/moving_mean_variance.md): Compute exponentially weighted moving {mean,variance} of a streaming value. (deprecated)

[`quadrature_scheme_softmaxnormal_gauss_hermite(...)`](../tfp/distributions/quadrature_scheme_softmaxnormal_gauss_hermite.md): Use Gauss-Hermite quadrature to form quadrature on `K - 1` simplex.

[`quadrature_scheme_softmaxnormal_quantiles(...)`](../tfp/distributions/quadrature_scheme_softmaxnormal_quantiles.md): Use SoftmaxNormal quantiles to form quadrature on `K - 1` simplex.

[`quadrature_scheme_lognormal_gauss_hermite(...)`](../tfp/distributions/quadrature_scheme_lognormal_gauss_hermite.md): Use Gauss-Hermite quadrature to form quadrature on positive-reals.

[`quadrature_scheme_lognormal_quantiles(...)`](../tfp/distributions/quadrature_scheme_lognormal_quantiles.md): Use LogNormal quantiles to form quadrature on positive-reals.

## Other Members

<h3 id="FULLY_REPARAMETERIZED"><code>FULLY_REPARAMETERIZED</code></h3>

<h3 id="NOT_REPARAMETERIZED"><code>NOT_REPARAMETERIZED</code></h3>

