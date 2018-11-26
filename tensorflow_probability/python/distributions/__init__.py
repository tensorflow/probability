# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Statistical distributions and bijective transformations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,line-too-long,g-importing-member

from tensorflow_probability.python.distributions.autoregressive import Autoregressive
from tensorflow_probability.python.distributions.batch_reshape import BatchReshape
from tensorflow_probability.python.distributions.bernoulli import Bernoulli
from tensorflow_probability.python.distributions.beta import Beta
from tensorflow_probability.python.distributions.binomial import Binomial
from tensorflow_probability.python.distributions.categorical import Categorical
from tensorflow_probability.python.distributions.cauchy import Cauchy
from tensorflow_probability.python.distributions.chi2 import Chi2
from tensorflow_probability.python.distributions.chi2 import Chi2WithAbsDf
from tensorflow_probability.python.distributions.conditional_distribution import ConditionalDistribution
from tensorflow_probability.python.distributions.conditional_transformed_distribution import ConditionalTransformedDistribution
from tensorflow_probability.python.distributions.deterministic import Deterministic
from tensorflow_probability.python.distributions.deterministic import VectorDeterministic
from tensorflow_probability.python.distributions.dirichlet import Dirichlet
from tensorflow_probability.python.distributions.dirichlet_multinomial import DirichletMultinomial
from tensorflow_probability.python.distributions.distribution import Distribution
from tensorflow_probability.python.distributions.exponential import Exponential
from tensorflow_probability.python.distributions.gamma import Gamma
from tensorflow_probability.python.distributions.gamma_gamma import GammaGamma
from tensorflow_probability.python.distributions.gaussian_process import GaussianProcess
from tensorflow_probability.python.distributions.gaussian_process_regression_model import GaussianProcessRegressionModel
from tensorflow_probability.python.distributions.geometric import Geometric
from tensorflow_probability.python.distributions.gumbel import Gumbel
from tensorflow_probability.python.distributions.half_cauchy import HalfCauchy
from tensorflow_probability.python.distributions.half_normal import HalfNormal
from tensorflow_probability.python.distributions.hidden_markov_model import HiddenMarkovModel
from tensorflow_probability.python.distributions.independent import Independent
from tensorflow_probability.python.distributions.internal.moving_stats import assign_log_moving_mean_exp
from tensorflow_probability.python.distributions.internal.moving_stats import assign_moving_mean_variance
from tensorflow_probability.python.distributions.internal.moving_stats import moving_mean_variance
from tensorflow_probability.python.distributions.inverse_gamma import InverseGamma
from tensorflow_probability.python.distributions.inverse_gamma import InverseGammaWithSoftplusConcentrationRate
from tensorflow_probability.python.distributions.inverse_gaussian import InverseGaussian
from tensorflow_probability.python.distributions.kullback_leibler import kl_divergence
from tensorflow_probability.python.distributions.kullback_leibler import RegisterKL
from tensorflow_probability.python.distributions.kumaraswamy import Kumaraswamy
from tensorflow_probability.python.distributions.laplace import Laplace
from tensorflow_probability.python.distributions.linear_gaussian_ssm import LinearGaussianStateSpaceModel
from tensorflow_probability.python.distributions.lkj import LKJ
from tensorflow_probability.python.distributions.logistic import Logistic
from tensorflow_probability.python.distributions.lognormal import LogNormal
from tensorflow_probability.python.distributions.mixture import Mixture
from tensorflow_probability.python.distributions.mixture_same_family import MixtureSameFamily
from tensorflow_probability.python.distributions.multinomial import Multinomial
from tensorflow_probability.python.distributions.multivariate_student_t import MultivariateStudentTLinearOperator
from tensorflow_probability.python.distributions.mvn_diag import MultivariateNormalDiag
from tensorflow_probability.python.distributions.mvn_diag import MultivariateNormalDiagWithSoftplusScale
from tensorflow_probability.python.distributions.mvn_diag_plus_low_rank import MultivariateNormalDiagPlusLowRank
from tensorflow_probability.python.distributions.mvn_full_covariance import MultivariateNormalFullCovariance
from tensorflow_probability.python.distributions.mvn_linear_operator import MultivariateNormalLinearOperator
from tensorflow_probability.python.distributions.mvn_tril import MultivariateNormalTriL
from tensorflow_probability.python.distributions.negative_binomial import NegativeBinomial
from tensorflow_probability.python.distributions.normal import Normal
from tensorflow_probability.python.distributions.normal_conjugate_posteriors import normal_conjugates_known_scale_posterior
from tensorflow_probability.python.distributions.normal_conjugate_posteriors import normal_conjugates_known_scale_predictive
from tensorflow_probability.python.distributions.onehot_categorical import OneHotCategorical
from tensorflow_probability.python.distributions.pareto import Pareto
from tensorflow_probability.python.distributions.poisson import Poisson
from tensorflow_probability.python.distributions.poisson_lognormal import PoissonLogNormalQuadratureCompound
from tensorflow_probability.python.distributions.poisson_lognormal import quadrature_scheme_lognormal_gauss_hermite
from tensorflow_probability.python.distributions.poisson_lognormal import quadrature_scheme_lognormal_quantiles
from tensorflow_probability.python.distributions.quantized_distribution import QuantizedDistribution
from tensorflow_probability.python.distributions.relaxed_bernoulli import RelaxedBernoulli
from tensorflow_probability.python.distributions.relaxed_onehot_categorical import ExpRelaxedOneHotCategorical
from tensorflow_probability.python.distributions.relaxed_onehot_categorical import RelaxedOneHotCategorical
from tensorflow_probability.python.distributions.sample_stats import auto_correlation
from tensorflow_probability.python.distributions.sample_stats import percentile
from tensorflow_probability.python.distributions.seed_stream import SeedStream
from tensorflow_probability.python.distributions.sinh_arcsinh import SinhArcsinh
from tensorflow_probability.python.distributions.student_t import StudentT
from tensorflow_probability.python.distributions.student_t_process import StudentTProcess
from tensorflow_probability.python.distributions.transformed_distribution import TransformedDistribution
from tensorflow_probability.python.distributions.triangular import Triangular
from tensorflow_probability.python.distributions.truncated_normal import TruncatedNormal
from tensorflow_probability.python.distributions.uniform import Uniform
from tensorflow_probability.python.distributions.vector_diffeomixture import quadrature_scheme_softmaxnormal_gauss_hermite
from tensorflow_probability.python.distributions.vector_diffeomixture import quadrature_scheme_softmaxnormal_quantiles
from tensorflow_probability.python.distributions.vector_diffeomixture import VectorDiffeomixture
from tensorflow_probability.python.distributions.vector_exponential_diag import VectorExponentialDiag
from tensorflow_probability.python.distributions.vector_laplace_diag import VectorLaplaceDiag
from tensorflow_probability.python.distributions.vector_sinh_arcsinh_diag import VectorSinhArcsinhDiag
from tensorflow_probability.python.distributions.von_mises import VonMises
from tensorflow_probability.python.distributions.von_mises_fisher import VonMisesFisher
from tensorflow_probability.python.distributions.wishart import Wishart
from tensorflow_probability.python.distributions.zipf import Zipf

from tensorflow_probability.python.internal.distribution_util import fill_triangular
from tensorflow_probability.python.internal.distribution_util import fill_triangular_inverse
from tensorflow_probability.python.internal.distribution_util import matrix_diag_transform
from tensorflow_probability.python.internal.distribution_util import reduce_weighted_logsumexp
from tensorflow_probability.python.internal.distribution_util import softplus_inverse
from tensorflow_probability.python.internal.distribution_util import tridiag
from tensorflow_probability.python.internal.reparameterization import FULLY_REPARAMETERIZED
from tensorflow_probability.python.internal.reparameterization import NOT_REPARAMETERIZED
from tensorflow_probability.python.internal.reparameterization import ReparameterizationType

from tensorflow.python.ops.distributions.beta import BetaWithSoftplusConcentration
from tensorflow.python.ops.distributions.exponential import ExponentialWithSoftplusRate
from tensorflow.python.ops.distributions.gamma import GammaWithSoftplusConcentrationRate
from tensorflow.python.ops.distributions.laplace import LaplaceWithSoftplusScale
from tensorflow.python.ops.distributions.normal import NormalWithSoftplusScale
from tensorflow.python.ops.distributions.student_t import StudentTWithAbsDfSoftplusScale

# pylint: enable=unused-import,wildcard-import,line-too-long,g-importing-member

__all__ = [
    'auto_correlation',
    'Cauchy',
    'ConditionalDistribution',
    'ConditionalTransformedDistribution',
    'FULLY_REPARAMETERIZED',
    'NOT_REPARAMETERIZED',
    'ReparameterizationType',
    'Distribution',
    'Autoregressive',
    'BatchReshape',
    'Bernoulli',
    'Beta',
    'Binomial',
    'BetaWithSoftplusConcentration',
    'Categorical',
    'Chi2',
    'Chi2WithAbsDf',
    'Deterministic',
    'VectorDeterministic',
    'Exponential',
    'ExponentialWithSoftplusRate',
    'VectorExponentialDiag',
    'Gamma',
    'GammaGamma',
    'InverseGaussian',
    'GammaWithSoftplusConcentrationRate',
    'Geometric',
    'GaussianProcess',
    'GaussianProcessRegressionModel',
    'Gumbel',
    'HalfCauchy',
    'HalfNormal',
    'HiddenMarkovModel',
    'Independent',
    'InverseGamma',
    'InverseGammaWithSoftplusConcentrationRate',
    'Kumaraswamy',
    'LinearGaussianStateSpaceModel',
    'Laplace',
    'LaplaceWithSoftplusScale',
    'LKJ',
    'Logistic',
    'LogNormal',
    'NegativeBinomial',
    'Normal',
    'NormalWithSoftplusScale',
    'Poisson',
    'PoissonLogNormalQuadratureCompound',
    'SeedStream',
    'SinhArcsinh',
    'StudentT',
    'StudentTWithAbsDfSoftplusScale',
    'StudentTProcess',
    'Triangular',
    'TruncatedNormal',
    'Uniform',
    'MultivariateNormalDiag',
    'MultivariateNormalFullCovariance',
    'MultivariateNormalLinearOperator',
    'MultivariateNormalTriL',
    'MultivariateNormalDiagPlusLowRank',
    'MultivariateNormalDiagWithSoftplusScale',
    'MultivariateStudentTLinearOperator',
    'Dirichlet',
    'DirichletMultinomial',
    'Multinomial',
    'VectorDiffeomixture',
    'VectorLaplaceDiag',
    'VectorSinhArcsinhDiag',
    'VonMises',
    'VonMisesFisher',
    'Wishart',
    'TransformedDistribution',
    'QuantizedDistribution',
    'Mixture',
    'MixtureSameFamily',
    'ExpRelaxedOneHotCategorical',
    'OneHotCategorical',
    'Pareto',
    'RelaxedBernoulli',
    'RelaxedOneHotCategorical',
    'Zipf',
    'kl_divergence',
    'RegisterKL',
    'fill_triangular',
    'fill_triangular_inverse',
    'matrix_diag_transform',
    'reduce_weighted_logsumexp',
    'softplus_inverse',
    'tridiag',
    'normal_conjugates_known_scale_posterior',
    'normal_conjugates_known_scale_predictive',
    'percentile',
    'assign_moving_mean_variance',
    'assign_log_moving_mean_exp',
    'moving_mean_variance',
    'quadrature_scheme_softmaxnormal_gauss_hermite',
    'quadrature_scheme_softmaxnormal_quantiles',
    'quadrature_scheme_lognormal_gauss_hermite',
    'quadrature_scheme_lognormal_quantiles',
]
