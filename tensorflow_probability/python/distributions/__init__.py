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
"""Statistical distributions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,line-too-long,g-importing-member,g-bad-import-order

# Distributions:
from tensorflow_probability.python.distributions.autoregressive import Autoregressive
from tensorflow_probability.python.distributions.batch_reshape import BatchReshape
from tensorflow_probability.python.distributions.bernoulli import Bernoulli
from tensorflow_probability.python.distributions.beta import Beta
from tensorflow_probability.python.distributions.binomial import Binomial
from tensorflow_probability.python.distributions.blockwise import Blockwise
from tensorflow_probability.python.distributions.categorical import Categorical
from tensorflow_probability.python.distributions.cauchy import Cauchy
from tensorflow_probability.python.distributions.chi import Chi
from tensorflow_probability.python.distributions.chi2 import Chi2
from tensorflow_probability.python.distributions.cholesky_lkj import CholeskyLKJ
from tensorflow_probability.python.distributions.deterministic import Deterministic
from tensorflow_probability.python.distributions.deterministic import VectorDeterministic
from tensorflow_probability.python.distributions.dirichlet import Dirichlet
from tensorflow_probability.python.distributions.dirichlet_multinomial import DirichletMultinomial
from tensorflow_probability.python.distributions.distribution import Distribution
from tensorflow_probability.python.distributions.doublesided_maxwell import DoublesidedMaxwell
from tensorflow_probability.python.distributions.empirical import Empirical
from tensorflow_probability.python.distributions.exponential import Exponential
from tensorflow_probability.python.distributions.finite_discrete import FiniteDiscrete
from tensorflow_probability.python.distributions.gamma import Gamma
from tensorflow_probability.python.distributions.gamma_gamma import GammaGamma
from tensorflow_probability.python.distributions.gaussian_process import GaussianProcess
from tensorflow_probability.python.distributions.gaussian_process_regression_model import GaussianProcessRegressionModel
from tensorflow_probability.python.distributions.generalized_pareto import GeneralizedPareto
from tensorflow_probability.python.distributions.geometric import Geometric
from tensorflow_probability.python.distributions.gumbel import Gumbel
from tensorflow_probability.python.distributions.half_cauchy import HalfCauchy
from tensorflow_probability.python.distributions.half_normal import HalfNormal
from tensorflow_probability.python.distributions.hidden_markov_model import HiddenMarkovModel
from tensorflow_probability.python.distributions.horseshoe import Horseshoe
from tensorflow_probability.python.distributions.independent import Independent
from tensorflow_probability.python.distributions.inverse_gamma import InverseGamma
from tensorflow_probability.python.distributions.inverse_gaussian import InverseGaussian
from tensorflow_probability.python.distributions.joint_distribution import JointDistribution
from tensorflow_probability.python.distributions.joint_distribution_coroutine import JointDistributionCoroutine
from tensorflow_probability.python.distributions.joint_distribution_named import JointDistributionNamed
from tensorflow_probability.python.distributions.joint_distribution_sequential import JointDistributionSequential
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
from tensorflow_probability.python.distributions.mvn_diag_plus_low_rank import MultivariateNormalDiagPlusLowRank
from tensorflow_probability.python.distributions.mvn_full_covariance import MultivariateNormalFullCovariance
from tensorflow_probability.python.distributions.mvn_linear_operator import MultivariateNormalLinearOperator
from tensorflow_probability.python.distributions.mvn_tril import MultivariateNormalTriL
from tensorflow_probability.python.distributions.negative_binomial import NegativeBinomial
from tensorflow_probability.python.distributions.normal import Normal
from tensorflow_probability.python.distributions.onehot_categorical import OneHotCategorical
from tensorflow_probability.python.distributions.pareto import Pareto
from tensorflow_probability.python.distributions.pert import PERT
from tensorflow_probability.python.distributions.plackett_luce import PlackettLuce
from tensorflow_probability.python.distributions.poisson import Poisson
from tensorflow_probability.python.distributions.poisson_lognormal import PoissonLogNormalQuadratureCompound
from tensorflow_probability.python.distributions.probit_bernoulli import ProbitBernoulli
from tensorflow_probability.python.distributions.quantized_distribution import QuantizedDistribution
from tensorflow_probability.python.distributions.relaxed_bernoulli import RelaxedBernoulli
from tensorflow_probability.python.distributions.relaxed_onehot_categorical import ExpRelaxedOneHotCategorical
from tensorflow_probability.python.distributions.relaxed_onehot_categorical import RelaxedOneHotCategorical
from tensorflow_probability.python.distributions.sample import Sample
from tensorflow_probability.python.distributions.sinh_arcsinh import SinhArcsinh
from tensorflow_probability.python.distributions.student_t import StudentT
from tensorflow_probability.python.distributions.student_t_process import StudentTProcess
from tensorflow_probability.python.distributions.transformed_distribution import TransformedDistribution
from tensorflow_probability.python.distributions.triangular import Triangular
from tensorflow_probability.python.distributions.truncated_normal import TruncatedNormal
from tensorflow_probability.python.distributions.uniform import Uniform
from tensorflow_probability.python.distributions.variational_gaussian_process import VariationalGaussianProcess
from tensorflow_probability.python.distributions.vector_diffeomixture import VectorDiffeomixture
from tensorflow_probability.python.distributions.vector_exponential_diag import VectorExponentialDiag
from tensorflow_probability.python.distributions.vector_laplace_diag import VectorLaplaceDiag
from tensorflow_probability.python.distributions.vector_sinh_arcsinh_diag import VectorSinhArcsinhDiag
from tensorflow_probability.python.distributions.von_mises import VonMises
from tensorflow_probability.python.distributions.von_mises_fisher import VonMisesFisher
from tensorflow_probability.python.distributions.wishart import Wishart
from tensorflow_probability.python.distributions.zipf import Zipf

# Utilities/Other:
from tensorflow_probability.python.distributions.deprecated_linalg import matrix_diag_transform
from tensorflow_probability.python.distributions.deprecated_linalg import tridiag
from tensorflow_probability.python.distributions.internal.moving_stats import assign_log_moving_mean_exp
from tensorflow_probability.python.distributions.internal.moving_stats import assign_moving_mean_variance
from tensorflow_probability.python.distributions.internal.moving_stats import moving_mean_variance
from tensorflow_probability.python.distributions.kullback_leibler import RegisterKL
from tensorflow_probability.python.distributions.kullback_leibler import kl_divergence
from tensorflow_probability.python.distributions.normal_conjugate_posteriors import normal_conjugates_known_scale_posterior
from tensorflow_probability.python.distributions.normal_conjugate_posteriors import normal_conjugates_known_scale_predictive
from tensorflow_probability.python.distributions.poisson_lognormal import quadrature_scheme_lognormal_gauss_hermite
from tensorflow_probability.python.distributions.poisson_lognormal import quadrature_scheme_lognormal_quantiles
from tensorflow_probability.python.distributions.vector_diffeomixture import quadrature_scheme_softmaxnormal_gauss_hermite
from tensorflow_probability.python.distributions.vector_diffeomixture import quadrature_scheme_softmaxnormal_quantiles
from tensorflow_probability.python.internal.reparameterization import FULLY_REPARAMETERIZED
from tensorflow_probability.python.internal.reparameterization import NOT_REPARAMETERIZED
from tensorflow_probability.python.internal.reparameterization import ReparameterizationType

# Deprecated:
from tensorflow_probability.python.math.generic import reduce_weighted_logsumexp as _reduce_weighted_logsumexp
from tensorflow_probability.python.math.generic import softplus_inverse as _softplus_inverse
from tensorflow_probability.python.math.linalg import fill_triangular as _fill_triangular
from tensorflow_probability.python.math.linalg import fill_triangular_inverse as _fill_triangular_inverse
from tensorflow_probability.python.util.seed_stream import SeedStream as _SeedStream

# Module management:
from tensorflow_probability.python.distributions.kullback_leibler import augment_kl_xent_docs
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import


_deprecated = deprecation.deprecated(
    '2019-10-01',
    'This function has moved to `tfp.math`.')

fill_triangular = _deprecated(_fill_triangular)

fill_triangular_inverse = _deprecated(_fill_triangular_inverse)

softplus_inverse = _deprecated(_softplus_inverse)

reduce_weighted_logsumexp = _deprecated(_reduce_weighted_logsumexp)


class SeedStream(_SeedStream):

  __init__ = deprecation.deprecated(
      '2019-10-01', 'SeedStream has moved to `tfp.util.SeedStream`.')(
          _SeedStream.__init__)


del deprecation, _deprecated


import sys as _sys  # pylint: disable=g-import-not-at-top
augment_kl_xent_docs(_sys.modules[__name__])
del augment_kl_xent_docs
del _sys

# pylint: enable=unused-import,wildcard-import,line-too-long,g-importing-member,g-bad-import-order

__all__ = [
    'FULLY_REPARAMETERIZED',
    'NOT_REPARAMETERIZED',
    'ReparameterizationType',
    'Distribution',
    'Autoregressive',
    'BatchReshape',
    'Bernoulli',
    'Beta',
    'Binomial',
    'Blockwise',
    'Categorical',
    'Cauchy',
    'Chi',
    'Chi2',
    'CholeskyLKJ',
    'Deterministic',
    'DoublesidedMaxwell',
    'VectorDeterministic',
    'Empirical',
    'Exponential',
    'VectorExponentialDiag',
    'Gamma',
    'GammaGamma',
    'InverseGaussian',
    'GeneralizedPareto',
    'Geometric',
    'GaussianProcess',
    'GaussianProcessRegressionModel',
    'VariationalGaussianProcess',
    'Gumbel',
    'HalfCauchy',
    'HalfNormal',
    'HiddenMarkovModel',
    'Horseshoe',
    'Independent',
    'InverseGamma',
    'JointDistribution',
    'JointDistributionCoroutine',
    'JointDistributionNamed',
    'JointDistributionSequential',
    'Kumaraswamy',
    'LinearGaussianStateSpaceModel',
    'Laplace',
    'LKJ',
    'Logistic',
    'LogNormal',
    'NegativeBinomial',
    'Normal',
    'Poisson',
    'PoissonLogNormalQuadratureCompound',
    'ProbitBernoulli',
    'Sample',
    'SeedStream',
    'SinhArcsinh',
    'StudentT',
    'StudentTProcess',
    'Triangular',
    'TruncatedNormal',
    'Uniform',
    'MultivariateNormalDiag',
    'MultivariateNormalFullCovariance',
    'MultivariateNormalLinearOperator',
    'MultivariateNormalTriL',
    'MultivariateNormalDiagPlusLowRank',
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
    'PERT',
    'PlackettLuce',
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
    'assign_moving_mean_variance',
    'assign_log_moving_mean_exp',
    'moving_mean_variance',
    'quadrature_scheme_softmaxnormal_gauss_hermite',
    'quadrature_scheme_softmaxnormal_quantiles',
    'quadrature_scheme_lognormal_gauss_hermite',
    'quadrature_scheme_lognormal_quantiles',
]
