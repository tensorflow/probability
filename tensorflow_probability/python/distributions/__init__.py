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
from tensorflow_probability.python.distributions.bates import Bates
from tensorflow_probability.python.distributions.bernoulli import Bernoulli
from tensorflow_probability.python.distributions.beta import Beta
from tensorflow_probability.python.distributions.beta_binomial import BetaBinomial
from tensorflow_probability.python.distributions.binomial import Binomial
from tensorflow_probability.python.distributions.blockwise import Blockwise
from tensorflow_probability.python.distributions.categorical import Categorical
from tensorflow_probability.python.distributions.cauchy import Cauchy
from tensorflow_probability.python.distributions.chi import Chi
from tensorflow_probability.python.distributions.chi2 import Chi2
from tensorflow_probability.python.distributions.cholesky_lkj import CholeskyLKJ
from tensorflow_probability.python.distributions.continuous_bernoulli import ContinuousBernoulli
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
from tensorflow_probability.python.distributions.generalized_normal import GeneralizedNormal
from tensorflow_probability.python.distributions.generalized_pareto import GeneralizedPareto
from tensorflow_probability.python.distributions.geometric import Geometric
from tensorflow_probability.python.distributions.gumbel import Gumbel
from tensorflow_probability.python.distributions.half_cauchy import HalfCauchy
from tensorflow_probability.python.distributions.half_normal import HalfNormal
from tensorflow_probability.python.distributions.half_student_t import HalfStudentT
from tensorflow_probability.python.distributions.hidden_markov_model import HiddenMarkovModel
from tensorflow_probability.python.distributions.horseshoe import Horseshoe
from tensorflow_probability.python.distributions.independent import Independent
from tensorflow_probability.python.distributions.inverse_gamma import InverseGamma
from tensorflow_probability.python.distributions.inverse_gaussian import InverseGaussian
from tensorflow_probability.python.distributions.johnson_su import JohnsonSU
from tensorflow_probability.python.distributions.joint_distribution import JointDistribution
from tensorflow_probability.python.distributions.joint_distribution_coroutine import JointDistributionCoroutine
from tensorflow_probability.python.distributions.joint_distribution_auto_batched import JointDistributionCoroutineAutoBatched
from tensorflow_probability.python.distributions.joint_distribution_named import JointDistributionNamed
from tensorflow_probability.python.distributions.joint_distribution_auto_batched import JointDistributionNamedAutoBatched
from tensorflow_probability.python.distributions.joint_distribution_sequential import JointDistributionSequential
from tensorflow_probability.python.distributions.joint_distribution_auto_batched import JointDistributionSequentialAutoBatched
from tensorflow_probability.python.distributions.kumaraswamy import Kumaraswamy
from tensorflow_probability.python.distributions.laplace import Laplace
from tensorflow_probability.python.distributions.linear_gaussian_ssm import LinearGaussianStateSpaceModel
from tensorflow_probability.python.distributions.lkj import LKJ
from tensorflow_probability.python.distributions.logistic import Logistic
from tensorflow_probability.python.distributions.loglogistic import LogLogistic
from tensorflow_probability.python.distributions.lognormal import LogNormal
from tensorflow_probability.python.distributions.logitnormal import LogitNormal
from tensorflow_probability.python.distributions.mixture import Mixture
from tensorflow_probability.python.distributions.mixture_same_family import MixtureSameFamily
from tensorflow_probability.python.distributions.moyal import Moyal
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
from tensorflow_probability.python.distributions.ordered_logistic import OrderedLogistic
from tensorflow_probability.python.distributions.pareto import Pareto
from tensorflow_probability.python.distributions.pert import PERT
from tensorflow_probability.python.distributions.pixel_cnn import PixelCNN
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
from tensorflow_probability.python.distributions.vector_exponential_diag import VectorExponentialDiag
from tensorflow_probability.python.distributions.von_mises import VonMises
from tensorflow_probability.python.distributions.von_mises_fisher import VonMisesFisher
from tensorflow_probability.python.distributions.wishart import WishartLinearOperator
from tensorflow_probability.python.distributions.wishart import WishartTriL
from tensorflow_probability.python.distributions.zipf import Zipf

# Utilities/Other:
from tensorflow_probability.python.distributions.joint_distribution_util import independent_joint_distribution_from_structure
from tensorflow_probability.python.distributions.kullback_leibler import RegisterKL
from tensorflow_probability.python.distributions.kullback_leibler import kl_divergence
from tensorflow_probability.python.distributions.normal_conjugate_posteriors import mvn_conjugate_linear_update
from tensorflow_probability.python.distributions.normal_conjugate_posteriors import normal_conjugates_known_scale_posterior
from tensorflow_probability.python.distributions.normal_conjugate_posteriors import normal_conjugates_known_scale_predictive
from tensorflow_probability.python.distributions.poisson_lognormal import quadrature_scheme_lognormal_gauss_hermite
from tensorflow_probability.python.distributions.poisson_lognormal import quadrature_scheme_lognormal_quantiles
from tensorflow_probability.python.internal.reparameterization import FULLY_REPARAMETERIZED
from tensorflow_probability.python.internal.reparameterization import NOT_REPARAMETERIZED
from tensorflow_probability.python.internal.reparameterization import ReparameterizationType

# Module management:
from tensorflow_probability.python.distributions.kullback_leibler import augment_kl_xent_docs
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import


import sys as _sys  # pylint: disable=g-import-not-at-top
augment_kl_xent_docs(_sys.modules[__name__])
del augment_kl_xent_docs
del _sys

# pylint: enable=unused-import,wildcard-import,g-importing-member,g-bad-import-order
# pylint: enable=line-too-long

__all__ = [
    'FULLY_REPARAMETERIZED',
    'NOT_REPARAMETERIZED',
    'ReparameterizationType',
    'Distribution',
    'Autoregressive',
    'BatchReshape',
    'Bates',
    'Bernoulli',
    'Beta',
    'BetaBinomial',
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
    'GeneralizedNormal',
    'GeneralizedPareto',
    'Geometric',
    'GaussianProcess',
    'GaussianProcessRegressionModel',
    'VariationalGaussianProcess',
    'Gumbel',
    'HalfCauchy',
    'HalfNormal',
    'HalfStudentT',
    'HiddenMarkovModel',
    'Horseshoe',
    'Independent',
    'InverseGamma',
    'JohnsonSU',
    'JointDistribution',
    'JointDistributionCoroutine',
    'JointDistributionCoroutineAutoBatched',
    'JointDistributionNamed',
    'JointDistributionNamedAutoBatched',
    'JointDistributionSequential',
    'JointDistributionSequentialAutoBatched',
    'Kumaraswamy',
    'LinearGaussianStateSpaceModel',
    'Laplace',
    'LKJ',
    'Logistic',
    'LogLogistic',
    'LogNormal',
    'LogitNormal',
    'Moyal',
    'NegativeBinomial',
    'Normal',
    'PixelCNN',
    'Poisson',
    'PoissonLogNormalQuadratureCompound',
    'ProbitBernoulli',
    'Sample',
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
    'VonMises',
    'VonMisesFisher',
    'WishartLinearOperator',
    'WishartTriL',
    'TransformedDistribution',
    'QuantizedDistribution',
    'Mixture',
    'MixtureSameFamily',
    'ExpRelaxedOneHotCategorical',
    'OneHotCategorical',
    'OrderedLogistic',
    'Pareto',
    'PERT',
    'PlackettLuce',
    'RelaxedBernoulli',
    'RelaxedOneHotCategorical',
    'Zipf',
    'kl_divergence',
    'RegisterKL',
    'independent_joint_distribution_from_structure',
    'mvn_conjugate_linear_update',
    'normal_conjugates_known_scale_posterior',
    'normal_conjugates_known_scale_predictive',
    'quadrature_scheme_lognormal_gauss_hermite',
    'quadrature_scheme_lognormal_quantiles',
]
