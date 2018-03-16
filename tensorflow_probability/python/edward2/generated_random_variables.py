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
"""Random variables generated from non-deprecated TensorFlow distributions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import inspect
import tensorflow as tf

from tensorflow_probability.python.edward2.interceptor import interceptable
from tensorflow_probability.python.edward2.random_variable import RandomVariable
from tensorflow_probability.python.util import docstring as docstring_util

tfd = tf.contrib.distributions

__all__ = [
    "Autoregressive",
    "Bernoulli",
    "Beta",
    "Binomial",
    "Categorical",
    "Cauchy",
    "Chi2",
    "ConditionalTransformedDistribution",
    "Deterministic",
    "Dirichlet",
    "DirichletMultinomial",
    "ExpRelaxedOneHotCategorical",
    "Exponential",
    "Gamma",
    "Geometric",
    "HalfNormal",
    "Independent",
    "InverseGamma",
    "Kumaraswamy",
    "Laplace",
    "Logistic",
    "Mixture",
    "MixtureSameFamily",
    "Multinomial",
    "MultivariateNormalDiag",
    "MultivariateNormalFullCovariance",
    "MultivariateNormalTriL",
    "NegativeBinomial",
    "Normal",
    "OneHotCategorical",
    "Poisson",
    "PoissonLogNormalQuadratureCompound",
    "QuantizedDistribution",
    "RelaxedBernoulli",
    "RelaxedOneHotCategorical",
    "SinhArcsinh",
    "StudentT",
    "TransformedDistribution",
    "Uniform",
    "VectorDeterministic",
    "VectorDiffeomixture",
    "VectorExponentialDiag",
    "VectorLaplaceDiag",
    "VectorSinhArcsinhDiag",
    "WishartCholesky",
    "WishartFull",
]


def _make_random_variable(distribution_cls):
  """Factory function to make random variable given distribution class."""
  @interceptable
  @functools.wraps(distribution_cls, assigned=("__module__", "__name__"))
  @docstring_util.expand_docstring(
      cls=distribution_cls.__name__,
      doc=inspect.cleandoc(distribution_cls.__init__.__doc__))
  def func(*args, **kwargs):
    # pylint: disable=g-doc-args
    """Create a random variable for @{cls}.

    See @{cls} for more details.

    Returns:
      RandomVariable.

    #### Original Docstring for Distribution

    @{doc}
    """
    # pylint: enable=g-doc-args
    sample_shape = kwargs.pop("sample_shape", ())
    value = kwargs.pop("value", None)
    return RandomVariable(distribution=distribution_cls(*args, **kwargs),
                          sample_shape=sample_shape,
                          value=value)
  return func


# pylint: disable=invalid-name
Autoregressive = _make_random_variable(tfd.Autoregressive)
Bernoulli = _make_random_variable(tfd.Bernoulli)
Beta = _make_random_variable(tfd.Beta)
Binomial = _make_random_variable(tfd.Binomial)
Categorical = _make_random_variable(tfd.Categorical)
Cauchy = _make_random_variable(tfd.Cauchy)
Chi2 = _make_random_variable(tfd.Chi2)
ConditionalTransformedDistribution = _make_random_variable(
    tfd.ConditionalTransformedDistribution)
Deterministic = _make_random_variable(tfd.Deterministic)
Dirichlet = _make_random_variable(tfd.Dirichlet)
DirichletMultinomial = _make_random_variable(tfd.DirichletMultinomial)
ExpRelaxedOneHotCategorical = _make_random_variable(
    tfd.ExpRelaxedOneHotCategorical)
Exponential = _make_random_variable(tfd.Exponential)
Gamma = _make_random_variable(tfd.Gamma)
Geometric = _make_random_variable(tfd.Geometric)
HalfNormal = _make_random_variable(tfd.HalfNormal)
Independent = _make_random_variable(tfd.Independent)
InverseGamma = _make_random_variable(tfd.InverseGamma)
Kumaraswamy = _make_random_variable(tfd.Kumaraswamy)
Laplace = _make_random_variable(tfd.Laplace)
Logistic = _make_random_variable(tfd.Logistic)
Mixture = _make_random_variable(tfd.Mixture)
MixtureSameFamily = _make_random_variable(tfd.MixtureSameFamily)
Multinomial = _make_random_variable(tfd.Multinomial)
MultivariateNormalDiag = _make_random_variable(tfd.MultivariateNormalDiag)
MultivariateNormalFullCovariance = _make_random_variable(
    tfd.MultivariateNormalFullCovariance)
MultivariateNormalTriL = _make_random_variable(tfd.MultivariateNormalTriL)
NegativeBinomial = _make_random_variable(tfd.NegativeBinomial)
Normal = _make_random_variable(tfd.Normal)
OneHotCategorical = _make_random_variable(tfd.OneHotCategorical)
Poisson = _make_random_variable(tfd.Poisson)
PoissonLogNormalQuadratureCompound = _make_random_variable(
    tfd.PoissonLogNormalQuadratureCompound)
QuantizedDistribution = _make_random_variable(tfd.QuantizedDistribution)
RelaxedBernoulli = _make_random_variable(tfd.RelaxedBernoulli)
RelaxedOneHotCategorical = _make_random_variable(tfd.RelaxedOneHotCategorical)
SinhArcsinh = _make_random_variable(tfd.SinhArcsinh)
StudentT = _make_random_variable(tfd.StudentT)
TransformedDistribution = _make_random_variable(tfd.TransformedDistribution)
Uniform = _make_random_variable(tfd.Uniform)
VectorDeterministic = _make_random_variable(tfd.VectorDeterministic)
VectorDiffeomixture = _make_random_variable(tfd.VectorDiffeomixture)
VectorExponentialDiag = _make_random_variable(tfd.VectorExponentialDiag)
VectorLaplaceDiag = _make_random_variable(tfd.VectorLaplaceDiag)
VectorSinhArcsinhDiag = _make_random_variable(tfd.VectorSinhArcsinhDiag)
WishartCholesky = _make_random_variable(tfd.WishartCholesky)
WishartFull = _make_random_variable(tfd.WishartFull)
# pylint: enable=invalid-name
