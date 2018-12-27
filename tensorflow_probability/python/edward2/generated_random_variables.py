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

from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.edward2.interceptor import interceptable
from tensorflow_probability.python.edward2.random_variable import RandomVariable

from tensorflow_probability.python.util import docstring as docstring_util


rv_all = [
    'Autoregressive',
    'BatchReshape',
    'Bernoulli',
    'Beta',
    'BetaWithSoftplusConcentration',
    'Binomial',
    'Categorical',
    'Cauchy',
    'Chi2',
    'Chi2WithAbsDf',
    'ConditionalTransformedDistribution',
    'Deterministic',
    'Dirichlet',
    'DirichletMultinomial',
    'ExpRelaxedOneHotCategorical',
    'Exponential',
    'ExponentialWithSoftplusRate',
    'Gamma',
    'GammaGamma',
    'GammaWithSoftplusConcentrationRate',
    'Geometric',
    'GaussianProcess',
    'GaussianProcessRegressionModel',
    'Gumbel',
    'HalfCauchy',
    'HalfNormal',
    'HiddenMarkovModel',
    'Horseshoe',
    'Independent',
    'InverseGamma',
    'InverseGammaWithSoftplusConcentrationRate',
    'InverseGaussian',
    'Kumaraswamy',
    'LinearGaussianStateSpaceModel',
    'Laplace',
    'LaplaceWithSoftplusScale',
    'LKJ',
    'Logistic',
    'LogNormal',
    'Mixture',
    'MixtureSameFamily',
    'Multinomial',
    'MultivariateNormalDiag',
    'MultivariateNormalFullCovariance',
    'MultivariateNormalLinearOperator',
    'MultivariateNormalTriL',
    'MultivariateNormalDiagPlusLowRank',
    'MultivariateNormalDiagWithSoftplusScale',
    'MultivariateStudentTLinearOperator',
    'NegativeBinomial',
    'Normal',
    'NormalWithSoftplusScale',
    'OneHotCategorical',
    'Pareto',
    'Poisson',
    'PoissonLogNormalQuadratureCompound',
    'QuantizedDistribution',
    'RelaxedBernoulli',
    'RelaxedOneHotCategorical',
    'SinhArcsinh',
    'StudentT',
    'StudentTWithAbsDfSoftplusScale',
    'StudentTProcess',
    'TransformedDistribution',
    'Triangular',
    'TruncatedNormal',
    'Uniform',
    'VectorDeterministic',
    'VectorDiffeomixture',
    'VectorExponentialDiag',
    'VectorLaplaceDiag',
    'VectorSinhArcsinhDiag',
    'VonMises',
    'VonMisesFisher',
    'Wishart',
    'Zipf',
]

__all__ = rv_all + [
    'as_random_variable'
]


def _simple_name(distribution):
  """Infer the original name passed into a distribution constructor.

  Distributions typically follow the pattern of
  with.name_scope(name) as name:
    super(name=name)
  so we attempt to reverse the name-scope transformation to allow
  addressing of RVs by the distribution's original, user-visible
  name kwarg.

  Args:
    distribution: a tfd.Distribution instance.
  Returns:
    simple_name: the original name passed into the Distribution.

  #### Example

  ```
  d1 = tfd.Normal(0., 1., name='x') # d1.name = 'x/'
  d2 = tfd.Normal(0., 1., name='x') # d2.name = 'x_2/'
  _simple_name(d2) # returns 'x'

  ```

  """
  simple_name = distribution.name

  # turn 'scope/x/' into 'x'
  if simple_name.endswith('/'):
    simple_name = simple_name.split('/')[-2]

  # turn 'x_3' into 'x'
  parts = simple_name.split('_')
  if parts[-1].isdigit():
    simple_name = '_'.join(parts[:-1])

  return simple_name


@interceptable
def _build_custom_rv(distribution, sample_shape, value, name):
  """RandomVariable constructor with a dummy name argument."""
  # Program transformations (e.g., `make_log_joint_fn`) assume that
  # the traced constructor has `name` and `value` kwargs, enabling
  # them to override the value of an RV according to its name.
  # User-defined RVs inherit their name from the provided
  # distribution; this helper method exposes the name as a dummy kwarg
  # so that it's visible to program transformations.
  del name  # unused
  return RandomVariable(distribution=distribution,
                        sample_shape=sample_shape,
                        value=value)


def as_random_variable(distribution,
                       sample_shape=(),
                       value=None):
  """Wrap an existing distribution as a traceable random variable.

  This enables the use of custom or user-provided distributions in
  Edward models. Unlike a bare `RandomVariable` object, this method
  wraps the constructor so it is included in the Edward trace and its
  values can be properly intercepted and overridden.

  Where possible, you should prefer the built-in constructors
  (`ed.Normal`, etc); these simultaneously construct a Distribution
  and a RandomVariable object so that the distribution parameters
  themselves may be intercepted and overridden. RVs constructed via
  `as_random_variable()` have a fixed distribution and may not support
  program transformations (e.g, conjugate marginalization) that rely
  on overriding distribution parameters.

  Args:
    distribution: tfd.Distribution governing the distribution of the random
      variable, such as sampling and log-probabilities.
    sample_shape: tf.TensorShape of samples to draw from the random variable.
      Default is `()` corresponding to a single sample.
    value: Fixed tf.Tensor to associate with random variable. Must have shape
      `sample_shape + distribution.batch_shape + distribution.event_shape`.
      Default is to sample from random variable according to `sample_shape`.

  Returns:
    rv: a `RandomVariable` wrapping the provided distribution.

  #### Example

  ```python
  from tensorflow_probability import distributions as tfd
  from tensorflow_probability import edward2 as ed

  def model():
    # equivalent to ed.Normal(0., 1., name='x')
    return ed.as_random_variable(tfd.Normal(0., 1., name='x'))

  log_joint = ed.make_log_joint_fn(model)
  output = log_joint(x=2.)
  ```
  """

  return _build_custom_rv(distribution=distribution,
                          sample_shape=sample_shape,
                          value=value,
                          name=_simple_name(distribution))


def _make_random_variable(distribution_cls):
  """Factory function to make random variable given distribution class."""

  @interceptable
  @functools.wraps(distribution_cls, assigned=('__module__', '__name__'))
  @docstring_util.expand_docstring(
      cls=distribution_cls.__name__,
      doc=inspect.cleandoc(distribution_cls.__init__.__doc__ or ''))
  def func(*args, **kwargs):
    # pylint: disable=g-doc-args
    """Create a random variable for ${cls}.

    See ${cls} for more details.

    Returns:
      RandomVariable.

    #### Original Docstring for Distribution

    ${doc}
    """
    # pylint: enable=g-doc-args
    sample_shape = kwargs.pop('sample_shape', ())
    value = kwargs.pop('value', None)
    return RandomVariable(distribution=distribution_cls(*args, **kwargs),
                          sample_shape=sample_shape,
                          value=value)
  return func


# pylint: disable=invalid-name
Autoregressive = _make_random_variable(tfd.Autoregressive)
BatchReshape = _make_random_variable(tfd.BatchReshape)
Bernoulli = _make_random_variable(tfd.Bernoulli)
Beta = _make_random_variable(tfd.Beta)
BetaWithSoftplusConcentration = _make_random_variable(
    tfd.BetaWithSoftplusConcentration)
Binomial = _make_random_variable(tfd.Binomial)
Categorical = _make_random_variable(tfd.Categorical)
Cauchy = _make_random_variable(tfd.Cauchy)
Chi2 = _make_random_variable(tfd.Chi2)
Chi2WithAbsDf = _make_random_variable(tfd.Chi2WithAbsDf)
ConditionalTransformedDistribution = _make_random_variable(
    tfd.ConditionalTransformedDistribution)
Deterministic = _make_random_variable(tfd.Deterministic)
Dirichlet = _make_random_variable(tfd.Dirichlet)
DirichletMultinomial = _make_random_variable(tfd.DirichletMultinomial)
ExpRelaxedOneHotCategorical = _make_random_variable(
    tfd.ExpRelaxedOneHotCategorical)
Exponential = _make_random_variable(tfd.Exponential)
ExponentialWithSoftplusRate = _make_random_variable(
    tfd.ExponentialWithSoftplusRate)
Gamma = _make_random_variable(tfd.Gamma)
GammaGamma = _make_random_variable(tfd.GammaGamma)
GammaWithSoftplusConcentrationRate = _make_random_variable(
    tfd.GammaWithSoftplusConcentrationRate)
Geometric = _make_random_variable(tfd.Geometric)
GaussianProcess = _make_random_variable(tfd.GaussianProcess)
GaussianProcessRegressionModel = _make_random_variable(
    tfd.GaussianProcessRegressionModel)
Gumbel = _make_random_variable(tfd.Gumbel)
HalfCauchy = _make_random_variable(tfd.HalfCauchy)
HalfNormal = _make_random_variable(tfd.HalfNormal)
HiddenMarkovModel = _make_random_variable(tfd.HiddenMarkovModel)
Horseshoe = _make_random_variable(tfd.Horseshoe)
Independent = _make_random_variable(tfd.Independent)
InverseGamma = _make_random_variable(tfd.InverseGamma)
InverseGammaWithSoftplusConcentrationRate = _make_random_variable(
    tfd.InverseGammaWithSoftplusConcentrationRate)
InverseGaussian = _make_random_variable(tfd.InverseGaussian)
Kumaraswamy = _make_random_variable(tfd.Kumaraswamy)
LinearGaussianStateSpaceModel = _make_random_variable(
    tfd.LinearGaussianStateSpaceModel)
Laplace = _make_random_variable(tfd.Laplace)
LaplaceWithSoftplusScale = _make_random_variable(tfd.LaplaceWithSoftplusScale)
LKJ = _make_random_variable(tfd.LKJ)
Logistic = _make_random_variable(tfd.Logistic)
LogNormal = _make_random_variable(tfd.LogNormal)
Mixture = _make_random_variable(tfd.Mixture)
MixtureSameFamily = _make_random_variable(tfd.MixtureSameFamily)
Multinomial = _make_random_variable(tfd.Multinomial)
MultivariateNormalDiag = _make_random_variable(tfd.MultivariateNormalDiag)
MultivariateNormalFullCovariance = _make_random_variable(
    tfd.MultivariateNormalFullCovariance)
MultivariateNormalLinearOperator = _make_random_variable(
    tfd.MultivariateNormalLinearOperator)
MultivariateNormalTriL = _make_random_variable(tfd.MultivariateNormalTriL)
MultivariateNormalDiagPlusLowRank = _make_random_variable(
    tfd.MultivariateNormalDiagPlusLowRank)
MultivariateNormalDiagWithSoftplusScale = _make_random_variable(
    tfd.MultivariateNormalDiagWithSoftplusScale)
MultivariateStudentTLinearOperator = _make_random_variable(
    tfd.MultivariateStudentTLinearOperator)
NegativeBinomial = _make_random_variable(tfd.NegativeBinomial)
Normal = _make_random_variable(tfd.Normal)
NormalWithSoftplusScale = _make_random_variable(tfd.NormalWithSoftplusScale)
OneHotCategorical = _make_random_variable(tfd.OneHotCategorical)
Pareto = _make_random_variable(tfd.Pareto)
Poisson = _make_random_variable(tfd.Poisson)
PoissonLogNormalQuadratureCompound = _make_random_variable(
    tfd.PoissonLogNormalQuadratureCompound)
QuantizedDistribution = _make_random_variable(tfd.QuantizedDistribution)
RelaxedBernoulli = _make_random_variable(tfd.RelaxedBernoulli)
RelaxedOneHotCategorical = _make_random_variable(tfd.RelaxedOneHotCategorical)
SinhArcsinh = _make_random_variable(tfd.SinhArcsinh)
StudentT = _make_random_variable(tfd.StudentT)
StudentTWithAbsDfSoftplusScale = _make_random_variable(
    tfd.StudentTWithAbsDfSoftplusScale)
StudentTProcess = _make_random_variable(tfd.StudentTProcess)
TransformedDistribution = _make_random_variable(tfd.TransformedDistribution)
Triangular = _make_random_variable(tfd.Triangular)
TruncatedNormal = _make_random_variable(tfd.TruncatedNormal)
Uniform = _make_random_variable(tfd.Uniform)
VectorDeterministic = _make_random_variable(tfd.VectorDeterministic)
VectorDiffeomixture = _make_random_variable(tfd.VectorDiffeomixture)
VectorExponentialDiag = _make_random_variable(tfd.VectorExponentialDiag)
VectorLaplaceDiag = _make_random_variable(tfd.VectorLaplaceDiag)
VectorSinhArcsinhDiag = _make_random_variable(tfd.VectorSinhArcsinhDiag)
VonMises = _make_random_variable(tfd.VonMises)
VonMisesFisher = _make_random_variable(tfd.VonMisesFisher)
Wishart = _make_random_variable(tfd.Wishart)
Zipf = _make_random_variable(tfd.Zipf)
# pylint: enable=invalid-name
