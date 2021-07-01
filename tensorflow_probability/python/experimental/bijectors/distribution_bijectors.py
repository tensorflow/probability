# Copyright 2020 The TensorFlow Probability Authors.
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
"""Utilities for constructing bijectors from distributions."""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function


import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.distributions import deterministic
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import joint_distribution
from tensorflow_probability.python.distributions import markov_chain
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import sample
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.distributions import uniform

from tensorflow_probability.python.experimental.bijectors import scalar_function_with_inferred_inverse
from tensorflow_probability.python.internal import callable_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers


# pylint: disable=g-long-lambda,protected-access
preconditioning_bijector_fns = {
    deterministic.Deterministic: (
        lambda d: d.experimental_default_event_space_bijector()),
    independent.Independent: lambda d: make_distribution_bijector(
        d.distribution),
    markov_chain.MarkovChain: lambda d: markov_chain._MarkovChainBijector(
        chain=d,
        transition_bijector=make_distribution_bijector(
            d.transition_fn(
                0, d.initial_state_prior.sample(seed=samplers.zeros_seed()))),
        bijector_fn=make_distribution_bijector),
    normal.Normal: lambda d: tfb.Shift(d.loc)(tfb.Scale(d.scale)),
    sample.Sample: lambda d: sample._DefaultSampleBijector(
        distribution=d.distribution,
        sample_shape=d.sample_shape,
        sum_fn=d._sum_fn(),
        bijector=make_distribution_bijector(d.distribution)),
    uniform.Uniform: lambda d: (
        tfb.Shift(d.low)(tfb.Scale(d.high - d.low)(tfb.NormalCDF())))
}
# pylint: enable=g-long-lambda,protected-access


def make_distribution_bijector(distribution, name='make_distribution_bijector'):
  """Builds a bijector to approximately transform `N(0, 1)` into `distribution`.

  This represents a distribution as a bijector that
  transforms a (multivariate) standard normal distribution into the distribution
  of interest.

  Args:
    distribution: A `tfd.Distribution` instance; this may be a joint
      distribution.
    name: Python `str` name for ops created by this method.
  Returns:
    distribution_bijector: a `tfb.Bijector` instance such that
      `distribution_bijector(tfd.Normal(0., 1.))` is approximately equivalent
      to `distribution`.

  #### Examples

  This method may be used to convert structured variational distributions into
  MCMC preconditioners. Consider a model containing
  [funnel geometry](https://crackedbassoon.com/writing/funneling), which may
  be difficult for an MCMC algorithm to sample directly.

  ```python
  model_with_funnel = tfd.JointDistributionSequentialAutoBatched([
      tfd.Normal(loc=-1., scale=2., name='z'),
      lambda z: tfd.Normal(loc=[0., 0., 0.], scale=tf.exp(z), name='x'),
      lambda x: tfd.Poisson(log_rate=x, name='y')])
  pinned_model = tfp.experimental.distributions.JointDistributionPinned(
      model_with_funnel, y=[1, 3, 0])
  ```

  We can approximate the posterior in this model using a structured variational
  surrogate distribution, which will capture the funnel geometry, but cannot
  exactly represent the (non-Gaussian) posterior.

  ```python
  # Build and fit a structured surrogate posterior distribution.
  surrogate_posterior = tfp.experimental.vi.build_asvi_surrogate_posterior(
    pinned_model)
  _ = tfp.vi.fit_surrogate_posterior(pinned_model.unnormalized_log_prob,
                                     surrogate_posterior=surrogate_posterior,
                                     optimizer=tf.optimizers.Adam(0.01),
                                     num_steps=200)
  ```

  Creating a preconditioning bijector allows us to obtain higher-quality
  posterior samples, without any Gaussianity assumption, by using the surrogate
  to guide an MCMC sampler.

  ```python
  surrogate_posterior_bijector = (
    tfp.experimental.bijectors.make_distribution_bijector(surrogate_posterior))
  samples, _ = tfp.mcmc.sample_chain(
    kernel=tfp.mcmc.DualAveragingStepSizeAdaptation(
      tfp.mcmc.TransformedTransitionKernel(
        tfp.mcmc.NoUTurnSampler(pinned_model.unnormalized_log_prob,
                                step_size=0.1),
        bijector=surrogate_posterior_bijector),
      num_adaptation_steps=80),
    current_state=surrogate_posterior.sample(),
    num_burnin_steps=100,
    trace_fn=lambda _0, _1: [],
    num_results=500)
  ```

  #### Mathematical details

  The bijectors returned by this method generally follow the following
  principles, although the specific bijectors returned may vary without notice.

  Normal distributions are reparameterized by a location-scale transform.

  ```python
  b = tfp.experimental.bijectors.make_distribution_bijector(
    tfd.Normal(loc=10., scale=5.))
  # ==> tfb.Shift(10.)(tfb.Scale(5.)))

  b = tfp.experimental.bijectors.make_distribution_bijector(
    tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril))
  # ==> tfb.Shift(loc)(tfb.ScaleMatvecTriL(scale_tril))
  ```

  The distribution's `quantile` function is used, when available:

  ```python
  d = tfd.Cauchy(loc=loc, scale=scale)
  b = tfp.experimental.bijectors.make_distribution_bijector(d)
  # ==> tfb.Inline(forward_fn=d.quantile, inverse_fn=d.cdf)(tfb.NormalCDF())
  ```

  Otherwise, a quantile function is derived by inverting the CDF:

  ```python
  d = tfd.Gamma(concentration=alpha, rate=beta)
  b = tfp.experimental.bijectors.make_distribution_bijector(d)
  # ==> tfb.Invert(
  #  tfp.experimental.bijectors.ScalarFunctionWithInferredInverse(fn=d.cdf))(
  #    tfb.NormalCDF())
  ```

  Transformed distributions are represented by chaining the
  transforming bijector with a preconditioning bijector for the base
  distribution:

  ```python
  b = tfp.experimental.bijectors.make_distribution_bijector(
    tfb.Exp(tfd.Normal(loc=10., scale=5.)))
  # ==> tfb.Exp(tfb.Shift(10.)(tfb.Scale(5.)))
  ```

  Joint distributions are represented by a joint bijector, which converts each
  component distribution to a bijector with parameters conditioned
  on the previous variables in the model. The joint bijector's inputs and
  outputs follow the structure of the joint distribution.

  ```python
  jd = tfd.JointDistributionNamed(
      {'a': tfd.InverseGamma(concentration=2., scale=1.),
       'b': lambda a: tfd.Normal(loc=3., scale=tf.sqrt(a))})
  b = tfp.experimental.bijectors.make_distribution_bijector(jd)
  whitened_jd = tfb.Invert(b)(jd)
  x = whitened_jd.sample()
  # x <=> {'a': tfd.Normal(0., 1.).sample(), 'b': tfd.Normal(0., 1.).sample()}
  ```

  """

  with tf.name_scope(name):
    event_space_bijector = (
        distribution.experimental_default_event_space_bijector())
    if event_space_bijector is None:  # Fail if the distribution is discrete.
      raise NotImplementedError(
          'Cannot transform distribution {} to a standard normal '
          'distribution.'.format(distribution))

    # Recurse over joint distributions.
    if isinstance(distribution, joint_distribution.JointDistribution):
      return joint_distribution._DefaultJointBijector(  # pylint: disable=protected-access
          distribution, bijector_fn=make_distribution_bijector)

    # Recurse through transformed distributions.
    if isinstance(distribution,
                  transformed_distribution.TransformedDistribution):
      return distribution.bijector(
          make_distribution_bijector(distribution.distribution))

    # If we've annotated a specific bijector for this distribution, use that.
    if isinstance(distribution, tuple(preconditioning_bijector_fns)):
      return preconditioning_bijector_fns[type(distribution)](distribution)

    # Otherwise, if this distribution implements a CDF and inverse CDF, build
    # a bijector from those.
    implements_cdf = False
    implements_quantile = False
    input_spec = tf.zeros(shape=distribution.event_shape,
                          dtype=distribution.dtype)
    try:
      callable_util.get_output_spec(distribution.cdf, input_spec)
      implements_cdf = True
    except NotImplementedError:
      pass
    try:
      callable_util.get_output_spec(distribution.quantile, input_spec)
      implements_quantile = True
    except NotImplementedError:
      pass
    if implements_cdf and implements_quantile:
      # This path will only trigger for scalar distributions, since multivariate
      # distributions have non-invertible CDF and so cannot define a `quantile`.
      return tfb.Inline(forward_fn=distribution.quantile,
                        inverse_fn=distribution.cdf,
                        forward_min_event_ndims=ps.rank_from_shape(
                            distribution.event_shape_tensor,
                            distribution.event_shape))(tfb.NormalCDF())

    # If the events are scalar, try to invert the CDF numerically.
    if implements_cdf and tf.get_static_value(distribution.is_scalar_event()):
      return tfb.Invert(
          scalar_function_with_inferred_inverse
          .ScalarFunctionWithInferredInverse(
              distribution.cdf,
              domain_constraint_fn=(event_space_bijector)))(tfb.NormalCDF())

    raise NotImplementedError('Could not automatically construct a '
                              'bijector for distribution type '
                              '{}; it does not implement an invertible '
                              'CDF.'.format(distribution))
