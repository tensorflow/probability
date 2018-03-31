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
"""Methods for Bayesian inference over model weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
# Dependency imports
import numpy as np

import tensorflow as tf

tfd = tf.contrib.distributions

# key for Collection of KL terms used in the ELBO loss
VI_KL_LOSSES = "vi_kl_losses"
VI_PRIORS = "vi_priors"
VI_QDISTS = "vi_qdists"


def make_posterior_mvndiag(true_getter, base_var, name=None):
  """Builds a diagonal Gaussian posterior for a Variable `base_var`.

  This method is intended to be passed as an argument to
  `build_bayesian_getter`, not called directly.

  By using MVNDiag, we double the number of parameters in the model.
  If this is a problem one can imagine using alternatives like:
  `MVNDiag(scale_identity=s)`.

  Args:
    true_getter: The true getter being wrapped by our custom
      Bayesian getter. This should be used in place of any
      `tf.get_variable()` calls inside this method.
    base_var: The `tf.Variable` for which we are defining a
      variational posterior. This is treated as the mean of
      the Gaussian approximate posterior.
    name: The `Variable`'s name as passed into the Bayesian
      getter (note this will include any prefixes added
      by the calling model's `variable_scope`).

  Returns:
    q: A Distribution object representing the posterior.
  """

  with tf.name_scope(name, "make_posterior_mvndiag", [base_var]):
    # it's generally useful to initialize the weight posterior
    # close to a delta distribution (i.e., small stddev), so we
    # get a large initial gradient on the mean params
    sigma_init = np.log(np.expm1(0.01))  # == softplus_inverse(0.01)

    scale_diag = tf.nn.softplus(true_getter(
        name="{}/make_posterior_mvndiag/softplus_inverse_sigma".format(name),
        shape=base_var.shape.as_list(),
        dtype=base_var.dtype,
        initializer=tf.constant_initializer(sigma_init),
        trainable=True))
    return tfd.Independent(tfd.Normal(
        loc=base_var,
        scale=scale_diag,
        name="{}_posterior".format(name)),
                           reinterpreted_batch_ndims=tf.rank(base_var))


def make_posterior_point(true_getter, base_var, name=None):  # pylint: disable=unused-argument
  """Builds a point-mass approximate 'posterior' for a Variable `base_var`.

  This allows us to construct the model joint density `p(x, z)` as a
  special case of the ELBO, and in particular to perform MAP
  estimation as a degenerate special case of variational inference.

  Args:
    true_getter: The true getter being wrapped by our custom
      Bayesian getter. This should be used in place of any
      `tf.get_variable()` calls inside this method.
    base_var: The `tf.Variable` for which we are defining a
      variational posterior. This is treated as the mean of
      the Gaussian approximate posterior.
    name: The `Variable`'s name as passed into the Bayesian
      getter (note this will include any prefixes added
      by the calling model's `variable_scope`).

  Returns:
    q: A Distribution object representing the posterior.
  """
  with tf.name_scope(name, "make_posterior_point", [base_var]):
    return tfd.Independent(tfd.Deterministic(
        loc=base_var,
        name="{}_posterior".format(name)),
                           reinterpreted_batch_ndims=tf.rank(base_var))


def make_prior_mvndiag(true_getter, base_var, name=None):  # pylint: disable=unused-argument
  """Builds a standard Gaussian prior for a Variable `base_var`.

  This method is intended to be passed as an argument to
  `build_bayesian_getter`, not called directly.

  Args:
    true_getter: The true getter being wrapped by our custom
      Bayesian getter. This should be used in place of any
      `tf.get_variable()` calls inside this method.
    base_var: The `tf.Variable` for which we are defining a
      variational posterior.
    name: The `Variable`'s name as passed into the Bayesian
      getter (note this will include any prefixes added
      by the calling model's `variable_scope`).

  Returns:
    prior: A tfd.Distribution instance representing the prior
      distribution over weights.
  """
  with tf.name_scope(name, "make_prior_mvndiag", [base_var]):
    sigma = tf.ones_like(base_var)
    return tfd.Independent(tfd.Normal(
        loc=np.float32(0.),
        scale=sigma,
        name="{}_prior".format(name)),
                           reinterpreted_batch_ndims=tf.rank(base_var))


def build_bayesian_getter(prior_fn, posterior_fn,
                          mean_only=False):
  """Builds a custom getter that constructs a prior and posterior.

  When activated using `tf.variable_scope`, this getter intercepts
  all `tf.get_variable` calls, returning a sample from a variational
  posterior distribution in place of a point variable.

  Args:
    prior_fn: A Python `callable` having signature `(true_getter,
      base_var, name)`.  Given a Variable `base_var`, this should
      return a `Distribution` having the same `shape` and `dtype` as
      base_var. To avoid infinite recursion, any trainable parameters
      for this distribution must be created using `true_getter` in
      place of `tf.get_variable`. See `make_prior_mvndiag` for a
      simple example.
    posterior_fn: A Python `callable` having signature
      `(true_getter, base_var, name)`.  Given a Variable `base_var`,
      this should return a `Distribution` having the same `shape` and
      `dtype` as base_var. To avoid infinite recursion, any trainable
      parameters for this distribution must be created using
      `true_getter` in place of `tf.get_variable`. See
      `make_posterior_mvndiag` for a simple example.
    mean_only: Construct a getter that returns the posterior mean
    instead of a posterior sample.

  Returns:
   bayesian_getter: A Python `callable` usable as a `custom_getter` in
     `tf.variable_scope`.  If `mean_only=False`, it returns a random
     draw from a (trainable) approximate posterior, which may be used
     to optimize a stochastic variational bound. If `mean_only`=True,
     it returns the posterior mean, which may (or may not) be useful
     for making predictions.
  """

  def bayesian_getter(getter, name, *args, **kwargs):
    """A `custom_getter` building prior and posterior `Distribution`s."""

    var = getter(name, *args, **kwargs)

    # Save any named arguments (e.g., reuse=True) to also
    # apply to additional variables (e.g., standard deviations)
    # created inside the prior and posterior models.
    default_getter = functools.partial(getter, **kwargs)
    p = prior_fn(default_getter, var, name=name)
    tf.add_to_collection(VI_PRIORS, p)

    q = posterior_fn(default_getter, var, name=name)
    tf.add_to_collection(VI_QDISTS, q)

    draw = q.sample(name="sample")
    kl_monte_carlo = tf.reduce_sum(q.log_prob(draw) - p.log_prob(draw))
    tf.add_to_collection(VI_KL_LOSSES, kl_monte_carlo)

    if mean_only:
      return q.mean()
    return draw

  return bayesian_getter


def bayesianify(build_model,
                inputs,
                observations,
                num_examples,
                prior_fn=make_prior_mvndiag,
                posterior_fn=make_posterior_mvndiag):
  """Make any model Bayesian by wrapping it with a custom Variable getter.

  This method demonstrates the use of a Bayesian getter, which
  intercepts calls to `tf.get_variable` to return a sample from a
  trainable variational posterior distribution over the value of that
  variable. Note that simple variational approximations may not be the
  most effective representation of uncertainty over neural net
  weights; sampling-based methods such as HMC will typically give a
  more faithful approximation of the posterior at the cost of
  additional computation.

  Args:
    build_model: A Python `callable` that takes argument `inputs` (typically
      a `Tensor`) and returns `model`, a `Distribution` matching the
      shape of `observations`.
    inputs: The input to `build_model`.
    observations: A `Tensor` of observed values, used to compute the loss
      `model.log_prob(observations)'.
    num_examples: The total number of examples in the dataset (this is
      different from the shapes of `inputs` and `observations`, which
      are typically minibatches). This is required to scale the KL
      term to produce an unbiased approximation of the ELBO.
    prior_fn: A Python `callable` to build a prior distribution.
      See `build_bayesian_getter` for details.
    posterior_fn: A Python `callable` to build a posterior distribution.
      See `build_bayesian_getter` for details.

  Returns:
    elbo_loss: A scalar `Tensor` computing the negation of the (stochastic)
      variational evidence bound (i.e., `E[elbo_loss] >= -log p(inputs)`).
    model: The `Distribution` over observables returned by `build_model`.
  """

  getter = build_bayesian_getter(prior_fn,
                                 posterior_fn)

  with tf.variable_scope("model", custom_getter=getter):
    model = build_model(inputs)

  avg_logp_y_given_w = tf.reduce_mean(model.log_prob(observations))

  avg_stochastic_kl = sum(tf.get_collection(VI_KL_LOSSES)) / num_examples
  elbo_loss = -(avg_logp_y_given_w - avg_stochastic_kl)

  return elbo_loss, model
