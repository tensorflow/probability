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
"""Monte Carlo expectation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


__all__ = [
    'expectation',
]


def expectation(f, samples, log_prob=None, use_reparametrization=True,
                axis=0, keep_dims=False, name=None):
  """Computes the Monte-Carlo approximation of `E_p[f(X)]`.

  This function computes the Monte-Carlo approximation of an expectation, i.e.,

  ```none
  E_p[f(X)] approx= m**-1 sum_i^m f(x_j),  x_j ~iid p(X)
  ```

  where:

  - `x_j = samples[j, ...]`,
  - `log(p(samples)) = log_prob(samples)` and
  - `m = prod(shape(samples)[axis])`.

  Tricks: Reparameterization and Score-Gradient

  When p is "reparameterized", i.e., a diffeomorphic transformation of a
  parameterless distribution (e.g.,
  `Normal(Y; m, s) <=> Y = sX + m, X ~ Normal(0,1)`), we can swap gradient and
  expectation, i.e.,
  `grad[ Avg{ s_i : i=1...n } ] = Avg{ grad[s_i] : i=1...n }` where
  `S_n = Avg{s_i}` and `s_i = f(x_i), x_i ~ p`.

  However, if p is not reparameterized, TensorFlow's gradient will be incorrect
  since the chain-rule stops at samples of non-reparameterized distributions.
  (The non-differentiated result, `approx_expectation`, is the same regardless
  of `use_reparametrization`.) In this circumstance using the Score-Gradient
  trick results in an unbiased gradient, i.e.,

  ```none
  grad[ E_p[f(X)] ]
  = grad[ int dx p(x) f(x) ]
  = int dx grad[ p(x) f(x) ]
  = int dx [ p'(x) f(x) + p(x) f'(x) ]
  = int dx p(x) [p'(x) / p(x) f(x) + f'(x) ]
  = int dx p(x) grad[ f(x) p(x) / stop_grad[p(x)] ]
  = E_p[ grad[ f(x) p(x) / stop_grad[p(x)] ] ]
  ```

  Unless p is not reparametrized, it is usually preferable to
  `use_reparametrization = True`.

  Warning: users are responsible for verifying `p` is a "reparameterized"
  distribution.

  Example Use:

  ```python
  # Monte-Carlo approximation of a reparameterized distribution, e.g., Normal.

  num_draws = int(1e5)
  p = tfp.distributions.Normal(loc=0., scale=1.)
  q = tfp.distributions.Normal(loc=1., scale=2.)
  exact_kl_normal_normal = tfp.distributions.kl_divergence(p, q)
  # ==> 0.44314718
  approx_kl_normal_normal = tfp.monte_carlo.expectation(
      f=lambda x: p.log_prob(x) - q.log_prob(x),
      samples=p.sample(num_draws, seed=42),
      log_prob=p.log_prob,
      use_reparametrization=(p.reparameterization_type
                             == tfp.distributions.FULLY_REPARAMETERIZED))
  # ==> 0.44632751
  # Relative Error: <1%

  # Monte-Carlo approximation of non-reparameterized distribution,
  # e.g., Bernoulli.

  num_draws = int(1e5)
  p = tfp.distributions.Bernoulli(probs=0.4)
  q = tfp.distributions.Bernoulli(probs=0.8)
  exact_kl_bernoulli_bernoulli = tfp.distributions.kl_divergence(p, q)
  # ==> 0.38190854
  approx_kl_bernoulli_bernoulli = tfp.monte_carlo.expectation(
      f=lambda x: p.log_prob(x) - q.log_prob(x),
      samples=p.sample(num_draws, seed=42),
      log_prob=p.log_prob,
      use_reparametrization=(p.reparameterization_type
                             == tfp.distributions.FULLY_REPARAMETERIZED))
  # ==> 0.38336259
  # Relative Error: <1%

  # For comparing the gradients, see `expectation_test.py`.
  ```

  Note: The above example is for illustration only. To compute approximate
  KL-divergence, the following is preferred:

  ```python
  approx_kl_p_q = bf.monte_carlo_csiszar_f_divergence(
      f=bf.kl_reverse,
      p_log_prob=q.log_prob,
      q=p,
      num_draws=num_draws)
  ```

  Args:
    f: Python callable which can return `f(samples)`.
    samples: `Tensor` of samples used to form the Monte-Carlo approximation of
      `E_p[f(X)]`.  A batch of samples should be indexed by `axis` dimensions.
    log_prob: Python callable which can return `log_prob(samples)`. Must
      correspond to the natural-logarithm of the pdf/pmf of each sample. Only
      required/used if `use_reparametrization=False`.
      Default value: `None`.
    use_reparametrization: Python `bool` indicating that the approximation
      should use the fact that the gradient of samples is unbiased. Whether
      `True` or `False`, this arg only affects the gradient of the resulting
      `approx_expectation`.
      Default value: `True`.
    axis: The dimensions to average. If `None`, averages all
      dimensions.
      Default value: `0` (the left-most dimension).
    keep_dims: If True, retains averaged dimensions using size `1`.
      Default value: `False`.
    name: A `name_scope` for operations created by this function.
      Default value: `None` (which implies "expectation").

  Returns:
    approx_expectation: `Tensor` corresponding to the Monte-Carlo approximation
      of `E_p[f(X)]`.

  Raises:
    ValueError: if `f` is not a Python `callable`.
    ValueError: if `use_reparametrization=False` and `log_prob` is not a Python
      `callable`.
  """

  with tf.compat.v1.name_scope(name, 'expectation', [samples]):
    if not callable(f):
      raise ValueError('`f` must be a callable function.')
    if use_reparametrization:
      return tf.reduce_mean(
          input_tensor=f(samples), axis=axis, keepdims=keep_dims)
    else:
      if not callable(log_prob):
        raise ValueError('`log_prob` must be a callable function.')
      stop = tf.stop_gradient  # For readability.
      x = stop(samples)
      logpx = log_prob(x)
      fx = f(x)  # Call `f` once in case it has side-effects.
      # To achieve this, we use the fact that:
      #   `h(x) - stop(h(x)) == zeros_like(h(x))`
      # but its gradient is grad[h(x)].
      #
      # This technique was published as:
      # Jakob Foerster, Greg Farquhar, Maruan Al-Shedivat, Tim Rocktaeschel,
      # Eric P. Xing, Shimon Whiteson (ICML 2018)
      # "DiCE: The Infinitely Differentiable Monte-Carlo Estimator"
      # https://arxiv.org/abs/1802.05098
      #
      # Unlike using:
      #   fx = fx + stop(fx) * (logpx - stop(logpx)),
      # DiCE ensures that any order gradients of the objective
      # are unbiased gradient estimators.
      #
      # Note that IEEE754 specifies that `x - x == 0.` and `x + 0. == x`, hence
      # this trick loses no precision. For more discussion regarding the
      # relevant portions of the IEEE754 standard, see the StackOverflow
      # question,
      # "Is there a floating point value of x, for which x-x == 0 is false?"
      # http://stackoverflow.com/q/2686644
      dice = fx * tf.exp(logpx - stop(logpx))
      return tf.reduce_mean(input_tensor=dice, axis=axis, keepdims=keep_dims)


def _sample_mean(values):
  """Mean over sample indices.  In this module this is always [0]."""
  return tf.reduce_mean(input_tensor=values, axis=[0])


def _sample_max(values):
  """Max over sample indices.  In this module this is always [0]."""
  return tf.reduce_max(input_tensor=values, axis=[0])


def _get_samples(dist, z, n, seed):
  """Check args and return samples."""
  with tf.compat.v1.name_scope('get_samples', values=[z, n]):
    if (n is None) == (z is None):
      raise ValueError(
          'Must specify exactly one of arguments "n" and "z".  Found: '
          'n = %s, z = %s' % (n, z))
    if n is not None:
      return dist.sample(n, seed=seed)
    else:
      return tf.convert_to_tensor(value=z, name='z')
