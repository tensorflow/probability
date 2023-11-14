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
"""Csiszar f-Divergence and helpers."""

import enum
import functools
import warnings

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python import monte_carlo
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import nest_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal.reparameterization import FULLY_REPARAMETERIZED
from tensorflow_probability.python.stats.leave_one_out import log_soomean_exp

from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'GradientEstimators',
    'amari_alpha',
    'arithmetic_geometric',
    'chi_square',
    'csiszar_vimco',
    'dual_csiszar_function',
    'jeffreys',
    'jensen_shannon',
    'kl_forward',
    'kl_reverse',
    'log1p_abs',
    'modified_gan',
    'monte_carlo_variational_loss',
    'pearson',
    'squared_hellinger',
    'symmetrized_csiszar_function',
    't_power',
    'total_variation',
    'triangular',
]


def _call_fn_maybe_with_seed(fn, args, *, seed=None):
  try:
    return nest_util.call_fn(functools.partial(fn, seed=seed), args)
  except (TypeError, ValueError) as e:
    if ("'seed'" in str(e) or ('one of *args or **kwargs' in str(e))):
      return nest_util.call_fn(fn, args)
    else:
      raise e


class GradientEstimators(enum.Enum):
  """Gradient estimators for variational losses.

  Variational losses implemented by `monte_carlo_variational_loss` are
  defined in general as an expectation of some `fn` under the surrogate
  posterior,

  ```
  loss = expectation(fn, surrogate_posterior)
  ```

  where the expectation is estimated in practice using a finite `sample_size`
  number of samples:

  ```
  zs = surrogate_posterior.sample(sample_size)
  loss_estimate = 1 / sample_size * sum([fn(z) for z in z])
  ```

  Gradient estimators define a stochastic estimate of the *gradient* of the
  above expectation with respect to the parameters of the surrogate posterior.

  Members:
    SCORE_FUNCTION: Also known as REINFORCE [1] or the log-derivative gradient
      estimator [2]. This estimator works with any surrogate posterior, but
      gradient estimates may be very noisy.
    REPARAMETERIZATION: Reparameterization gradients as introduced by Kingma
      and Welling [3]. These require a continuous-valued surrogate that sets
      `reparameterization_type=FULLY_REPARAMETERIZED` (which must implement
      reparameterized sampling either directly or via implicit
      reparameterization [4]), and typically yield much lower-variance gradient
      estimates than the generic score function estimator.
    DOUBLY_REPARAMETERIZED: The doubly-reparameterized estimator presented by
      Tucker et al. [5] for importance-weighted bounds. Note that this includes
      the sticking-the-landing estimator developed by Roeder et al. [6] as a
      special case when `importance_sample_size=1`. Compared to 'vanilla'
      reparameterization, this can provide even lower-variance gradient
      estimates, but requires a copy of the surrogate posterior with no gradient
      to its parameters (passed to the loss as `stopped_surrogate_posterior`),
      and incurs an additional evaluation of the surrogate density at each step.
    VIMCO: An extension of the score-function estimator, introduced by Minh and
      Rezende [7], with reduced variance when `importance_sample_size > 1`.

  #### References

  [1] R. J. Williams. Simple statistical gradient-following algorithms
      for connectionist reinforcement learning.
      __Machine Learning, 8(3-4), 229–256__, 1992.

  [2] Shakir Mohamed. Machine Learning Trick of the Day: Log Derivative Trick.
      2015.
      https://blog.shakirm.com/2015/11/machine-learning-trick-of-the-day-5-log-derivative-trick/

  [3] Diederik P. Kingma, and Max Welling. Auto-encoding variational bayes.
      __arXiv preprint arXiv:1312.6114__, 2013. https://arxiv.org/abs/1312.6114

  [4] Michael Figurnov, Shakir Mohamed, and Andriy Mnih. Implicit
      reparameterization gradients. __arXiv preprint arXiv:1805.08498__, 2018.
      https://arxiv.org/abs/1805.08498

  [5] George Tucker, Dieterich Lawson, Shixiang Gu, and Chris J. Maddison.
      Doubly reparameterized gradient estimators for Monte Carlo objectives.
      __arXiv preprint arXiv:1810.04152__, 2018.
      https://arxiv.org/abs/1810.04152

  [6] Geoffrey Roeder, Yuhuai Wu, and David Duvenaud. Sticking the landing:
      Simple, lower-variance gradient estimators for variational inference.
      __arXiv preprint arXiv:1703.09194__, 2017.
      https://arxiv.org/abs/1703.09194

  [7] Andriy Mnih and Danilo Rezende. Variational Inference for Monte Carlo
      objectives. In _International Conference on Machine Learning_, 2016.
      https://arxiv.org/abs/1602.06725
  """
  SCORE_FUNCTION = 0
  REPARAMETERIZATION = 1
  DOUBLY_REPARAMETERIZED = 2
  VIMCO = 3


def amari_alpha(logu, alpha=1., self_normalized=False, name=None):
  """The Amari-alpha Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  When `self_normalized = True`, the Amari-alpha Csiszar-function is:

  ```none
  f(u) = { -log(u) + (u - 1),     alpha = 0
         { u log(u) - (u - 1),    alpha = 1
         { [(u**alpha - 1) - alpha (u - 1)] / (alpha (alpha - 1)),    otherwise
  ```

  When `self_normalized = False` the `(u - 1)` terms are omitted.

  Warning: when `alpha != 0` and/or `self_normalized = True` this function makes
  non-log-space calculations and may therefore be numerically unstable for
  `|logu| >> 0`.

  For more information, see:
    A. Cichocki and S. Amari. "Families of Alpha-Beta-and GammaDivergences:
    Flexible and Robust Measures of Similarities." Entropy, vol. 12, no. 6, pp.
    1532-1568, 2010.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    alpha: `float`-like Python scalar. (See Mathematical Details for meaning.)
    self_normalized: Python `bool` indicating whether `f'(u=1)=0`. When
      `f'(u=1)=0` the implied Csiszar f-Divergence remains non-negative even
      when `p, q` are unnormalized measures.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    amari_alpha_of_u: `float`-like `Tensor` of the Csiszar-function evaluated
      at `u = exp(logu)`.

  Raises:
    TypeError: if `alpha` is `None` or a `Tensor`.
    TypeError: if `self_normalized` is `None` or a `Tensor`.
  """
  with tf.name_scope(name or 'amari_alpha'):
    if tf.get_static_value(alpha) is None:
      raise TypeError('Argument `alpha` cannot be `None` or `Tensor` type.')
    if tf.get_static_value(self_normalized) is None:
      raise TypeError(
          'Argument `self_normalized` cannot be `None` or `Tensor` type.')

    logu = tf.convert_to_tensor(logu, name='logu')

    if alpha == 0.:
      f = -logu
    elif alpha == 1.:
      f = tf.exp(logu) * logu
    else:
      f = tf.math.expm1(alpha * logu) / (alpha * (alpha - 1.))

    if not self_normalized:
      return f

    if alpha == 0.:
      return f + tf.math.expm1(logu)
    elif alpha == 1.:
      return f - tf.math.expm1(logu)
    else:
      return f - tf.math.expm1(logu) / (alpha - 1.)


def kl_reverse(logu, self_normalized=False, name=None):
  """The reverse Kullback-Leibler Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  When `self_normalized = True`, the KL-reverse Csiszar-function is:

  ```none
  f(u) = -log(u) + (u - 1)
  ```

  When `self_normalized = False` the `(u - 1)` term is omitted.

  Observe that as an f-Divergence, this Csiszar-function implies:

  ```none
  D_f[p, q] = KL[q, p]
  ```

  The KL is "reverse" because in maximum likelihood we think of minimizing `q`
  as in `KL[p, q]`.

  Warning: when self_normalized = True` this function makes non-log-space
  calculations and may therefore be numerically unstable for `|logu| >> 0`.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    self_normalized: Python `bool` indicating whether `f'(u=1)=0`. When
      `f'(u=1)=0` the implied Csiszar f-Divergence remains non-negative even
      when `p, q` are unnormalized measures.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    kl_reverse_of_u: `float`-like `Tensor` of the Csiszar-function evaluated at
      `u = exp(logu)`.

  Raises:
    TypeError: if `self_normalized` is `None` or a `Tensor`.
  """

  with tf.name_scope(name or 'kl_reverse'):
    return amari_alpha(logu, alpha=0., self_normalized=self_normalized)


def kl_forward(logu, self_normalized=False, name=None):
  """The forward Kullback-Leibler Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  When `self_normalized = True`, the KL-forward Csiszar-function is:

  ```none
  f(u) = u log(u) - (u - 1)
  ```

  When `self_normalized = False` the `(u - 1)` term is omitted.

  Observe that as an f-Divergence, this Csiszar-function implies:

  ```none
  D_f[p, q] = KL[p, q]
  ```

  The KL is "forward" because in maximum likelihood we think of minimizing `q`
  as in `KL[p, q]`.

  Warning: this function makes non-log-space calculations and may therefore be
  numerically unstable for `|logu| >> 0`.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    self_normalized: Python `bool` indicating whether `f'(u=1)=0`. When
      `f'(u=1)=0` the implied Csiszar f-Divergence remains non-negative even
      when `p, q` are unnormalized measures.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    kl_forward_of_u: `float`-like `Tensor` of the Csiszar-function evaluated at
      `u = exp(logu)`.

  Raises:
    TypeError: if `self_normalized` is `None` or a `Tensor`.
  """

  with tf.name_scope(name or 'kl_forward'):
    return amari_alpha(logu, alpha=1., self_normalized=self_normalized)


def jensen_shannon(logu, self_normalized=False, name=None):
  """The Jensen-Shannon Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  When `self_normalized = True`, the Jensen-Shannon Csiszar-function is:

  ```none
  f(u) = u log(u) - (1 + u) log(1 + u) + (u + 1) log(2)
  ```

  When `self_normalized = False` the `(u + 1) log(2)` term is omitted.

  Observe that as an f-Divergence, this Csiszar-function implies:

  ```none
  D_f[p, q] = KL[p, m] + KL[q, m]
  m(x) = 0.5 p(x) + 0.5 q(x)
  ```

  In a sense, this divergence is the "reverse" of the Arithmetic-Geometric
  f-Divergence.

  This Csiszar-function induces a symmetric f-Divergence, i.e.,
  `D_f[p, q] = D_f[q, p]`.

  Warning: this function makes non-log-space calculations and may therefore be
  numerically unstable for `|logu| >> 0`.

  For more information, see:
    Lin, J. "Divergence measures based on the Shannon entropy." IEEE Trans.
    Inf. Th., 37, 145-151, 1991.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    self_normalized: Python `bool` indicating whether `f'(u=1)=0`. When
      `f'(u=1)=0` the implied Csiszar f-Divergence remains non-negative even
      when `p, q` are unnormalized measures.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    jensen_shannon_of_u: `float`-like `Tensor` of the Csiszar-function
      evaluated at `u = exp(logu)`.
  """

  with tf.name_scope(name or 'jensen_shannon'):
    logu = tf.convert_to_tensor(logu, name='logu')
    y = tf.nn.softplus(logu)
    if self_normalized:
      y -= np.log(2.)
    # TODO(jvdillon): Maybe leverage the fact that:
    # (x-sp(x))*exp(x) approx= expm1(-1.1x + 0.5) for x>12?
    # Basically, take advantage of x approx= softplus(x) for x>>0.
    return (logu - y) * tf.exp(logu) - y


def arithmetic_geometric(logu, self_normalized=False, name=None):
  """The Arithmetic-Geometric Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  When `self_normalized = True` the Arithmetic-Geometric Csiszar-function is:

  ```none
  f(u) = (1 + u) log( (1 + u) / sqrt(u) ) - (1 + u) log(2)
  ```

  When `self_normalized = False` the `(1 + u) log(2)` term is omitted.

  Observe that as an f-Divergence, this Csiszar-function implies:

  ```none
  D_f[p, q] = KL[m, p] + KL[m, q]
  m(x) = 0.5 p(x) + 0.5 q(x)
  ```

  In a sense, this divergence is the "reverse" of the Jensen-Shannon
  f-Divergence.

  This Csiszar-function induces a symmetric f-Divergence, i.e.,
  `D_f[p, q] = D_f[q, p]`.

  Warning: this function makes non-log-space calculations and may therefore be
  numerically unstable for `|logu| >> 0`.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    self_normalized: Python `bool` indicating whether `f'(u=1)=0`. When
      `f'(u=1)=0` the implied Csiszar f-Divergence remains non-negative even
      when `p, q` are unnormalized measures.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    arithmetic_geometric_of_u: `float`-like `Tensor` of the
      Csiszar-function evaluated at `u = exp(logu)`.
  """

  with tf.name_scope(name or 'arithmetic_geometric'):
    logu = tf.convert_to_tensor(logu, name='logu')
    y = tf.nn.softplus(logu) - 0.5 * logu
    if self_normalized:
      y -= np.log(2.)
    return (1. + tf.exp(logu)) * y


def total_variation(logu, name=None):
  """The Total Variation Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  The Total-Variation Csiszar-function is:

  ```none
  f(u) = 0.5 |u - 1|
  ```

  Warning: this function makes non-log-space calculations and may therefore be
  numerically unstable for `|logu| >> 0`.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    total_variation_of_u: `float`-like `Tensor` of the Csiszar-function
      evaluated at `u = exp(logu)`.
  """

  with tf.name_scope(name or 'total_variation'):
    logu = tf.convert_to_tensor(logu, name='logu')
    return 0.5 * tf.abs(tf.math.expm1(logu))


def pearson(logu, name=None):
  """The Pearson Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  The Pearson Csiszar-function is:

  ```none
  f(u) = (u - 1)**2
  ```

  Warning: this function makes non-log-space calculations and may therefore be
  numerically unstable for `|logu| >> 0`.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    pearson_of_u: `float`-like `Tensor` of the Csiszar-function evaluated at
      `u = exp(logu)`.
  """

  with tf.name_scope(name or 'pearson'):
    logu = tf.convert_to_tensor(logu, name='logu')
    return tf.square(tf.math.expm1(logu))


def squared_hellinger(logu, name=None):
  """The Squared-Hellinger Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  The Squared-Hellinger Csiszar-function is:

  ```none
  f(u) = (sqrt(u) - 1)**2
  ```

  This Csiszar-function induces a symmetric f-Divergence, i.e.,
  `D_f[p, q] = D_f[q, p]`.

  Warning: this function makes non-log-space calculations and may therefore be
  numerically unstable for `|logu| >> 0`.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    squared_hellinger_of_u: `float`-like `Tensor` of the Csiszar-function
      evaluated at `u = exp(logu)`.
  """

  with tf.name_scope(name or 'squared_hellinger'):
    logu = tf.convert_to_tensor(logu, name='logu')
    return pearson(0.5 * logu)


def triangular(logu, name=None):
  """The Triangular Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  The Triangular Csiszar-function is:

  ```none
  f(u) = (u - 1)**2 / (1 + u)
  ```

  This Csiszar-function induces a symmetric f-Divergence, i.e.,
  `D_f[p, q] = D_f[q, p]`.

  Warning: this function makes non-log-space calculations and may therefore be
  numerically unstable for `|logu| >> 0`.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    triangular_of_u: `float`-like `Tensor` of the Csiszar-function evaluated
      at `u = exp(logu)`.
  """

  with tf.name_scope(name or 'triangular'):
    logu = tf.convert_to_tensor(logu, name='logu')
    return pearson(logu) / (1. + tf.exp(logu))


def t_power(logu, t, self_normalized=False, name=None):
  """The T-Power Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  When `self_normalized = True` the T-Power Csiszar-function is:

  ```none
  f(u) = s [ u**t - 1 - t(u - 1) ]
  s = { -1   0 < t < 1
      { +1   otherwise
  ```

  When `self_normalized = False` the `- t(u - 1)` term is omitted.

  This is similar to the `amari_alpha` Csiszar-function, with the associated
  divergence being the same up to factors depending only on `t`.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    t:  `Tensor` of same `dtype` as `logu` and broadcastable shape.
    self_normalized: Python `bool` indicating whether `f'(u=1)=0`.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    t_power_of_u: `float`-like `Tensor` of the Csiszar-function evaluated
      at `u = exp(logu)`.
  """
  with tf.name_scope(name or 't_power'):
    logu = tf.convert_to_tensor(logu, name='logu')
    t = tf.convert_to_tensor(
        t, dtype=dtype_util.base_dtype(logu.dtype), name='t')
    fu = tf.math.expm1(t * logu)
    if self_normalized:
      fu = fu - t * tf.math.expm1(logu)
    return tf.where((0 < t) & (t < 1), -fu, fu)


def log1p_abs(logu, name=None):
  """The log1p-abs Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  The Log1p-Abs Csiszar-function is:

  ```none
  f(u) = u**(sign(u-1)) - 1
  ```

  This function is so-named because it was invented from the following recipe.
  Choose a convex function g such that g(0)=0 and solve for f:

  ```none
  log(1 + f(u)) = g(log(u)).
    <=>
  f(u) = exp(g(log(u))) - 1
  ```

  That is, the graph is identically `g` when y-axis is `log1p`-domain and x-axis
  is `log`-domain.

  Warning: this function makes non-log-space calculations and may therefore be
  numerically unstable for `|logu| >> 0`.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    log1p_abs_of_u: `float`-like `Tensor` of the Csiszar-function evaluated
      at `u = exp(logu)`.
  """

  with tf.name_scope(name or 'log1p_abs'):
    logu = tf.convert_to_tensor(logu, name='logu')
    return tf.math.expm1(tf.abs(logu))


def jeffreys(logu, name=None):
  """The Jeffreys Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  The Jeffreys Csiszar-function is:

  ```none
  f(u) = 0.5 ( u log(u) - log(u) )
       = 0.5 kl_forward + 0.5 kl_reverse
       = symmetrized_csiszar_function(kl_reverse)
       = symmetrized_csiszar_function(kl_forward)
  ```

  This Csiszar-function induces a symmetric f-Divergence, i.e.,
  `D_f[p, q] = D_f[q, p]`.

  Warning: this function makes non-log-space calculations and may therefore be
  numerically unstable for `|logu| >> 0`.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    jeffreys_of_u: `float`-like `Tensor` of the Csiszar-function evaluated
      at `u = exp(logu)`.
  """

  with tf.name_scope(name or 'jeffreys'):
    logu = tf.convert_to_tensor(logu, name='logu')
    return 0.5 * tf.math.expm1(logu) * logu


def chi_square(logu, name=None):
  """The chi-Square Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  The Chi-square Csiszar-function is:

  ```none
  f(u) = u**2 - 1
  ```

  Warning: this function makes non-log-space calculations and may therefore be
  numerically unstable for `|logu| >> 0`.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    chi_square_of_u: `float`-like `Tensor` of the Csiszar-function evaluated
      at `u = exp(logu)`.
  """

  with tf.name_scope(name or 'chi_square'):
    logu = tf.convert_to_tensor(logu, name='logu')
    return tf.math.expm1(2. * logu)


def modified_gan(logu, self_normalized=False, name=None):
  """The Modified-GAN Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  When `self_normalized = True` the modified-GAN (Generative/Adversarial
  Network) Csiszar-function is:

  ```none
  f(u) = log(1 + u) - log(u) + 0.5 (u - 1)
  ```

  When `self_normalized = False` the `0.5 (u - 1)` is omitted.

  The unmodified GAN Csiszar-function is identical to Jensen-Shannon (with
  `self_normalized = False`).

  Warning: this function makes non-log-space calculations and may therefore be
  numerically unstable for `|logu| >> 0`.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    self_normalized: Python `bool` indicating whether `f'(u=1)=0`. When
      `f'(u=1)=0` the implied Csiszar f-Divergence remains non-negative even
      when `p, q` are unnormalized measures.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    chi_square_of_u: `float`-like `Tensor` of the Csiszar-function evaluated
      at `u = exp(logu)`.
  """

  with tf.name_scope(name or 'chi_square'):
    logu = tf.convert_to_tensor(logu, name='logu')
    y = tf.nn.softplus(logu) - logu
    if self_normalized:
      y += 0.5 * tf.math.expm1(logu)
    return y


def dual_csiszar_function(logu, csiszar_function, name=None):
  """Calculates the dual Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  The Csiszar-dual is defined as:

  ```none
  f^*(u) = u f(1 / u)
  ```

  where `f` is some other Csiszar-function.

  For example, the dual of `kl_reverse` is `kl_forward`, i.e.,

  ```none
  f(u) = -log(u)
  f^*(u) = u f(1 / u) = -u log(1 / u) = u log(u)
  ```

  The dual of the dual is the original function:

  ```none
  f^**(u) = {u f(1/u)}^*(u) = u (1/u) f(1/(1/u)) = f(u)
  ```

  Warning: this function makes non-log-space calculations and may therefore be
  numerically unstable for `|logu| >> 0`.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    csiszar_function: Python `callable` representing a Csiszar-function over
      log-domain.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    dual_f_of_u: `float`-like `Tensor` of the result of calculating the dual of
      `f` at `u = exp(logu)`.
  """

  with tf.name_scope(name or 'dual_csiszar_function'):
    return tf.exp(logu) * csiszar_function(-logu)


def symmetrized_csiszar_function(logu, csiszar_function, name=None):
  """Symmetrizes a Csiszar-function in log-space.

  A Csiszar-function is a member of,

  ```none
  F = { f:R_+ to R : f convex }.
  ```

  The symmetrized Csiszar-function is defined as:

  ```none
  f_g(u) = 0.5 g(u) + 0.5 u g (1 / u)
  ```

  where `g` is some other Csiszar-function.

  We say the function is "symmetrized" because:

  ```none
  D_{f_g}[p, q] = D_{f_g}[q, p]
  ```

  for all `p << >> q` (i.e., `support(p) = support(q)`).

  There exists alternatives for symmetrizing a Csiszar-function. For example,

  ```none
  f_g(u) = max(f(u), f^*(u)),
  ```

  where `f^*` is the dual Csiszar-function, also implies a symmetric
  f-Divergence.

  Example:

  When either of the following functions are symmetrized, we obtain the
  Jensen-Shannon Csiszar-function, i.e.,

  ```none
  g(u) = -log(u) - (1 + u) log((1 + u) / 2) + u - 1
  h(u) = log(4) + 2 u log(u / (1 + u))
  ```

  implies,

  ```none
  f_g(u) = f_h(u) = u log(u) - (1 + u) log((1 + u) / 2)
         = jensen_shannon(log(u)).
  ```

  Warning: this function makes non-log-space calculations and may therefore be
  numerically unstable for `|logu| >> 0`.

  Args:
    logu: `float`-like `Tensor` representing `log(u)` from above.
    csiszar_function: Python `callable` representing a Csiszar-function over
      log-domain.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    symmetrized_g_of_u: `float`-like `Tensor` of the result of applying the
      symmetrization of `g` evaluated at `u = exp(logu)`.
  """

  with tf.name_scope(name or 'symmetrized_csiszar_function'):
    logu = tf.convert_to_tensor(logu, name='logu')
    return 0.5 * (csiszar_function(logu) +
                  dual_csiszar_function(logu, csiszar_function))


def _choose_gradient_estimator(use_reparameterization,
                               reparameterization_types):
  """Infers a default gradient estimator from args to a variational loss."""
  if use_reparameterization is None:
    use_reparameterization = all(
        reparameterization_type == FULLY_REPARAMETERIZED
        for reparameterization_type in reparameterization_types)
  if use_reparameterization:
    return GradientEstimators.REPARAMETERIZATION
  else:
    warnings.warn(
        'Using score-function gradient estimate, which may have high '
        'variance. To disable this warning, explicitly pass '
        '`gradient_estimator=tfp.vi.GradientEstimators.SCORE_FUNCTION`.')
    return GradientEstimators.SCORE_FUNCTION


@deprecation.deprecated_args(
    '2022-06-01',
    'Please pass either '
    '`gradient_estimator=GradientEstimators.REPARAMETERIZATION` (for '
    '`use_reparameterization=True`) or '
    '`gradient_estimator=GradientEstimators.SCORE_FUNCTION` (for '
    '`use_reparameterization=False`).',
    'use_reparameterization')
def monte_carlo_variational_loss(
    target_log_prob_fn,
    surrogate_posterior,
    sample_size=1,
    importance_sample_size=1,
    discrepancy_fn=kl_reverse,
    use_reparameterization=None,
    gradient_estimator=None,
    stopped_surrogate_posterior=None,
    seed=None,
    name=None):
  """Monte-Carlo approximation of an f-Divergence variational loss.

  Variational losses measure the divergence between an unnormalized target
  distribution `p` (provided via `target_log_prob_fn`) and a surrogate
  distribution `q` (provided as `surrogate_posterior`). When the
  target distribution is an unnormalized posterior from conditioning a model on
  data, minimizing the loss with respect to the parameters of
  `surrogate_posterior` performs approximate posterior inference.

  This function defines losses of the form
  `E_q[discrepancy_fn(log(u))]`, where `u = p(z) / q(z)` in the (default) case
  where `importance_sample_size == 1`, and
  `u = mean([p(z[k]) / q(z[k]) for k in range(importance_sample_size)]))` more
  generally. These losses are sometimes known as f-divergences [1, 2].

  The default behavior (`discrepancy_fn == tfp.vi.kl_reverse`, where
  `tfp.vi.kl_reverse = lambda logu: -logu`, and
  `importance_sample_size == 1`) computes an unbiased estimate of the standard
  evidence lower bound (ELBO) [3]. The bound may be tightened by setting
  `importance_sample_size > 1` [4], and the variance of the estimate reduced by
  setting `sample_size > 1`. Other discrepancies of interest
  available under `tfp.vi` include the forward `KL[p||q]`, total variation
  distance, Amari alpha-divergences, and [more](
  https://en.wikipedia.org/wiki/F-divergence).

  Args:
    target_log_prob_fn: Python callable that takes a set of `Tensor` arguments
      and returns a `Tensor` log-density. Given
      `q_sample = surrogate_posterior.sample(sample_size)`, this
      will be called as `target_log_prob_fn(*q_sample)` if `q_sample` is a list
      or a tuple, `target_log_prob_fn(**q_sample)` if `q_sample` is a
      dictionary, or `target_log_prob_fn(q_sample)` if `q_sample` is a `Tensor`.
      It should support batched evaluation, i.e., should return a result of
      shape `[sample_size]`.
    surrogate_posterior: A `tfp.distributions.Distribution`
      instance defining a variational posterior (could be a
      `tfd.JointDistribution`). If using `tf.Variable` parameters, the
      distribution's `log_prob` and (if reparameterizeable) `sample` methods
      must directly invoke all ops that generate gradients to the underlying
      variables. One way to ensure this is to use `tfp.util.TransformedVariable`
      and/or `tfp.util.DeferredTensor` to represent any parameters defined as
      transformations of unconstrained variables, so that the transformations
      execute at runtime instead of at distribution creation.
    sample_size: Integer scalar number of Monte Carlo samples used to
      approximate the variational divergence. Larger values may stabilize
      the optimization, but at higher cost per step in time and memory.
      Default value: `1`.
    importance_sample_size: Python `int` number of terms used to define an
      importance-weighted divergence. If `importance_sample_size > 1`, then the
      `surrogate_posterior` is optimized to function as an importance-sampling
      proposal distribution. In this case it often makes sense to use
      importance sampling to approximate posterior expectations (see
      `tfp.vi.fit_surrogate_posterior` for an example).
      Default value: `1`.
    discrepancy_fn: Python `callable` representing a Csiszar `f` function in
      in log-space. That is, `discrepancy_fn(log(u)) = f(u)`, where `f` is
      convex in `u`.
      Default value: `tfp.vi.kl_reverse`.
    use_reparameterization: Deprecated; use `gradient_estimator` instead.
    gradient_estimator: Optional element from `tfp.vi.GradientEstimators`
      specifying the stochastic gradient estimator to associate with the
      variational loss. If `None`, a default estimator (either score-function or
      reparameterization) is chosen based on
      `surrogate_posterior.reparameterization_type`.
      Default value: `None`.
    stopped_surrogate_posterior: Optional copy of `surrogate_posterior` with
      stopped gradients to the parameters, e.g.,
      `tfd.Normal(loc=tf.stop_gradient(loc), scale=tf.stop_gradient(scale))`.
      Required if and only if
      `gradient_estimator == tfp.vi.GradientEstimators.DOUBLY_REPARAMETERIZED`.
      Default value: `None`.
    seed: PRNG seed for `surrogate_posterior.sample`; see
      `tfp.random.sanitize_seed` for details.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    monte_carlo_variational_loss: `float`-like `Tensor` Monte Carlo
      approximation of the Csiszar f-Divergence.

  Raises:
    ValueError: if `surrogate_posterior` is not a reparameterized
      distribution and `use_reparameterization = True`. A distribution is said
      to be "reparameterized" when its samples are generated by transforming the
      samples of another distribution that does not depend on the first
      distribution's parameters. This property ensures the gradient with respect
      to parameters is valid.
    TypeError: if `target_log_prob_fn` is not a Python `callable`.

  #### Csiszar f-divergences

  A Csiszar function `f` is a convex function from `R^+` (the positive reals)
  to `R`. The Csiszar f-Divergence is given by:

  ```none
  D_f[p(X), q(X)] := E_{q(X)}[ f( p(X) / q(X) ) ]
                  ~= m**-1 sum_j^m f( p(x_j) / q(x_j) ),
                             where x_j ~iid q(X)
  ```

  For example, `f = lambda u: -log(u)` recovers `KL[q||p]`, while `f =
  lambda u: u * log(u)` recovers the forward `KL[p||q]`. These and other
  functions are available in `tfp.vi`.

  #### Example Application:

  The Csiszar f-Divergence is a useful framework for variational inference.
  I.e., observe that,

  ```none
  f(p(x)) =  f( E_{q(Z | x)}[ p(x, Z) / q(Z | x) ] )
          <= E_{q(Z | x)}[ f( p(x, Z) / q(Z | x) ) ]
          := D_f[p(x, Z), q(Z | x)]
  ```

  The inequality follows from the fact that the "perspective" of `f`, i.e.,
  `(s, t) |-> t f(s / t))`, is convex in `(s, t)` when `s/t in domain(f)` and
  `t` is a real. Since the above framework includes the popular Evidence Lower
  BOund (ELBO) as a special case, i.e., `f(u) = -log(u)`, we call this framework
  "Evidence Divergence Bound Optimization" (EDBO).

  #### References:

  [1]: https://en.wikipedia.org/wiki/F-divergence

  [2]: Ali, Syed Mumtaz, and Samuel D. Silvey. "A general class of coefficients
       of divergence of one distribution from another." Journal of the Royal
       Statistical Society: Series B (Methodological) 28.1 (1966): 131-142.

  [3]: Christopher M. Bishop. Pattern Recognition and Machine Learning.
       Springer, 2006.

  [4]  Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov. Importance Weighted
       Autoencoders. In _International Conference on Learning
       Representations_, 2016. https://arxiv.org/abs/1509.00519

  """
  with tf.name_scope(name or 'monte_carlo_variational_loss'):
    if not callable(target_log_prob_fn):
      raise TypeError('`target_log_prob_fn` must be a Python `callable`'
                      'function.')

    sample_seed, target_seed = samplers.split_seed(seed, 2)
    reparameterization_types = tf.nest.flatten(
        surrogate_posterior.reparameterization_type)
    if gradient_estimator is None:
      gradient_estimator = _choose_gradient_estimator(
          use_reparameterization=use_reparameterization,
          reparameterization_types=reparameterization_types)

    if gradient_estimator == GradientEstimators.VIMCO:
      return csiszar_vimco(f=discrepancy_fn,
                           p_log_prob=target_log_prob_fn,
                           q=surrogate_posterior,
                           num_draws=importance_sample_size,
                           num_batch_draws=sample_size,
                           seed=seed)
    if gradient_estimator == GradientEstimators.SCORE_FUNCTION:
      if tf.get_static_value(importance_sample_size) != 1:
        # TODO(b/213378570): Support score function gradients for
        # importance-weighted bounds.
        raise ValueError('Score-function gradients are not supported for '
                         'losses with `importance_sample_size != 1`.')
      # Score fn objective requires explicit gradients of `log_prob`.
      q_samples = surrogate_posterior.sample(
          [sample_size * importance_sample_size], seed=sample_seed)
      q_lp = None
    else:
      if any(reparameterization_type != FULLY_REPARAMETERIZED
             for reparameterization_type in reparameterization_types):
        warnings.warn(
            'Reparameterization gradients requested, but '
            '`surrogate_posterior.reparameterization_type` is not fully '
            'reparameterized (saw: {}). Gradient estimates may be '
            'biased.'.format(surrogate_posterior.reparameterization_type))
      # Attempt to avoid bijector inverses by computing the surrogate log prob
      # during the forward sampling pass.
      q_samples, q_lp = surrogate_posterior.experimental_sample_and_log_prob(
          [sample_size * importance_sample_size], seed=sample_seed)

    return monte_carlo.expectation(
        f=_make_importance_weighted_divergence_fn(
            target_log_prob_fn,
            surrogate_posterior=surrogate_posterior,
            discrepancy_fn=discrepancy_fn,
            precomputed_surrogate_log_prob=q_lp,
            importance_sample_size=importance_sample_size,
            gradient_estimator=gradient_estimator,
            stopped_surrogate_posterior=stopped_surrogate_posterior,
            seed=target_seed),
        samples=q_samples,
        # Log-prob is only used if `gradient_estimator == SCORE_FUNCTION`.
        log_prob=surrogate_posterior.log_prob,
        use_reparameterization=(
            gradient_estimator != GradientEstimators.SCORE_FUNCTION))


def _make_importance_weighted_divergence_fn(
    target_log_prob_fn,
    surrogate_posterior,
    discrepancy_fn,
    precomputed_surrogate_log_prob=None,
    importance_sample_size=1,
    gradient_estimator=GradientEstimators.REPARAMETERIZATION,
    stopped_surrogate_posterior=None,
    seed=None):
  """Defines a function to compute an importance-weighted divergence."""

  def divergence_fn(q_samples):
    q_lp = precomputed_surrogate_log_prob
    target_log_prob = _call_fn_maybe_with_seed(
        target_log_prob_fn, q_samples, seed=seed)

    if gradient_estimator == GradientEstimators.DOUBLY_REPARAMETERIZED:
      # Sticking-the-landing is the special case of doubly-reparameterized
      # gradients with `importance_sample_size=1`.
      q_lp = stopped_surrogate_posterior.log_prob(q_samples)
    else:
      if q_lp is None:
        q_lp = surrogate_posterior.log_prob(q_samples)
    log_weights = target_log_prob - q_lp
    return discrepancy_fn(log_weights)

  def importance_weighted_divergence_fn(q_samples):
    q_lp = precomputed_surrogate_log_prob
    if q_lp is None:
      q_lp = surrogate_posterior.log_prob(q_samples)
    target_log_prob = _call_fn_maybe_with_seed(
        target_log_prob_fn, q_samples, seed=seed)
    log_weights = target_log_prob - q_lp

    # Explicitly break out `importance_sample_size` as a separate axis.
    log_weights = tf.reshape(
        log_weights,
        ps.concat([[-1, importance_sample_size],
                   ps.shape(log_weights)[1:]], axis=0))
    log_sum_weights = tf.reduce_logsumexp(log_weights, axis=1)
    log_avg_weights = log_sum_weights - tf.math.log(
        tf.cast(importance_sample_size, dtype=log_weights.dtype))

    if gradient_estimator == GradientEstimators.DOUBLY_REPARAMETERIZED:
      # Adapted from original implementation at
      # https://github.com/google-research/google-research/blob/master/dreg_estimators/model.py
      normalized_weights = tf.stop_gradient(tf.nn.softmax(log_weights, axis=1))
      log_weights_with_stopped_q = tf.reshape(
          target_log_prob - stopped_surrogate_posterior.log_prob(q_samples),
          ps.shape(log_weights))
      dreg_objective = tf.reduce_sum(
          log_weights_with_stopped_q * tf.square(normalized_weights), axis=1)
      # Replace the objective's gradient with the doubly-reparameterized
      # gradient.
      log_avg_weights = tf.stop_gradient(log_avg_weights) + (
          dreg_objective - tf.stop_gradient(dreg_objective))

    return discrepancy_fn(log_avg_weights)

  if tf.get_static_value(importance_sample_size) == 1:
    return divergence_fn
  return importance_weighted_divergence_fn


@deprecation.deprecated(
    '2022-06-01',
    'Use `monte_carlo_variational_loss` with '
    '`gradient_estimator=tfp.vi.GradientEstimators.VIMCO`.')
def csiszar_vimco(f,
                  p_log_prob,
                  q,
                  num_draws,
                  num_batch_draws=1,
                  seed=None,
                  name=None):
  """Use VIMCO to lower the variance of gradient[csiszar_function(log(Avg(u))].

  This function generalizes VIMCO [(Mnih and Rezende, 2016)][1] to Csiszar
  f-Divergences.

  Note: if `q.reparameterization_type = tfd.FULLY_REPARAMETERIZED`,
  consider using `monte_carlo_variational_loss`.

  The VIMCO loss is:

  ```none
  vimco = f(log(Avg{u[i] : i=0,...,m-1}))
  where,
    logu[i] = log( p(x, h[i]) / q(h[i] | x) )
    h[i] iid~ q(H | x)
  ```

  Interestingly, the VIMCO gradient is not the naive gradient of `vimco`.
  Rather, it is characterized by:

  ```none
  grad[vimco] - variance_reducing_term
  where,
    variance_reducing_term = Sum{ grad[log q(h[i] | x)] *
                                    (vimco - f(log Avg{h[j;i] : j=0,...,m-1}))
                                 : i=0, ..., m-1 }
    h[j;i] = { u[j]                             j!=i
             { GeometricAverage{ u[k] : k!=i}   j==i
  ```

  (We omitted `stop_gradient` for brevity. See implementation for more details.)

  The `Avg{h[j;i] : j}` term is a kind of "swap-out average" where the `i`-th
  element has been replaced by the leave-`i`-out Geometric-average.

  This implementation prefers numerical precision over efficiency, i.e.,
  `O(num_draws * num_batch_draws * prod(batch_shape) * prod(event_shape))`.
  (The constant may be fairly large, perhaps around 12.)

  Args:
    f: Python `callable` representing a Csiszar-function in log-space.
    p_log_prob: Python `callable` representing the natural-log of the
      probability under distribution `p`. (In variational inference `p` is the
      joint distribution.)
    q: `tf.Distribution`-like instance; must implement: `sample(n, seed)`, and
      `log_prob(x)`. (In variational inference `q` is the approximate posterior
      distribution.)
    num_draws: Integer scalar number of draws used to approximate the
      f-Divergence expectation.
    num_batch_draws: Integer scalar number of draws used to approximate the
      f-Divergence expectation.
    seed: PRNG seed for `q.sample`; see `tfp.random.sanitize_seed` for details.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    vimco: The Csiszar f-Divergence generalized VIMCO objective.

  Raises:
    ValueError: if `num_draws < 2`.

  #### References

  [1]: Andriy Mnih and Danilo Rezende. Variational Inference for Monte Carlo
       objectives. In _International Conference on Machine Learning_, 2016.
       https://arxiv.org/abs/1602.06725
  """
  with tf.name_scope(name or 'csiszar_vimco'):
    if num_draws < 2:
      raise ValueError('Must specify num_draws > 1.')
    stop = tf.stop_gradient  # For readability.

    sample_seed, target_seed = samplers.split_seed(seed, 2)
    q_sample = q.sample(sample_shape=[num_draws, num_batch_draws],
                        seed=sample_seed)
    x = tf.nest.map_structure(stop, q_sample)
    logqx = q.log_prob(x)
    logu = _call_fn_maybe_with_seed(p_log_prob, x, seed=target_seed) - logqx
    f_log_sooavg_u, f_log_avg_u = map(f, log_soomean_exp(logu, axis=0))

    dotprod = tf.reduce_sum(
        logqx * stop(f_log_avg_u - f_log_sooavg_u),
        axis=0)  # Sum over iid samples.
    # We now rewrite f_log_avg_u so that:
    #   `grad[f_log_avg_u] := grad[f_log_avg_u + dotprod]`.
    # To achieve this, we use a trick that
    #   `f(x) - stop(f(x)) == zeros_like(f(x))`
    # but its gradient is grad[f(x)].
    # Note that IEEE754 specifies that `x - x == 0.` and `x + 0. == x`, hence
    # this trick loses no precision. For more discussion regarding the relevant
    # portions of the IEEE754 standard, see the StackOverflow question,
    # "Is there a floating point value of x, for which x-x == 0 is false?"
    # http://stackoverflow.com/q/2686644
    # Following is same as adding zeros_like(dot_prod).
    f_log_avg_u = f_log_avg_u + dotprod - stop(dotprod)
    return tf.reduce_mean(f_log_avg_u, axis=0)  # Avg over batches.
