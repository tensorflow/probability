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

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python import monte_carlo
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import nest_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal.reparameterization import FULLY_REPARAMETERIZED
from tensorflow_probability.python.stats.leave_one_out import log_soomean_exp


__all__ = [
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


def monte_carlo_variational_loss(target_log_prob_fn,
                                 surrogate_posterior,
                                 sample_size=1,
                                 importance_sample_size=1,
                                 discrepancy_fn=kl_reverse,
                                 use_reparameterization=None,
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
      `tfd.JointDistribution`). Crucially, the distribution's `log_prob` and
      (if reparameterizeable) `sample` methods must directly invoke all ops
      that generate gradients to the underlying variables. One way to ensure
      this is to use `tfp.util.TransformedVariable` and/or
      `tfp.util.DeferredTensor` to represent any parameters defined as
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
    use_reparameterization: Python `bool`. When `None` (the default),
      automatically set to:
      `surrogate_posterior.reparameterization_type ==
      tfd.FULLY_REPARAMETERIZED`. When `True` uses the standard Monte-Carlo
      average. When `False` uses the score-gradient trick. (See above for
      details.)  When `False`, consider using `csiszar_vimco`.
    seed: Python `int` seed for `surrogate_posterior.sample`.
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

  #### Tricks: Reparameterization and Score-Gradient

  When q is "reparameterized", i.e., a diffeomorphic transformation of a
  parameterless distribution (e.g.,
  `Normal(Y; m, s) <=> Y = sX + m, X ~ Normal(0,1)`), we can swap gradient and
  expectation, i.e.,
  `grad[Avg{ s_i : i=1...n }] = Avg{ grad[s_i] : i=1...n }` where `S_n=Avg{s_i}`
  and `s_i = f(x_i), x_i ~iid q(X)`.

  However, if q is not reparameterized, TensorFlow's gradient will be incorrect
  since the chain-rule stops at samples of unreparameterized distributions. In
  this circumstance using the Score-Gradient trick results in an unbiased
  gradient, i.e.,

  ```none
  grad[ E_q[f(X)] ]
  = grad[ int dx q(x) f(x) ]
  = int dx grad[ q(x) f(x) ]
  = int dx [ q'(x) f(x) + q(x) f'(x) ]
  = int dx q(x) [q'(x) / q(x) f(x) + f'(x) ]
  = int dx q(x) grad[ f(x) q(x) / stop_grad[q(x)] ]
  = E_q[ grad[ f(x) q(x) / stop_grad[q(x)] ] ]
  ```

  Unless `q.reparameterization_type != tfd.FULLY_REPARAMETERIZED` it is
  usually preferable to set `use_reparameterization = True`.

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
    reparameterization_types = tf.nest.flatten(
        surrogate_posterior.reparameterization_type)
    if use_reparameterization is None:
      use_reparameterization = all(
          reparameterization_type == FULLY_REPARAMETERIZED
          for reparameterization_type in reparameterization_types)
    elif (use_reparameterization and
          any(reparameterization_type != FULLY_REPARAMETERIZED
              for reparameterization_type in reparameterization_types)):
      # TODO(jvdillon): Consider only raising an exception if the gradient is
      # requested.
      raise ValueError(
          'Distribution `surrogate_posterior` must be reparameterized, i.e.,'
          'a diffeomorphic transformation of a parameterless distribution. '
          '(Otherwise this function has a biased gradient.)')
    if not callable(target_log_prob_fn):
      raise TypeError('`target_log_prob_fn` must be a Python `callable`'
                      'function.')

    if use_reparameterization:
      # Attempt to avoid bijector inverses by computing the surrogate log prob
      # during the forward sampling pass.
      q_samples, q_lp = surrogate_posterior.experimental_sample_and_log_prob(
          [sample_size * importance_sample_size], seed=seed)
    else:
      # Score fn objective requires explicit gradients of `log_prob`.
      q_samples = surrogate_posterior.sample(
          [sample_size * importance_sample_size], seed=seed)
      q_lp = None

    return monte_carlo.expectation(
        f=_make_importance_weighted_divergence_fn(
            target_log_prob_fn,
            surrogate_posterior=surrogate_posterior,
            discrepancy_fn=discrepancy_fn,
            precomputed_surrogate_log_prob=q_lp,
            importance_sample_size=importance_sample_size),
        samples=q_samples,
        # Log-prob is only used if use_reparameterization=False.
        log_prob=surrogate_posterior.log_prob,
        use_reparameterization=use_reparameterization)


def _make_importance_weighted_divergence_fn(
    target_log_prob_fn,
    surrogate_posterior,
    discrepancy_fn,
    precomputed_surrogate_log_prob=None,
    importance_sample_size=1):
  """Defines a function to compute an importance-weighted divergence."""

  def divergence_fn(q_samples):
    q_lp = precomputed_surrogate_log_prob
    if q_lp is None:
      q_lp = surrogate_posterior.log_prob(q_samples)

    target_log_prob = nest_util.call_fn(target_log_prob_fn, q_samples)
    log_weights = target_log_prob - q_lp
    if tf.get_static_value(importance_sample_size) == 1:
      # Bypass importance weighting.
      return discrepancy_fn(log_weights)

    # Explicitly break out `importance_sample_size` as a separate axis.
    log_weights = tf.reshape(
        log_weights,
        ps.concat([[-1, importance_sample_size],
                   ps.shape(log_weights)[1:]], axis=0))
    log_sum_weights = tf.reduce_logsumexp(log_weights, axis=1)
    log_avg_weights = log_sum_weights - tf.math.log(
        tf.cast(importance_sample_size, dtype=log_weights.dtype))
    return discrepancy_fn(log_avg_weights)

  return divergence_fn


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
    seed: Python `int` seed for `q.sample`.
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

    q_sample = q.sample(sample_shape=[num_draws, num_batch_draws], seed=seed)
    x = tf.nest.map_structure(stop, q_sample)
    logqx = q.log_prob(x)
    logu = nest_util.call_fn(p_log_prob, x) - logqx
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
