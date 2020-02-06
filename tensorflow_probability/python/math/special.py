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

"""Implements special functions in TensorFlow.

The Lambert W function is the inverse of `z = u * exp(u)`, i. e., it is the
function that satisfies `u = W(z) * exp(W(z))`.  The solution cannot
be expressed as elementary functions and is thus part of the *special*
functions in mathematics.  See https://en.wikipedia.org/wiki/Lambert_W_function.

In general it is a complex-values function with multiple branches. The `k=0`
branch is knowns as the *principal branch* of the Lambert W function and is
implemented here. See also `scipy.special.lambertw`.

# References

Corless, R.M., Gonnet, G.H., Hare, D.E.G. et al. On the LambertW function.
Adv Comput Math 5, 329-359 (1996) doi:10.1007/BF02124750
"""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import dtype_util


__all__ = [
    "lambertw",
    "lambertw_winitzki_approx",
]


def lambertw_winitzki_approx(z, name=None):
  """Computes Winitzki approximation to Lambert W function at z >= -1/exp(1).

  The approximation for z >= -1/exp(1) will be used as a starting point in the
  iterative algorithm to compute W(z). See _lambertw_principal_branch() below.
  See
  https://www.researchgate.net/post/Is_there_approximation_to_the_LambertWx_function
  and in particular (38) in
  https://pdfs.semanticscholar.org/e934/24f33e2742016ef18c36a80788400d2f17b4.pdf

  Args:
    z: value for which W(z) should be computed. Expected z >= -1/exp(1). If not
     then function will fail due to log(<0).
    name: optionally pass name for output.

  Returns:
    lambertw_winitzki_approx: Approximation for W(z) for z >= -1/exp(1).
  """
  with tf.name_scope(name or "lambertw_winitzki_approx"):
    z = tf.convert_to_tensor(z)
    # See eq (38) here:
    # https://pdfs.semanticscholar.org/e934/24f33e2742016ef18c36a80788400d2f17b4.pdf
    # or (10) here:
    # https://hal.archives-ouvertes.fr/hal-01586546/document
    log1pz = tf.math.log1p(z)
    return log1pz * (1. - tf.math.log1p(log1pz) / (2. + log1pz))


def _fritsch_iteration(unused_should_stop, z, w, tol):
  """Root finding iteration for W(z) using Fritsch iteration."""
  # See Section 2.3 in https://arxiv.org/pdf/1209.0735.pdf
  # Approximate W(z) by viewing iterative algorithm as multiplicative factor
  #
  #  W(n+1) = W(n) * (1 + error)
  #
  # where error can be expressed as a function of z and W(n). See paper for
  # details.
  z = tf.convert_to_tensor(z)
  w = tf.convert_to_tensor(w)
  zn = tf.math.log(tf.abs(z)) - tf.math.log(tf.abs(w)) - w
  wp1 = w + 1.0
  q = 2. * wp1 * (wp1 + 2. / 3. * zn)
  q_minus_2zn = q - 2. * zn
  error = zn / wp1 * (1. + zn / q_minus_2zn)
  # Check absolute tolerance (not relative).  Here the iteration error is
  # for relative tolerance, as W(n+1) = W(n) * (1 + error).  Use
  # W(n+1) - W(n) = W(n) * error to get absolute tolerance.
  converged = abs(error * w) <= tol
  should_stop_next = tf.reduce_all(converged)
  return should_stop_next, w * (1. + error), z, tol


def _newton_iteration(unused_should_stop, w, z, tol):
  """Newton iteration on root finding of w for the equation w * exp(w) = z."""
  w = tf.convert_to_tensor(w)
  z = tf.convert_to_tensor(z)
  delta = (w - z * tf.exp(-w)) / (1. + w)
  converged = tf.abs(delta) <= tol
  should_stop_next = tf.reduce_all(converged)
  return should_stop_next, w - delta, z, tol


def _lambertw_principal_branch(z, name=None):
  """Computes Lambert W of `z` element-wise at the principal (k = 0) branch.

  The Lambert W function is the inverse of `z = y * tf.exp(y)` and is a
  many-valued function. Here `y = W_0(z)`, where `W_0` is the Lambert W function
  evaluated at the 0-th branch (aka principal branch).

  Args:
    z: A Tensor with type `float32` or `float64`.
    name: A name for the operation (optional).
      Default value: `None` (i.e., 'lambertw_principal_branch').

  Returns:
    lambertw_principal_branch: A Tensor with same shape and same dtype as `z`.
  """
  with tf.name_scope(name or "lambertw_principal_branch"):
    z = tf.convert_to_tensor(z)
    np_finfo = np.finfo(dtype_util.as_numpy_dtype(z.dtype))
    tolerance = tf.convert_to_tensor(2. * np_finfo.resolution, dtype=z.dtype)
    # Start while loop with the initial value at the approximate Lambert W
    # solution, instead of 'z' (for z > -1 / exp(1)).  Using 'z' has bad
    # convergence properties especially for large z (z > 5).
    z0 = tf.where(z > -np.exp(-1.), lambertw_winitzki_approx(z), z)
    z0 = tf.while_loop(cond=lambda stop, *_: ~stop,
                       body=_newton_iteration,
                       loop_vars=(False, z0, z, tolerance))[1]
    return tf.cast(z0, dtype=z.dtype)


@tf.custom_gradient
def lambertw(z, name=None):
  """Computes Lambert W of `z` element-wise.

  The Lambert W function is the inverse of `z = w * tf.exp(w)`.  Lambert W is a
  complex-valued function, but here it returns only the real part of the image
  of the function. It also only returns the principal branch (also known as
  branch `0`).

  Args:
    z: A Tensor with type `float32` or `float64`.
    name: A name for the operation (optional).

  Returns:
    lambertw: The Lambert W function evaluated at `z`. A Tensor with same shape
      and same dtype as `z`.
  """
  with tf.name_scope(name or "lambertw"):
    z = tf.convert_to_tensor(z)
    wz = _lambertw_principal_branch(z, name)

    def grad(dy):
      """Computes the derivative of Lambert W of `z` element-wise.

      The first derivative W'(z) can be computed from W(z) as it holds

        W'(z) = W(z) / (z * (1 + W(z)))

      Args:
        dy: A Tensor with type `float32` or `float64`.

      Returns:
        A Tensor with same shape and dtype as `z`.
      """
      # At z = 0 the analytic expressions for the gradient results in a 0/0
      # expression.  However, the continuous expansion (l'Hospital rule) gives a
      # derivative of 1.0 at z = 0.  This case has to be handled separately with
      # a where clause.
      grad_wz = (dy * tf.where(tf.equal(z, 0.0),
                               tf.ones_like(wz),
                               wz / (z * (1. + wz))))
      return grad_wz

    return wz, grad
