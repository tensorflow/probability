# Copyright 2021 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
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
"""Thermodynamic integrals, for e.g. estimation of normalizing constants."""

import collections

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import broadcast_util as bu
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.stats import sample_stats

__all__ = ['remc_thermodynamic_integrals']


def remc_thermodynamic_integrals(
    inverse_temperatures,
    potential_energy,
    iid_chain_ndims=0,
):
  """Estimate thermodynamic integrals using results of ReplicaExchangeMC.

  Write the density, when tempering with inverse temperature `b`, as
  `p_b(x) = exp(-b * U(x)) f(x) / Z_b`. Here `Z_b` is a normalizing constant,
  and `U(x)` is the potential energy. f(x) is the untempered part, if any.

  Let `E_b[U(X)]` be the expected potential energy when `X ~ p_b`. Then,
  `-1 * integral_c^d E_b[U(X)] db = log[Z_d / Z_c]`, the log normalizing
  constant ratio.

  Let `Var_b[U(X)] be the variance of potential energy when `X ~ p_b(x)`. Then,
  `integral_c^d Var_b[U(X)] db = E_d[U(X)] - E_c[U(X)]`, the cross entropy
  difference.

  Integration is done via the trapezoidal rule. Assume `E_b[U(X)]` and
  `Var_b[U(X)]` have bounded second derivatives, uniform in `b`. Then, the
  bias due to approximation of the integral by a summation is `O(1 / K^2)`.

  Suppose `U(X)`, `X ~ p_b` has bounded fourth moment, uniform in `b`. Suppose
  further that the swap acceptance rate between every adjacent pair is greater
  than `C_s > 0`.  If we have `N` effective samples from each of the `n_replica`
  replicas, then the standard error of the summation is
  `O(1 / Sqrt(n_replica * N))`.

  Args:
    inverse_temperatures: `Tensor` of shape `[n_replica, ...]`, used to temper
      `n_replica` replicas. Assumed to be decreasing with respect to the replica
      index.
    potential_energy: The `potential_energy` field of
      `ReplicaExchangeMCKernelResults`, shape `[n_samples, n_replica, ...]`.
      If the kth replica has density `p_k(x) = exp(-beta_k * U(x)) * f_k(x)`,
      then `potential_energy[k]` is `U(X)`, where `X ~ p_k`.
    iid_chain_ndims: Number of dimensions in `potential_energy`, to the
      right of the replica dimension, that index independent identically
      distributed chains. In particular, the temperature for these chains should
      be identical. The sample means will be computed over these dimensions.

  Returns:
    ReplicaExchangeMCThermodynamicIntegrals namedtuple.
  """
  dtype = dtype_util.common_dtype(
      [inverse_temperatures, potential_energy], dtype_hint=tf.float32)
  inverse_temperatures = tf.convert_to_tensor(inverse_temperatures, dtype=dtype)
  potential_energy = tf.convert_to_tensor(potential_energy, dtype=dtype)

  # mean is E[U(beta)].
  # Reduction is over samples and (possibly) independent chains.
  # Squeeze out the singleton left over from samples in axis=0.
  # Keepdims so we can broadcast with inverse_temperatures, which *may* have
  # additional batch dimensions.
  iid_axis = ps.concat([[0], ps.range(2, 2 + iid_chain_ndims)], axis=0)
  mean = tf.reduce_mean(potential_energy, axis=iid_axis, keepdims=True)[0]
  var = sample_stats.variance(
      potential_energy, sample_axis=iid_axis, keepdims=True)[0]

  # Integrate over the single temperature dimension.
  # dx[k] = beta_k - beta_{k+1} > 0.
  dx = bu.left_justified_expand_dims_like(
      inverse_temperatures[:-1] - inverse_temperatures[1:], mean)
  def _trapz(y):
    avg_y = 0.5 * (y[:-1] + y[1:])
    return tf.reduce_sum(avg_y * dx, axis=0)

  def _squeeze_chains(x):
    # Squeeze with a reshape, since squeeze can't use tensors.
    return tf.reshape(x, ps.shape(x)[iid_chain_ndims:])

  return ReplicaExchangeMCThermodynamicIntegrals(
      log_normalizing_constant_ratio=-_squeeze_chains(_trapz(mean)),
      cross_entropy_difference=_squeeze_chains(_trapz(var)),
  )


class ReplicaExchangeMCThermodynamicIntegrals(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        'ReplicaExchangeMCThermodynamicIntegrals',
        [
            'log_normalizing_constant_ratio',
            'cross_entropy_difference',
        ])):
  """Classic thermodynamic integrals. E.g. normalizing constants."""
  __slots__ = ()
