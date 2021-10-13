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
"""Monte Carlo integration and helpers."""

import tensorflow.compat.v2 as tf

__all__ = [
    'expectation_importance_sampler',
]


def expectation_importance_sampler(f,
                                   log_p,
                                   sampling_dist_q,
                                   z=None,
                                   n=None,
                                   seed=None,
                                   name='expectation_importance_sampler'):
  r"""Monte Carlo estimate of \\(E_p[f(Z)] = E_q[f(Z) p(Z) / q(Z)]\\).

  With \\(p(z) := exp^{log_p(z)}\\), this `Op` returns

  \\(n^{-1} sum_{i=1}^n [ f(z_i) p(z_i) / q(z_i) ],  z_i ~ q,\\)
  \\(\approx E_q[ f(Z) p(Z) / q(Z) ]\\)
  \\(=       E_p[f(Z)]\\)

  This integral is done in log-space with max-subtraction to better handle the
  often extreme values that `f(z) p(z) / q(z)` can take on.

  User supplies either `Tensor` of samples `z`, or number of samples to draw `n`

  Args:
    f: Callable mapping samples from `sampling_dist_q` to `Tensors` with shape
      broadcastable to `q.batch_shape`.
      For example, `f` works "just like" `q.log_prob`.
    log_p:  Callable mapping samples from `sampling_dist_q` to `Tensors` with
      shape broadcastable to `q.batch_shape`.
      For example, `log_p` works "just like" `sampling_dist_q.log_prob`.
    sampling_dist_q:  The sampling distribution.
      `tfp.distributions.Distribution`.
      `float64` `dtype` recommended.
      `log_p` and `q` should be supported on the same set.
    z:  `Tensor` of samples from `q`, produced by `q.sample` for some `n`.
    n:  Integer `Tensor`.  Number of samples to generate if `z` is not provided.
    seed:  Python integer to seed the random number generator.
    name:  A name to give this `Op`.

  Returns:
    The importance sampling estimate.  `Tensor` with `shape` equal
      to batch shape of `q`, and `dtype` = `q.dtype`.
  """
  q = sampling_dist_q
  with tf.name_scope(name):
    z = _get_samples(q, z, n, seed)

    log_p_z = log_p(z)
    q_log_prob_z = q.log_prob(z)

    def _importance_sampler_positive_f(log_f_z):
      # Same as expectation_importance_sampler_logspace, but using Tensors
      # rather than samples and functions.  Allows us to sample once.
      log_values = log_f_z + log_p_z - q_log_prob_z
      return _logspace_mean(log_values)

    # With \\(f_{plus}(z) = max(0, f(z)), f_{minus}(z) = max(0, -f(z))\\),
    # \\(E_p[f(Z)] = E_p[f_{plus}(Z)] - E_p[f_{minus}(Z)]\\)
    # \\(          = E_p[f_{plus}(Z) + 1] - E_p[f_{minus}(Z) + 1]\\)
    # Without incurring bias, 1 is added to each to prevent zeros in logspace.
    # The logarithm is approximately linear around 1 + epsilon, so this is good
    # for small values of 'z' as well.
    f_z = f(z)
    log_f_plus_z = tf.math.log1p(tf.nn.relu(f_z))
    log_f_minus_z = tf.math.log1p(tf.nn.relu(-1. * f_z))

    log_f_plus_integral = _importance_sampler_positive_f(log_f_plus_z)
    log_f_minus_integral = _importance_sampler_positive_f(log_f_minus_z)

  return tf.math.exp(log_f_plus_integral) - tf.math.exp(log_f_minus_integral)


def _logspace_mean(log_values):
  """Evaluate `Log[E[values]]` in a stable manner.

  Args:
    log_values:  `Tensor` holding `Log[values]`.

  Returns:
    `Tensor` of same `dtype` as `log_values`, reduced across dim 0.
      `Log[Mean[values]]`.
  """
  # center = Max[Log[values]],  with stop-gradient
  # The center hopefully keep the exponentiated term small.  It is canceled
  # from the final result, so putting stop gradient on it will not change the
  # final result.  We put stop gradient on to eliminate unnecessary computation.
  center = tf.stop_gradient(_sample_max(log_values))

  # centered_values = exp{Log[values] - E[Log[values]]}
  centered_values = tf.math.exp(log_values - center)

  # log_mean_of_values = Log[ E[centered_values] ] + center
  #                    = Log[ E[exp{log_values - E[log_values]}] ] + center
  #                    = Log[E[values]] - E[log_values] + center
  #                    = Log[E[values]]
  log_mean_of_values = tf.math.log(_sample_mean(centered_values)) + center

  return log_mean_of_values


def _sample_mean(values):
  """Mean over sample indices.  In this module this is always [0]."""
  return tf.reduce_mean(values, axis=[0])


def _sample_max(values):
  """Max over sample indices.  In this module this is always [0]."""
  return tf.reduce_max(values, axis=[0])


def _get_samples(dist, z, n, seed):
  """Check args and return samples."""
  with tf.name_scope('get_samples'):
    if (n is None) == (z is None):
      raise ValueError(
          'Must specify exactly one of arguments "n" and "z".  Found: '
          'n = %s, z = %s' % (n, z))
    if n is not None:
      return dist.sample(n, seed=seed)
    else:
      return tf.convert_to_tensor(z, name='z')
