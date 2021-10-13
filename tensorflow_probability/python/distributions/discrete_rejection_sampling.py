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
"""Batched discrete rejection samplers."""

import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python import random as tfp_random
from tensorflow_probability.python.distributions import exponential
from tensorflow_probability.python.internal import batched_rejection_sampler as brs
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers

__all__ = [
    'log_concave_rejection_sampler',
]


def log_concave_rejection_sampler(
    mode,
    prob_fn,
    dtype,
    sample_shape=(),
    distribution_minimum=None,
    distribution_maximum=None,
    seed=None):
  """Utility for rejection sampling from log-concave discrete distributions.

  This utility constructs an easy-to-sample-from upper bound for a discrete
  univariate log-concave distribution (for discrete univariate distributions, a
  necessary and sufficient condition is p_k^2 >= p_{k-1} p_{k+1} for all k).
  The method requires that the mode of the distribution is known. While a better
  method can likely be derived for any given distribution, this method is
  general and easy to implement. The expected number of iterations is bounded by
  4+m, where m is the probability of the mode. For details, see [(Devroye,
  1979)][1].

  Args:
    mode: Tensor, the mode[s] of the [batch of] distribution[s].
    prob_fn: Python callable, counts -> prob(counts).
    dtype: DType of the generated samples.
    sample_shape: 0D or 1D `int32` `Tensor`. Shape of the generated samples.
    distribution_minimum: Tensor of type `dtype`. The minimum value
      taken by the distribution. The `prob` method will only be called on values
      greater than equal to the specified minimum. The shape must broadcast with
      the batch shape of the distribution. If unspecified, the domain is treated
      as unbounded below.
    distribution_maximum: Tensor of type `dtype`. The maximum value
      taken by the distribution. See `distribution_minimum` for details.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.

  Returns:
    samples: a `Tensor` with prepended dimensions `sample_shape`.

  #### References

  [1] Luc Devroye. A Simple Generator for Discrete Log-Concave
      Distributions. Computing, 1987.
  """
  mode = tf.broadcast_to(
      mode, ps.concat([sample_shape, ps.shape(mode)], axis=0))

  mode_height = prob_fn(mode)
  mode_shape = ps.shape(mode)

  top_width = 1. + mode_height / 2.  # w in ref [1].
  top_fraction = top_width / (1 + top_width)
  exponential_distribution = exponential.Exponential(
      rate=tf.ones([], dtype=dtype))  # E in ref [1].

  if distribution_minimum is None:
    distribution_minimum = tf.constant(-np.inf, dtype)
  if distribution_maximum is None:
    distribution_maximum = tf.constant(np.inf, dtype)

  def proposal(seed):
    """Proposal for log-concave rejection sampler."""
    (top_lobe_fractions_seed,
     exponential_samples_seed,
     top_selector_seed,
     rademacher_seed) = samplers.split_seed(seed, n=4)

    top_lobe_fractions = samplers.uniform(
        mode_shape, seed=top_lobe_fractions_seed, dtype=dtype)  # V in ref [1].
    top_offsets = top_lobe_fractions * top_width / mode_height

    exponential_samples = exponential_distribution.sample(
        mode_shape, seed=exponential_samples_seed)  # E in ref [1].
    exponential_height = (exponential_distribution.prob(exponential_samples) *
                          mode_height)
    exponential_offsets = (top_width + exponential_samples) / mode_height

    top_selector = samplers.uniform(
        mode_shape, seed=top_selector_seed, dtype=dtype)  # U in ref [1].
    on_top_mask = (top_selector <= top_fraction)

    unsigned_offsets = tf.where(on_top_mask, top_offsets, exponential_offsets)
    offsets = tf.round(
        tfp_random.rademacher(
            mode_shape, seed=rademacher_seed, dtype=dtype) *
        unsigned_offsets)

    potential_samples = mode + offsets
    envelope_height = tf.where(on_top_mask, mode_height, exponential_height)

    return potential_samples, envelope_height

  def target(values):
    # Check for out of bounds rather than in bounds to avoid accidentally
    # masking a `nan` value.
    out_of_bounds_mask = (
        (values < distribution_minimum) | (values > distribution_maximum))
    in_bounds_values = tf.where(
        out_of_bounds_mask, tf.constant(0., dtype=values.dtype), values)
    probs = prob_fn(in_bounds_values)
    return tf.where(out_of_bounds_mask, tf.zeros([], probs.dtype), probs)

  return tf.stop_gradient(
      brs.batched_rejection_sampler(
          proposal, target, seed, dtype=dtype)[0])  # Discard `num_iters`.
