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
"""Marginalizable probability distributions."""

from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

__all__ = [
    'MarginalizableJointDistributionCoroutine',
]


tfd = tfp.distributions


# TODO(b/144095516) remove this function when `tf.einsum` supports
# broadcasting along size-1 dimensions.
def _squeezed_einsum(formula, *args):
  """Modified einsum that squeezes size 1 dimensions from arguments."""

  lhs, rhs = formula.split('->')
  lhs = lhs.split(',')
  new_args = []
  new_formula = []

  for indices, arg in zip(lhs, args):
    axis = []
    new_indices = ''
    for i, index in enumerate(indices):
      if arg.shape[i] == 1:
        # This axis can be removed.
        axis.append(i)
      else:
        new_indices += index
    new_args.append(tf.squeeze(arg, axis=axis))
    new_formula.append(new_indices)

  new_formula = '{}->{}'.format(','.join(new_formula), rhs)
  result = tf.einsum(new_formula, *new_args)
  return result


def _support(dist):
  """Compute support of a discrete distribution.

  Currently supports Bernoulli and Categorical.

  Args:
    dist: a `tfd.Distribution` instance.
  """

  if isinstance(dist, tfd.Bernoulli):
    return tf.range(2)
  elif isinstance(dist, tfd.Categorical):
    return tf.range(tf.shape(dist.probs_parameter())[-1])
  else:
    raise ValueError('Unable to find support for distribution ' +
                     str(dist))


def _expand_right(a, n):
  """Insert multiple axes of size 1 at right end of tensor's shape.

  Equivalent to performing `expand_dims(..., -1)` `n` times.

  Args:
    a: tensor into which extra axes will be inserted.
    n: number of inserted axes.
  """

  return tf.reshape(a, tf.concat([
      tf.shape(a), tf.ones([n], dtype=tf.int32)], axis=0))


def _letter(i):
  alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
  if i < len(alphabet):
    return alphabet[i]
  else:
    raise ValueError('Too many latent variables in'
                     '`Marginalizable` distribution.')


class Marginalizable(object):
  """Marginazlization mixin.

  This enables a joint distribution class to compute marginalized
  log probabilities.
  """

  def marginalized_log_prob(self, values, name='marginalized_log_prob',
                            internal_type=None):
    """Log probability density/mass function.

    Args:
      values: Structure of `Tensor`-values corresponding to samples yielded
        by model. There are also two special values that can be provided
        instead of `Tensor` samples.
        'marginalize': treat this random variable as a latent variable
        that will be marginalized out by summing over the support of
        the variable.
        'tabulate': treat this as a latent variable whose posterior probability
        distribution is to be computed. The final result is a tensor
        of log probabilities with each tabulated variable corresponding to
        one axis of the tensor in the order they appear in the list
        of `values`.
      name: Python `str` prepended to names of ops created by this function.
      internal_type: because of the absence of an analog of
        einsum based on `logsumexp` it is easy to cause underflows at
        intermediate stages of the computation. This can be alleviated slightly
        by choosing `internal_type` to be a higher precision type such as
        `tf.float64`.

    Returns:
      Return log probability density/mass of `value` for this distribution.

    Notes:
      Currently only a single log probability can be computed, so lists or
      tensors containing multiple samples from the joint distribution are
      not supported.
      Currently only scalar distributions can be marginalized or tabulated.
      The performance of this operation is very sensitive to the reduction
      order chosen by `tf.einsum`. Incorrect ordering can result in
      orders of magnitude difference in the time taken to compute the result.
      This is work in progress and future versions may perform significantly
      faster.
      The number of latent (i.e. marginalized or tabulated) variables is
      limited to 52 in this version.
    """
    new_values = []
    indices = []
    formula = []
    table_rhs = []
    shift = 0

    with tf.name_scope(name):
      flat_values = self._model_flatten(values)

      ds = self._get_single_sample_distributions()

      for d in ds:
        d.event_shape.assert_has_rank(0)

      # Both 'marginalize' and 'tabulate' indicate that
      # instead of using samples provided by the user, this method
      # instead provides a tensor containing the entire support of
      # the distribution. The tensors are constructed so that the
      # support for each distribution is indexed by a different axis.
      # At the end, the probabilities are computed using `tf.einsum`
      # and the unique axes means that each tabulated or marginalized
      # variable corresponds to one symbol used in the `einsum`.
      for a, dist in zip(flat_values, ds):
        if a == 'marginalize':
          supp = _support(dist)
          new_values.append(_expand_right(supp, shift))
          indices = range(shift, -1, -1)
          shift += 1
          # By *not* placing an index on the right of the '->' in
          # the einsum below we ensure that this variable is
          # used for reduction, in effect marginalizing over
          # that variable.
          formula.append(indices)
        elif a == 'tabulate':
          supp = _support(dist)
          new_values.append(_expand_right(supp, shift))
          indices = range(shift, -1, -1)
          shift += 1
          # By placing an index on the right of the '->' in
          # the einsum below we ensure that this variable isn't
          # reduced over and that instead it is tabulated
          # in the resulting tensor.
          table_rhs.append(indices[0])
          formula.append(indices)
        else:
          new_values.append(_expand_right(a, shift))
          formula.append(indices)
      formula = [''.join(map(_letter, f)) for f in formula]
      formula_rhs = ''.join(map(_letter, table_rhs))
      formula = '{}->{}'.format(','.join(formula), formula_rhs)

      # There is no `logsumexp_einsum` yet.
      # TODO(b/144098450)
      # So use higher precision of user requests it.
      lpp = self.log_prob_parts(new_values)
      if internal_type:
        lpp = tf.cast(lpp, dtype=internal_type)
      return tf.math.log(_squeezed_einsum(formula, *map(tf.exp, lpp)))


class MarginalizableJointDistributionCoroutine(
    tfd.JointDistributionCoroutine, Marginalizable):
  pass
