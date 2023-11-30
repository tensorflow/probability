# Copyright 2019 The TensorFlow Probability Authors.
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

# pylint: disable=abstract-method,no-member,g-importing-member

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import bernoulli as bernoulli_lib
from tensorflow_probability.python.distributions import categorical as categorical_lib
from tensorflow_probability.python.distributions import joint_distribution as jd_lib
from tensorflow_probability.python.distributions import joint_distribution_coroutine as jdc_lib
from tensorflow_probability.python.distributions import sample as sample_lib
from tensorflow_probability.python.experimental.marginalize.logeinsumexp import logeinsumexp
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers


__all__ = [
    'MarginalizableJointDistributionCoroutine',
]


# TODO(b/144095516) remove this function when `tf.einsum` supports
# broadcasting along size-1 dimensions.
def _squeezed_einsum(einsum_fn, formula, *args):
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
  result = einsum_fn(new_formula, *new_args)
  return result


def _cartesian_product(*supports):
  """Construct "cartesian product" of tensors.

  Args:
    *supports: a sequence of tensors `s1, ..., sn`.

  Returns:
    This function computes a tensor analogous to the cartesian
    product of sets.
    If `t = _cartesian_product(s1, ..., sn)` then
    `t[i1, ..., in] = s1[i1] s2[i2] ... sn[in]`
    where the elements on the right hand side are concatenated
    together.

    In particular, if `s1, ..., sn` are the supports of `n`
    distributions, the cartesian product represents the support of the
    product distribution.

    For example if `a = [0, 1]`, b = [10, 20]` and
    `c = _cartesian_product(a, b)` then
    `c = [[[0, 10], [0, 20]], [[1, 10], [1, 20]]]`.
    In this case note (for example) that
    `a[0] = 0`, `b[1] = 20` and so `c[0, 1] = [0, 20]`.
  """

  return tf.stack(tf.meshgrid(*supports, indexing='ij'), axis=-1)


def _power(support, n):
  """Construct n-fold cartesian product of a tensor with itself."""
  return _cartesian_product(*(n * [tf.expand_dims(support, -1)]))


def _support(dist):
  """Compute support of a discrete distribution.

  Currently supports `Bernoulli`, `Categorical` and `Sample`.

  Args:
    dist: a `tfd.Distribution` instance.

  Returns:
    pair consisting of support of distribution and the rank of
    the underlying event type.
  """

  if isinstance(dist, bernoulli_lib.Bernoulli):
    return tf.range(2), 0
  elif isinstance(dist, categorical_lib.Categorical):
    return tf.range(tf.shape(dist.probs_parameter())[-1]), 0
  elif isinstance(dist, sample_lib.Sample):
    # The support of `tfd.Sample` is the n-fold cartesian product
    # of the supports of the underlying distributions where
    # `n` is the total size of the sample shape.

    sample_shape, n = dist._expand_sample_shape_to_vector(  # pylint: disable=protected-access
        dist.sample_shape, 'expand_sample_shape')
    p, rank = _support(dist.distribution)
    product = _power(p, n)
    new_shape = ps.concat([ps.shape(product)[:-1], sample_shape], axis=-1)

    new_rank = rank + tf.compat.dimension_value(sample_shape.shape[0])
    return tf.reshape(product, new_shape), new_rank
  else:
    raise ValueError('Unable to find support for distribution ' +
                     str(dist))


def _expand_right(a, n, pos):
  """Insert multiple dimensions of size 1 at position `pos` in tensor's shape.

  Equivalent to performing `expand_dims(..., pos)` `n` times.

  Args:
    a: tensor into which extra dimensions will be inserted.
    n: number of inserted dimensions.
    pos: choice of dimension for insertion. Must be negative.

  Returns:
    Tensor with inserted dimensions.
  """

  axis = ps.rank(a) + pos + 1
  return tf.reshape(a, ps.concat([
      ps.shape(a)[:axis],
      ps.ones([n], dtype=tf.int32),
      ps.shape(a)[axis:]], axis=0))


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
                            method='logeinsumexp', internal_type=None):
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
      method: Specifies method by which marginalization is carried out.
        'einsum': use `einsum`. For very small probabilities this may
        result in underflow causing log probabilities of `-inf` even when
        the result shoukd be finite.
        'logeinsumexp': performs an `einsum` designed to work in log space.
        Although it preserves precision better than the 'einsum' method it
        can be slow and use more memory.
      internal_type: because the `einsum` method can cause underflow this
        argument allows the user to specify the type in which the
        `einsum` is computed. For example `tf.float64`.

    Returns:
      Return log probability density/mass of `value` for this distribution.

    Notes:
      Currently only a single log probability can be computed, so lists or
      tensors containing multiple samples from the joint distribution are
      not supported.
      The number of latent (i.e. marginalized or tabulated) variables is
      limited to 52 in this version.
      The individual samples in `tfd.Sample` are mathematically independent
      but the marginalization algorithm used is unable to exploit this fact,
      meaning that computation time can easily grow exponentially with
      `sample_shape`.
    """
    new_values = []
    indices = []
    formula = []
    table_rhs = []

    # Number of independent variables created so far
    num_variables = 0

    with tf.name_scope(name):
      ds = self._call_execute_model(
          sample_and_trace_fn=jd_lib.trace_distributions_only,
          # Only used for tracing so can be fixed.
          seed=samplers.zeros_seed())

      # Both 'marginalize' and 'tabulate' indicate that
      # instead of using samples provided by the user, this method
      # instead provides a tensor containing the entire support of
      # the distribution. The tensors are constructed so that the
      # support for each distribution is indexed by a different axis.
      # At the end, the probabilities are computed using `tf.einsum`
      # and the unique axes means that each tabulated or marginalized
      # variable corresponds to one symbol used in the `einsum`.
      for value, dist in zip(values, ds):
        if value == 'marginalize':
          supp, rank = _support(dist)
          r = ps.rank(supp)
          num_new_variables = r - rank
          # We can think of supp as being a tensor containing tensors,
          # each of which is a draw from the distribution.
          # `rank` is the rank of samples from the distribution.
          # `num_new_variables` is the rank of the containing tensor.
          # When we marginalize over a variable we want the sum
          # over the containing tensor.
          # So `num_new_variables` is the number of new indices needed.
          # We use `expand_right` to ensure that each of these
          # new indices is unique and independent of previous
          # supports.
          new_values.append(_expand_right(supp, n=num_variables, pos=-1 - rank))
          num_variables += num_new_variables
          indices = np.arange(num_variables - 1, -1, -1)
          # By *not* placing indices on the right of the '->' in
          # the einsum below we ensure that this variable is
          # used for reduction, in effect marginalizing over
          # that variable.
          formula.append(indices)
        elif value == 'tabulate':
          supp, rank = _support(dist)
          r = ps.rank(supp)
          if r is None:
            raise ValueError('Need to be able to statically find rank of'
                             'support of random variable: {}'.format(str(dist)))
          num_new_variables = r - rank
          new_values.append(_expand_right(supp, n=num_variables, pos=-1 - rank))
          num_variables += num_new_variables
          indices = np.arange(num_variables - 1, -1, -1)
          # The first elements of `indices` are the newly
          # introduced variables.
          new_indices = indices[: num_new_variables]
          # By placing indices on the right of the '->' in
          # the einsum below we ensure that this variable isn't
          # reduced over and that instead it is tabulated
          # in the resulting tensor.
          table_rhs.extend(new_indices)
          formula.append(indices)
        else:
          new_values.append(_expand_right(value, num_variables, -1))
          indices = range(num_variables - 1, -1, -1)
          formula.append(indices)
      formula = [''.join(map(_letter, f)) for f in formula]
      formula_rhs = ''.join(map(_letter, table_rhs))
      formula = '{}->{}'.format(','.join(formula), formula_rhs)

      # There is no `logsumexp_einsum`.
      # So use higher precision if user requests it.
      lpp = self.log_prob_parts(new_values)
      if internal_type:
        lpp = [tf.cast(x, dtype=internal_type) for x in lpp]
      if method == 'logeinsumexp':
        return _squeezed_einsum(logeinsumexp, formula, *lpp)
      elif method == 'einsum':
        return tf.math.log(
            _squeezed_einsum(tf.einsum, formula, *map(tf.exp, lpp)))
      else:
        raise ValueError(
            'Unknown `marginalized_log_prob` method: \'{}\'.'.format(method))


class MarginalizableJointDistributionCoroutine(
    jdc_lib.JointDistributionCoroutine, Marginalizable):

  pass
