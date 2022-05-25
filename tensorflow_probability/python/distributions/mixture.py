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
"""The Mixture distribution class."""

import warnings

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import categorical
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.util.seed_stream import SeedStream
from tensorflow_probability.python.util.seed_stream import TENSOR_SEED_MSG_PREFIX


# Cause all warnings to always be triggered.
# Not having this means subsequent calls won't trigger the warning.
warnings.filterwarnings('always',
                        module='tensorflow_probability.*mixture',
                        append=True)  # Don't override user-set filters.


def _validate_cat_and_components(cat, components):
  """Basic checks on `cat` and `components` constructor args."""

  if not isinstance(cat, categorical.Categorical):
    raise TypeError('cat must be a Categorical distribution, but saw: %s' %
                    cat)
  if not components:
    raise ValueError('components must be a non-empty list or tuple')
  if not isinstance(components, (list, tuple)):
    raise TypeError('components must be a list or tuple, but saw: %s' %
                    components)
  if not all(isinstance(c, distribution.Distribution) for c in components):
    raise TypeError(
        'all entries in components must be Distribution instances'
        ' but saw: %s' % components)


class _Mixture(distribution.Distribution):
  """Mixture distribution.

  The `Mixture` object implements batched mixture distributions.
  The mixture model is defined by a `Categorical` distribution (the mixture)
  and a python list of `Distribution` objects.

  In the common case that the component distributions are all the same
  `Distribution` class (potentially with different parameters), it's probably
  better to use `tfp.distributions.MixtureSameFamily` instead.

  Methods supported include `log_prob`, `prob`, `mean`, `sample`, and
  `entropy_lower_bound`.


  #### Examples

  ```python
  # Create a mixture of two Gaussians:
  tfd = tfp.distributions
  mix = 0.3
  bimix_gauss = tfd.Mixture(
    cat=tfd.Categorical(probs=[mix, 1.-mix]),
    components=[
      tfd.Normal(loc=-1., scale=0.1),
      tfd.Normal(loc=+1., scale=0.5),
  ])

  # Plot the PDF.
  import matplotlib.pyplot as plt
  x = tf.linspace(-2., 3., int(1e4))
  plt.plot(x, bimix_gauss.prob(x))
  ```

  """

  def __init__(self,
               cat,
               components,
               validate_args=False,
               allow_nan_stats=True,
               name='Mixture'):
    """Initialize a Mixture distribution.

    A `Mixture` is defined by a `Categorical` (`cat`, representing the
    mixture probabilities) and a list of `Distribution` objects
    all having matching dtype, batch shape, event shape, support, and continuity
    properties (the components).

    The `num_classes` of `cat` must be possible to infer at graph construction
    time and match `len(components)`.

    In the common case that the component distributions are all the same
    `Distribution` class (potentially with different parameters), it's probably
    better to use `tfp.distributions.MixtureSameFamily` instead.

    Args:
      cat: A `Categorical` distribution instance, representing the probabilities
          of `distributions`.
      components: A list or tuple of `Distribution` instances.
        Each instance must have the same type, be defined on the same domain,
        and have matching `event_shape` and `batch_shape`.
      validate_args: Python `bool`, default `False`. If `True`, raise a runtime
        error if batch or event ranks are inconsistent between cat and any of
        the distributions. This is only checked if the ranks cannot be
        determined statically at graph construction time.
      allow_nan_stats: Boolean, default `True`. If `False`, raise an
       exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member. If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: A name for this distribution (optional).

    Raises:
      TypeError: If cat is not a `Categorical`, or `components` is not
        a list or tuple, or the elements of `components` are not
        instances of `Distribution`, or do not have matching `dtype`.
      ValueError: If `components` is an empty list or tuple, or its
        elements do not have a statically known event rank.
        If `cat.num_classes` cannot be inferred at graph creation time,
        or the constant value of `cat.num_classes` is not equal to
        `len(components)`, or all `components` and `cat` do not have
        matching static batch shapes, or all components do not
        have matching static event shapes.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:

      dtype = components[0].dtype
      if not all(d.dtype == dtype for d in components):
        raise TypeError('All components must have the same dtype, but saw '
                        'dtypes: %s' % [(d.name, d.dtype) for d in components])

      static_event_shape = components[0].event_shape
      static_batch_shape = cat.batch_shape
      for di, d in enumerate(components):
        if not tensorshape_util.is_compatible_with(static_batch_shape,
                                                   d.batch_shape):
          raise ValueError(
              'components[{}] batch shape must be compatible with cat '
              'shape and other component batch shapes ({} vs {})'.format(
                  di, static_batch_shape, d.batch_shape))
        if not tensorshape_util.is_compatible_with(static_event_shape,
                                                   d.event_shape):
          raise ValueError(
              'components[{}] event shape must be compatible with other '
              'component event shapes ({} vs {})'.format(
                  di, static_event_shape, d.event_shape))
        static_event_shape = tensorshape_util.merge_with(
            static_event_shape, d.event_shape)
        static_batch_shape = tensorshape_util.merge_with(
            static_batch_shape, d.batch_shape)
      if tensorshape_util.rank(static_event_shape) is None:
        raise ValueError(
            'Expected to know rank(event_shape) from components, but '
            'none of the components provide a static number of ndims')

      # pylint: disable=protected-access
      cat_dist_param = cat._probs if cat._logits is None else cat._logits
      # pylint: enable=protected-access
      static_num_components = tf.compat.dimension_value(
          cat_dist_param.shape[-1])
      if static_num_components is None:
        raise ValueError(
            'Could not infer number of classes from cat and unable '
            'to compare this value to the number of components passed in.')
      if static_num_components != len(components):
        raise ValueError('cat.num_classes != len(components): %d vs. %d' %
                         (static_num_components, len(components)))

      self._cat = cat
      self._components = list(components)
      self._num_components = static_num_components
      self._static_event_shape = static_event_shape
      self._static_batch_shape = static_batch_shape

      super(_Mixture, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @property
  def cat(self):
    return self._cat

  @property
  def components(self):
    return self._components

  @property
  def num_components(self):
    return self._num_components

  @property
  def experimental_is_sharded(self):
    sharded = self.cat.experimental_is_sharded
    if sharded != self.components.experimental_is_sharded:
      raise ValueError(
          '`Mixture.cat` sharding must match `Mixture.components`.')
    return sharded

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        cat=parameter_properties.BatchedComponentProperties(),
        components=parameter_properties.BatchedComponentProperties(
            event_ndims=lambda self: [0 for _ in self.components]))

  def _event_shape_tensor(self):
    return self._components[0].event_shape_tensor()

  def _event_shape(self):
    return self._static_event_shape

  def _expand_to_event_rank(self, x):
    """Expand the rank of x up to static_event_rank times for broadcasting.

    The static event rank was checked to not be None at construction time.

    Args:
      x: A tensor to expand.
    Returns:
      The expanded tensor.
    """
    expanded_x = x
    for _ in range(tensorshape_util.rank(self.event_shape)):
      expanded_x = tf.expand_dims(expanded_x, -1)
    return expanded_x

  def _mean(self):
    distribution_means = [d.mean() for d in self.components]
    cat_probs = self._cat_probs(log_probs=False)
    cat_probs = [self._expand_to_event_rank(c_p) for c_p in cat_probs]
    partial_means = [
        c_p * m for (c_p, m) in zip(cat_probs, distribution_means)
    ]
    # These should all be the same shape by virtue of matching
    # batch_shape and event_shape.
    return tf.add_n(partial_means)

  def _stddev(self):
    distribution_means = [d.mean() for d in self.components]
    distribution_devs = [d.stddev() for d in self.components]
    cat_probs = self._cat_probs(log_probs=False)

    stacked_means = tf.stack(distribution_means, axis=-1)
    stacked_devs = tf.stack(distribution_devs, axis=-1)
    cat_probs = [self._expand_to_event_rank(c_p) for c_p in cat_probs]
    broadcasted_cat_probs = (
        tf.stack(cat_probs, axis=-1) * tf.ones_like(stacked_means))

    batched_dev = distribution_util.mixture_stddev(
        tf.reshape(broadcasted_cat_probs, [-1, len(self.components)]),
        tf.reshape(stacked_means, [-1, len(self.components)]),
        tf.reshape(stacked_devs, [-1, len(self.components)]))

    # I.e. re-shape to list(batch_shape) + list(event_shape).
    return tf.reshape(batched_dev, tf.shape(broadcasted_cat_probs)[:-1])

  def _log_prob(self, x):
    x = tf.convert_to_tensor(x, name='x')
    distribution_log_probs = [d.log_prob(x) for d in self.components]
    cat_log_probs = self._cat_probs(log_probs=True)
    final_log_probs = [
        cat_lp + d_lp
        for (cat_lp, d_lp) in zip(cat_log_probs, distribution_log_probs)
    ]
    concat_log_probs = tf.stack(final_log_probs, 0)
    log_sum_exp = tf.reduce_logsumexp(concat_log_probs, axis=[0])
    return log_sum_exp

  def _log_cdf(self, x):
    x = tf.convert_to_tensor(x, name='x')
    distribution_log_cdfs = [d.log_cdf(x) for d in self.components]
    cat_log_probs = self._cat_probs(log_probs=True)
    final_log_cdfs = [
        cat_lp + d_lcdf
        for (cat_lp, d_lcdf) in zip(cat_log_probs, distribution_log_cdfs)
    ]
    concatted_log_cdfs = tf.stack(final_log_cdfs, axis=0)
    mixture_log_cdf = tf.reduce_logsumexp(concatted_log_cdfs, axis=[0])
    return mixture_log_cdf

  def _sample_n(self, n, seed=None):
    seeds = samplers.split_seed(seed, n=self.num_components + 1, salt='Mixture')
    try:
      seed_stream = SeedStream(seed, salt='Mixture')
    except TypeError as e:  # Can happen for Tensor seed.
      seed_stream = None
      seed_stream_err = e

    # This sampling approach is almost the same as the approach used by
    # `MixtureSameFamily`. The differences are due to having a list of
    # `Distribution` objects rather than a single object.
    samples = []
    cat_samples = self.cat.sample(n, seed=seeds[0])

    for c in range(self.num_components):
      try:
        samples.append(self.components[c].sample(n, seed=seeds[c + 1]))
        if seed_stream is not None:
          seed_stream()
      except TypeError as e:
        if ('Expected int for argument' not in str(e) and
            TENSOR_SEED_MSG_PREFIX not in str(e)):
          raise
        if seed_stream is None:
          raise seed_stream_err
        msg = (
            'Falling back to stateful sampling for `components[{}]` {} of '
            'type `{}`. Please update to use `tf.random.stateless_*` RNGs. '
            'This fallback may be removed after 20-Aug-2020. ({})')
        warnings.warn(msg.format(c, self.components[c].name,
                                 type(self.components[c]),
                                 str(e)))
        samples.append(self.components[c].sample(n, seed=seed_stream()))
    stack_axis = -1 - tensorshape_util.rank(self._static_event_shape)
    x = tf.stack(samples, axis=stack_axis)  # [n, B, k, E]
    # TODO(b/170730865): Is all this masking stuff really called for?
    npdt = dtype_util.as_numpy_dtype(x.dtype)
    mask = tf.one_hot(
        indices=cat_samples,  # [n, B]
        depth=self._num_components,  # == k
        on_value=npdt(1),
        off_value=npdt(0))  # [n, B, k]
    mask = distribution_util.pad_mixture_dimensions(
        mask, self, self._cat,
        tensorshape_util.rank(self._static_event_shape))  # [n, B, k, [1]*e]
    if x.dtype.is_floating:
      masked = tf.math.multiply_no_nan(x, mask)
    else:
      masked = x * mask
    return tf.reduce_sum(masked, axis=stack_axis)  # [n, B, E]

  def entropy_lower_bound(self, name='entropy_lower_bound'):
    r"""A lower bound on the entropy of this mixture model.

    The bound below is not always very tight, and its usefulness depends
    on the mixture probabilities and the components in use.

    A lower bound is useful for ELBO when the `Mixture` is the variational
    distribution:

    \\(
    \log p(x) >= ELBO = \int q(z) \log p(x, z) dz + H[q]
    \\)

    where \\( p \\) is the prior distribution, \\( q \\) is the variational,
    and \\( H[q] \\) is the entropy of \\( q \\). If there is a lower bound
    \\( G[q] \\) such that \\( H[q] \geq G[q] \\) then it can be used in
    place of \\( H[q] \\).

    For a mixture of distributions \\( q(Z) = \sum_i c_i q_i(Z) \\) with
    \\( \sum_i c_i = 1 \\), by the concavity of \\( f(x) = -x \log x \\), a
    simple lower bound is:

    \\(
    \begin{align}
    H[q] & = - \int q(z) \log q(z) dz \\\
       & = - \int (\sum_i c_i q_i(z)) \log(\sum_i c_i q_i(z)) dz \\\
       & \geq - \sum_i c_i \int q_i(z) \log q_i(z) dz \\\
       & = \sum_i c_i H[q_i]
    \end{align}
    \\)

    This is the term we calculate below for \\( G[q] \\).

    Args:
      name: A name for this operation (optional).

    Returns:
      A lower bound on the Mixture's entropy.
    """
    with self._name_and_control_scope(name):
      distribution_entropies = [d.entropy() for d in self.components]
      cat_probs = self._cat_probs(log_probs=False)
      partial_entropies = [
          c_p * m for (c_p, m) in zip(cat_probs, distribution_entropies)
      ]
      # These are all the same shape by virtue of matching batch_shape
      return tf.add_n(partial_entropies)

  def _cat_probs(self, log_probs):
    """Get a list of num_components batchwise probabilities."""
    if log_probs:
      x = tf.math.log_softmax(self.cat.logits_parameter())
    else:
      x = self.cat.probs_parameter()
    return tf.unstack(x, num=self.num_components, axis=-1)

  def _default_event_space_bijector(self):
    # TODO(b/146456627): Implement `default_event_space_bijector` for mixture
    # distributions.
    return

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []

    # Ensure that all batch shapes are consistent.
    #
    # NOTE: If the ranks/shapes are known statically, the assertions will be
    # evaluated statically, but the calls to `batch_shape_tensor()` here will
    # cause the assertions for `self._cat` and `self._components[i]` to be
    # checked.  In `__init__`, we could record whether or not all batch ranks/
    # shapes are known statically, in order to skip the checks here.
    cat_batch_shape = self._cat.batch_shape_tensor()
    cat_batch_rank = prefer_static.rank_from_shape(
        cat_batch_shape, self._cat.batch_shape)
    batch_shapes = [d.batch_shape_tensor() for d in self.components]
    batch_ranks = [prefer_static.rank_from_shape(batch_shapes[i],
                                                 self.components[i].batch_shape)
                   for i in range(len(self.components))]
    check_message = 'components[{}] batch shape must match cat batch shape'
    for i in range(len(self._components)):
      # NOTE: We have to compare the ranks, as well as the shapes, because
      # `assert_equal(shape1, shape2)` broadcasts and will return True if either
      # argument is `[]`.
      assertions.append(assert_util.assert_equal(
          batch_ranks[i], cat_batch_rank,
          message=check_message.format(i)))
      assertions.append(assert_util.assert_equal(
          batch_shapes[i], cat_batch_shape, message=check_message.format(i)))

    return assertions


class Mixture(_Mixture, distribution.AutoCompositeTensorDistribution):

  def __new__(cls, *args, **kwargs):
    """Maybe return a non-`CompositeTensor` `_Mixture`."""

    if cls is Mixture:
      if args:
        cat = args[0]
      else:
        cat = kwargs.get('cat')
      if len(args) > 1:
        components = args[1]
      else:
        components = kwargs.get('components')

      _validate_cat_and_components(cat, components)
      if not (isinstance(cat, tf.__internal__.CompositeTensor)
              and all(isinstance(d, tf.__internal__.CompositeTensor)
                      for d in components)):
        return _Mixture(*args, **kwargs)
    return super(Mixture, cls).__new__(cls)


Mixture.__doc__ = _Mixture.__doc__ + '\n' + (
    'If `cat` and all of `components` are `CompositeTensor`s, then the '
    'resulting `Mixture` instance is a `CompositeTensor` as well. Otherwise, a '
    'non-`CompositeTensor` `_Mixture` instance is created instead. '
    'Distribution subclasses that inherit from `Mixture` will also inherit '
    'from `CompositeTensor`.')
