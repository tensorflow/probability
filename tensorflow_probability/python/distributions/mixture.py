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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import categorical
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import seed_stream
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensorshape_util


class Mixture(distribution.Distribution):
  """Mixture distribution.

  The `Mixture` object implements batched mixture distributions.
  The mixture model is defined by a `Categorical` distribution (the mixture)
  and a python list of `Distribution` objects.

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
  x = tf.linspace(-2., 3., int(1e4)).eval()
  plt.plot(x, bimix_gauss.prob(x).eval());
  ```

  """

  def __init__(self,
               cat,
               components,
               validate_args=False,
               allow_nan_stats=True,
               use_static_graph=False,
               name="Mixture"):
    """Initialize a Mixture distribution.

    A `Mixture` is defined by a `Categorical` (`cat`, representing the
    mixture probabilities) and a list of `Distribution` objects
    all having matching dtype, batch shape, event shape, and continuity
    properties (the components).

    The `num_classes` of `cat` must be possible to infer at graph construction
    time and match `len(components)`.

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
      use_static_graph: Calls to `sample` will not rely on dynamic tensor
        indexing, allowing for some static graph compilation optimizations, but
        at the expense of sampling all underlying distributions in the mixture.
        (Possibly useful when running on TPUs).
        Default value: `False` (i.e., use dynamic indexing).
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
    if not isinstance(cat, categorical.Categorical):
      raise TypeError("cat must be a Categorical distribution, but saw: %s" %
                      cat)
    if not components:
      raise ValueError("components must be a non-empty list or tuple")
    if not isinstance(components, (list, tuple)):
      raise TypeError("components must be a list or tuple, but saw: %s" %
                      components)
    if not all(isinstance(c, distribution.Distribution) for c in components):
      raise TypeError(
          "all entries in components must be Distribution instances"
          " but saw: %s" % components)

    dtype = components[0].dtype
    if not all(d.dtype == dtype for d in components):
      raise TypeError("All components must have the same dtype, but saw "
                      "dtypes: %s" % [(d.name, d.dtype) for d in components])
    static_event_shape = components[0].event_shape
    static_batch_shape = cat.batch_shape
    for di, d in enumerate(components):
      if not tensorshape_util.is_compatible_with(static_batch_shape,
                                                 d.batch_shape):
        raise ValueError(
            "components[{}] batch shape must be compatible with cat "
            "shape and other component batch shapes".format(di))
      static_event_shape = tensorshape_util.merge_with(
          static_event_shape, d.event_shape)
      static_batch_shape = tensorshape_util.merge_with(
          static_batch_shape, d.batch_shape)
    if tensorshape_util.rank(static_event_shape) is None:
      raise ValueError(
          "Expected to know rank(event_shape) from components, but "
          "none of the components provide a static number of ndims")

    # Ensure that all batch and event ndims are consistent.
    with tf.name_scope(name) as name:
      num_components = cat._num_categories()
      static_num_components = tf.get_static_value(num_components)
      if static_num_components is None:
        raise ValueError(
            "Could not infer number of classes from cat and unable "
            "to compare this value to the number of components passed in.")
      # Possibly convert from numpy 0-D array.
      static_num_components = int(static_num_components)
      if static_num_components != len(components):
        raise ValueError("cat.num_classes != len(components): %d vs. %d" %
                         (static_num_components, len(components)))

      cat_batch_shape = cat.batch_shape_tensor()
      cat_batch_rank = tf.size(cat_batch_shape)
      if validate_args:
        batch_shapes = [d.batch_shape_tensor() for d in components]
        batch_ranks = [tf.size(bs) for bs in batch_shapes]
        check_message = ("components[%d] batch shape must match cat "
                         "batch shape")
        self._assertions = [
            assert_util.assert_equal(
                cat_batch_rank, batch_ranks[di], message=check_message % di)
            for di in range(len(components))
        ]
        self._assertions += [
            assert_util.assert_equal(
                cat_batch_shape, batch_shapes[di], message=check_message % di)
            for di in range(len(components))
        ]
      else:
        self._assertions = []

      self._cat = cat
      self._components = list(components)
      self._num_components = static_num_components
      self._static_event_shape = static_event_shape
      self._static_batch_shape = static_batch_shape

      self._use_static_graph = use_static_graph
      if use_static_graph and static_num_components is None:
        raise ValueError("Number of categories must be known statically when "
                         "`static_sample=True`.")
    # We let the Mixture distribution access _graph_parents since its arguably
    # more like a baseclass.
    graph_parents = self._cat._graph_parents  # pylint: disable=protected-access
    for c in self._components:
      graph_parents += c._graph_parents  # pylint: disable=protected-access

    super(Mixture, self).__init__(
        dtype=dtype,
        reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=graph_parents,
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

  def _batch_shape_tensor(self):
    return self._cat.batch_shape_tensor()

  def _batch_shape(self):
    return self._static_batch_shape

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
    with tf.control_dependencies(self._assertions):
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
    with tf.control_dependencies(self._assertions):
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
    with tf.control_dependencies(self._assertions):
      x = tf.convert_to_tensor(x, name="x")
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
    with tf.control_dependencies(self._assertions):
      x = tf.convert_to_tensor(x, name="x")
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
    if self._use_static_graph:
      with tf.control_dependencies(self._assertions):
        # This sampling approach is almost the same as the approach used by
        # `MixtureSameFamily`. The differences are due to having a list of
        # `Distribution` objects rather than a single object, and maintaining
        # random seed management that is consistent with the non-static code
        # path.
        samples = []
        cat_samples = self.cat.sample(n, seed=seed)
        stream = seed_stream.SeedStream(seed, salt="Mixture")

        for c in range(self.num_components):
          samples.append(self.components[c].sample(n, seed=stream()))
        stack_axis = -1 - tensorshape_util.rank(self._static_event_shape)
        x = tf.stack(samples, axis=stack_axis)  # [n, B, k, E]
        npdt = dtype_util.as_numpy_dtype(x.dtype)
        mask = tf.one_hot(
            indices=cat_samples,  # [n, B]
            depth=self._num_components,  # == k
            on_value=npdt(1),
            off_value=npdt(0))  # [n, B, k]
        mask = distribution_util.pad_mixture_dimensions(
            mask, self, self._cat,
            tensorshape_util.rank(self._static_event_shape))  # [n, B, k, [1]*e]
        return tf.reduce_sum(x * mask, axis=stack_axis)  # [n, B, E]

    with tf.control_dependencies(self._assertions):
      n = tf.convert_to_tensor(n, name="n")
      static_n = tf.get_static_value(n)
      n = int(static_n) if static_n is not None else n
      cat_samples = self.cat.sample(n, seed=seed)

      static_samples_shape = cat_samples.shape
      if tensorshape_util.is_fully_defined(static_samples_shape):
        samples_shape = tensorshape_util.as_list(static_samples_shape)
        samples_size = tensorshape_util.num_elements(static_samples_shape)
      else:
        samples_shape = tf.shape(cat_samples)
        samples_size = tf.size(cat_samples)
      static_batch_shape = self.batch_shape
      if tensorshape_util.is_fully_defined(static_batch_shape):
        batch_shape = tensorshape_util.as_list(static_batch_shape)
        batch_size = tensorshape_util.num_elements(static_batch_shape)
      else:
        batch_shape = self.batch_shape_tensor()
        batch_size = tf.reduce_prod(batch_shape)
      static_event_shape = self.event_shape
      if tensorshape_util.is_fully_defined(static_event_shape):
        event_shape = np.array(
            tensorshape_util.as_list(static_event_shape), dtype=np.int32)
      else:
        event_shape = self.event_shape_tensor()

      # Get indices into the raw cat sampling tensor. We will
      # need these to stitch sample values back out after sampling
      # within the component partitions.
      samples_raw_indices = tf.reshape(tf.range(0, samples_size), samples_shape)

      # Partition the raw indices so that we can use
      # dynamic_stitch later to reconstruct the samples from the
      # known partitions.
      partitioned_samples_indices = tf.dynamic_partition(
          data=samples_raw_indices,
          partitions=cat_samples,
          num_partitions=self.num_components)

      # Copy the batch indices n times, as we will need to know
      # these to pull out the appropriate rows within the
      # component partitions.
      batch_raw_indices = tf.reshape(
          tf.tile(tf.range(0, batch_size), [n]), samples_shape)

      # Explanation of the dynamic partitioning below:
      #   batch indices are i.e., [0, 1, 0, 1, 0, 1]
      # Suppose partitions are:
      #     [1 1 0 0 1 1]
      # After partitioning, batch indices are cut as:
      #     [batch_indices[x] for x in 2, 3]
      #     [batch_indices[x] for x in 0, 1, 4, 5]
      # i.e.
      #     [1 1] and [0 0 0 0]
      # Now we sample n=2 from part 0 and n=4 from part 1.
      # For part 0 we want samples from batch entries 1, 1 (samples 0, 1),
      # and for part 1 we want samples from batch entries 0, 0, 0, 0
      #   (samples 0, 1, 2, 3).
      partitioned_batch_indices = tf.dynamic_partition(
          data=batch_raw_indices,
          partitions=cat_samples,
          num_partitions=self.num_components)
      samples_class = [None for _ in range(self.num_components)]

      stream = seed_stream.SeedStream(seed, salt="Mixture")

      for c in range(self.num_components):
        n_class = tf.size(partitioned_samples_indices[c])
        samples_class_c = self.components[c].sample(
            n_class, seed=stream())

        # Pull out the correct batch entries from each index.
        # To do this, we may have to flatten the batch shape.

        # For sample s, batch element b of component c, we get the
        # partitioned batch indices from
        # partitioned_batch_indices[c]; and shift each element by
        # the sample index. The final lookup can be thought of as
        # a matrix gather along locations (s, b) in
        # samples_class_c where the n_class rows correspond to
        # samples within this component and the batch_size columns
        # correspond to batch elements within the component.
        #
        # Thus the lookup index is
        #   lookup[c, i] = batch_size * s[i] + b[c, i]
        # for i = 0 ... n_class[c] - 1.
        lookup_partitioned_batch_indices = (
            batch_size * tf.range(n_class) + partitioned_batch_indices[c])
        samples_class_c = tf.reshape(
            samples_class_c, tf.concat([[n_class * batch_size], event_shape],
                                       0))
        samples_class_c = tf.gather(
            samples_class_c,
            lookup_partitioned_batch_indices,
            name="samples_class_c_gather")
        samples_class[c] = samples_class_c

      # Stitch back together the samples across the components.
      lhs_flat_ret = tf.dynamic_stitch(
          indices=partitioned_samples_indices, data=samples_class)
      # Reshape back to proper sample, batch, and event shape.
      ret = tf.reshape(
          lhs_flat_ret, tf.concat(
              [samples_shape, self.event_shape_tensor()], 0))
      tensorshape_util.set_shape(
          ret,
          tensorshape_util.concatenate(static_samples_shape, self.event_shape))
      return ret

  def entropy_lower_bound(self, name="entropy_lower_bound"):
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
      with tf.control_dependencies(self._assertions):
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
