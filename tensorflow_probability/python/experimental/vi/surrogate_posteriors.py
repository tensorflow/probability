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
"""Utilities for constructing surrogate posteriors."""

from __future__ import absolute_import
from __future__ import division
# [internal] enable type annotations
from __future__ import print_function

import functools

import tensorflow.compat.v2 as tf

from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python import util as tfp_util
from tensorflow_probability.python.bijectors import softplus as softplus_lib
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static

from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import


def build_trainable_location_scale_distribution(initial_loc,
                                                initial_scale,
                                                event_ndims,
                                                distribution_fn=tfd.Normal,
                                                validate_args=False,
                                                name=None):
  """Builds a variational distribution from a location-scale family.

  Args:
    initial_loc: Float `Tensor` initial location.
    initial_scale: Float `Tensor` initial scale.
    event_ndims: Integer `Tensor` number of event dimensions in `initial_loc`.
    distribution_fn: Optional constructor for a `tfd.Distribution` instance
      in a location-scale family. This should have signature `dist =
      distribution_fn(loc, scale, validate_args)`.
      Default value: `tfd.Normal`.
    validate_args: Python `bool`. Whether to validate input with asserts. This
      imposes a runtime cost. If `validate_args` is `False`, and the inputs are
      invalid, correct behavior is not guaranteed.
      Default value: `False`.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` (i.e.,
        'build_trainable_location_scale_distribution').
  Returns:
    posterior_dist: A `tfd.Distribution` instance.
  """
  with tf.name_scope(name or 'build_trainable_location_scale_distribution'):
    dtype = dtype_util.common_dtype([initial_loc, initial_scale],
                                    dtype_hint=tf.float32)
    initial_loc = tf.convert_to_tensor(initial_loc, dtype=dtype)
    initial_scale = tf.convert_to_tensor(initial_scale, dtype=dtype)

    loc = tf.Variable(initial_value=initial_loc, name='loc')
    scale = tfp_util.TransformedVariable(
        initial_scale, softplus_lib.Softplus(), name='scale')
    posterior_dist = distribution_fn(loc=loc, scale=scale,
                                     validate_args=validate_args)

    # Ensure the distribution has the desired number of event dimensions.
    static_event_ndims = tf.get_static_value(event_ndims)
    if static_event_ndims is None or static_event_ndims > 0:
      posterior_dist = tfd.Independent(
          posterior_dist,
          reinterpreted_batch_ndims=event_ndims,
          validate_args=validate_args)

  return posterior_dist


def _get_event_shape_shallow_structure(event_shape):
  """Gets shallow structure, treating lists of ints at the leaves as atomic."""
  def _not_list_of_ints(s):
    if isinstance(s, list) or isinstance(s, tuple):
      return not all(isinstance(x, int) for x in s)
    return True

  return nest.get_traverse_shallow_structure(_not_list_of_ints, event_shape)


# Default constructors for `build_factored_surrogate_posterior`.
_sample_uniform_initial_loc = functools.partial(
    tf.random.uniform, minval=-2., maxval=2., dtype=tf.float32)
_build_trainable_normal_dist = functools.partial(
    build_trainable_location_scale_distribution, distribution_fn=tfd.Normal)


def build_factored_surrogate_posterior(
    event_shape=None,
    constraining_bijectors=None,
    initial_unconstrained_loc=_sample_uniform_initial_loc,
    initial_unconstrained_scale=1e-2,
    trainable_distribution_fn=_build_trainable_normal_dist,
    seed=None,
    validate_args=False,
    name=None):
  """Builds a joint variational posterior that factors over model variables.

  By default, this method creates an independent trainable Normal distribution
  for each variable, transformed using a bijector (if provided) to
  match the support of that variable. This makes extremely strong
  assumptions about the posterior: that it is approximately normal (or
  transformed normal), and that all model variables are independent.

  Args:
    event_shape: `Tensor` shape, or nested structure of `Tensor` shapes,
      specifying the event shape(s) of the posterior variables.
    constraining_bijectors: Optional `tfb.Bijector` instance, or nested
      structure of such instances, defining support(s) of the posterior
      variables. The structure must match that of `event_shape` and may
      contain `None` values. A posterior variable will
      be modeled as `tfd.TransformedDistribution(underlying_dist,
      constraining_bijector)` if a corresponding constraining bijector is
      specified, otherwise it is modeled as supported on the
      unconstrained real line.
    initial_unconstrained_loc: Optional Python `callable` with signature
      `tensor = initial_unconstrained_loc(shape, seed)` used to sample
      real-valued initializations for the unconstrained representation of each
      variable. May alternately be a nested structure of
      `Tensor`s, giving specific initial locations for each variable; these
      must have structure matching `event_shape` and shapes determined by the
      inverse image of `event_shape` under `constraining_bijectors`, which
      may optionally be prefixed with a common batch shape.
      Default value: `functools.partial(tf.random.uniform,
        minval=-2., maxval=2., dtype=tf.float32)`.
    initial_unconstrained_scale: Optional scalar float `Tensor` initial
      scale for the unconstrained distributions, or a nested structure of
      `Tensor` initial scales for each variable.
      Default value: `1e-2`.
    trainable_distribution_fn: Optional Python `callable` with signature
      `trainable_dist = trainable_distribution_fn(initial_loc, initial_scale,
      event_ndims, validate_args)`. This is called for each model variable to
      build the corresponding factor in the surrogate posterior. It is expected
      that the distribution returned is supported on unconstrained real values.
      Default value: `functools.partial(
        tfp.vi.experimental.build_trainable_location_scale_distribution,
        distribution_fn=tfd.Normal)`, i.e., a trainable Normal distribution.
    seed: Python integer to seed the random number generator. This is used
      only when `initial_loc` is not specified.
    validate_args: Python `bool`. Whether to validate input with asserts. This
      imposes a runtime cost. If `validate_args` is `False`, and the inputs are
      invalid, correct behavior is not guaranteed.
      Default value: `False`.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` (i.e., 'build_factored_surrogate_posterior').

  Returns:
    surrogate_posterior: A `tfd.Distribution` instance whose samples have
      shape and structure matching that of `event_shape` or `initial_loc`.

  ### Examples

  Consider a Gamma model with unknown parameters, expressed as a joint
  Distribution:

  ```python
  Root = tfd.JointDistributionCoroutine.Root
  def model_fn():
    concentration = yield Root(tfd.Exponential(1.))
    rate = yield Root(tfd.Exponential(1.))
    y = yield tfd.Sample(tfd.Gamma(concentration=concentration, rate=rate),
                         sample_shape=4)
  model = tfd.JointDistributionCoroutine(model_fn)
  ```

  Let's use variational inference to approximate the posterior over the
  data-generating parameters for some observed `y`. We'll build a
  surrogate posterior distribution by specifying the shapes of the latent
  `rate` and `concentration` parameters, and that both are constrained to
  be positive.

  ```python
  surrogate_posterior = tfp.vi.experimental.build_factored_surrogate_posterior(
    event_shape=model.event_shape_tensor()[:-1],  # Omit the observed `y`.
    constraining_bijectors=[tfb.Softplus(),   # Rate is positive.
                            tfb.Softplus()])  # Concentration is positive.
  ```

  This creates a trainable joint distribution, defined by variables in
  `surrogate_posterior.trainable_variables`. We use `fit_surrogate_posterior`
  to fit this distribution by minimizing a divergence to the true posterior.

  ```python
  y = [0.2, 0.5, 0.3, 0.7]
  losses = tfp.vi.fit_surrogate_posterior(
    lambda rate, concentration: model.log_prob([rate, concentration, y]),
    surrogate_posterior=surrogate_posterior,
    num_steps=100,
    optimizer=tf.optimizers.Adam(0.1),
    sample_size=10)

  # After optimization, samples from the surrogate will approximate
  # samples from the true posterior.
  samples = surrogate_posterior.sample(100)
  posterior_mean = [tf.reduce_mean(x) for x in samples]     # mean ~= [1.1, 2.1]
  posterior_std = [tf.math.reduce_std(x) for x in samples]  # std  ~= [0.3, 0.8]
  ```

  If we wanted to initialize the optimization at a specific location, we can
  specify one when we build the surrogate posterior. This function requires the
  initial location to be specified in *unconstrained* space; we do this by
  inverting the constraining bijectors (note this section also demonstrates the
  creation of a dict-structured model).

  ```python
  initial_loc = {'concentration': 0.4, 'rate': 0.2}
  constraining_bijectors={'concentration': tfb.Softplus(),   # Rate is positive.
                          'rate': tfb.Softplus()}   # Concentration is positive.
  initial_unconstrained_loc = tf.nest.map_fn(
    lambda b, x: b.inverse(x) if b is not None else x,
    constraining_bijectors, initial_loc)
  surrogate_posterior = tfp.vi.experimental.build_factored_surrogate_posterior(
    event_shape=tf.nest.map_fn(tf.shape, initial_loc),
    constraining_bijectors=constraining_bijectors,
    initial_unconstrained_loc=initial_unconstrained_state,
    initial_unconstrained_scale=1e-4)
  ```

  """

  with tf.name_scope(name or 'build_factored_surrogate_posterior'):
    seed = tfp_util.SeedStream(seed, salt='build_factored_surrogate_posterior')

    # Convert event shapes to Tensors.
    shallow_structure = _get_event_shape_shallow_structure(event_shape)
    event_shape = nest.map_structure_up_to(
        shallow_structure, lambda s: tf.convert_to_tensor(s, dtype=tf.int32),
        event_shape)
    flat_event_shapes = tf.nest.flatten(event_shape)

    # For simplicity, we'll work with flattened lists of state parts and
    # repack the structure at the end.
    if constraining_bijectors is not None:
      flat_bijectors = tf.nest.flatten(constraining_bijectors)
    else:
      flat_bijectors = [None for _ in flat_event_shapes]
    flat_unconstrained_event_shapes = [
        b.inverse_event_shape_tensor(s) if b is not None else s
        for s, b in zip(flat_event_shapes, flat_bijectors)]

    # Construct initial locations for the internal unconstrained dists.
    if callable(initial_unconstrained_loc):  # Sample random initialization.
      flat_unconstrained_locs = [initial_unconstrained_loc(
          shape=s, seed=seed()) for s in flat_unconstrained_event_shapes]
    else:  # Use provided initialization.
      flat_unconstrained_locs = nest.flatten_up_to(
          shallow_structure, initial_unconstrained_loc, check_types=False)

    if nest.is_nested(initial_unconstrained_scale):
      flat_unconstrained_scales = nest.flatten_up_to(
          shallow_structure, initial_unconstrained_scale, check_types=False)
    else:
      flat_unconstrained_scales = [
          initial_unconstrained_scale for _ in flat_unconstrained_locs]

    # Extract the rank of each event, so that we build distributions with the
    # correct event shapes.
    flat_unconstrained_event_ndims = [prefer_static.rank_from_shape(s)
                                      for s in flat_unconstrained_event_shapes]

    # Build the component surrogate posteriors.
    flat_component_dists = []
    for initial_loc, initial_scale, event_ndims, bijector in zip(
        flat_unconstrained_locs,
        flat_unconstrained_scales,
        flat_unconstrained_event_ndims,
        flat_bijectors):
      unconstrained_dist = trainable_distribution_fn(
          initial_loc=initial_loc, initial_scale=initial_scale,
          event_ndims=event_ndims, validate_args=validate_args)
      flat_component_dists.append(
          bijector(unconstrained_dist) if bijector is not None
          else unconstrained_dist)
    component_distributions = tf.nest.pack_sequence_as(
        event_shape, flat_component_dists)

    # Return a `Distribution` object whose events have the specified structure.
    if hasattr(component_distributions, 'sample'):    # Tensor-valued posterior.
      return component_distributions
    elif hasattr(component_distributions, 'keys'):  # Dict-valued posterior.
      return tfd.JointDistributionNamed(component_distributions,
                                        validate_args=validate_args,
                                        name=name)
    else:
      return tfd.JointDistributionSequential(component_distributions,
                                             validate_args=validate_args,
                                             name=name)
