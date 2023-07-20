# Copyright 2023 The TensorFlow Probability Authors.
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

"""Utilities for testing Tangent Spaces."""


import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import identity
from tensorflow_probability.python.experimental.tangent_spaces import spaces
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient


class SpacesTest(test_util.TestCase):
  """Base class for testing Tangent Spaces."""

  atol = 2e-3

  def _testSpace(
      self, bijector_class, event_ndims, bijector_params):
    # Ensure that the 'measure' is preserved under transformations.
    bijector = bijector_class(**bijector_params)

    @tf.function
    def transformed_log_prob(x):
      density_correction = bijector.experimental_compute_density_correction(
          x,
          self.tangent_space(x),
          backward_compat=True,
          event_ndims=event_ndims)[0]
      return -(self.log_volume() + density_correction)

    coords = self.generate_coords()
    z = self.embed_coords(coords)
    local_grads = []
    # Consider the batch shape here.
    bijector_batch_shape = tensorshape_util.as_list(
        bijector.experimental_batch_shape(x_event_ndims=event_ndims))
    # We need to compute gradients without reducing over batch members.
    if tensorshape_util.rank(bijector_batch_shape):
      batch_indices = np.indices(bijector_batch_shape)
      num_indices = batch_indices.shape[0]
      batch_indices = np.transpose(
          batch_indices,
          list(range(1, len(batch_indices.shape))) + [0])
      batch_indices = np.reshape(batch_indices, [-1, num_indices])
      batch_indices = [list(b) for b in batch_indices]
    else:
      batch_indices = [Ellipsis]

    num_outputs = self.flatten_bijector().forward(z).shape[-1]

    for b in batch_indices:
      grads_per_batch = []
      for i in range(num_outputs):
        grads_per_batch.append(
            gradient.value_and_gradient(
                lambda a: self.flatten_bijector().forward(  # pylint: disable=g-long-lambda
                    bijector[b].forward(self.embed_coords(a)))[..., i],  # pylint:disable=cell-var-from-loop
                coords)[1])
      grads_per_batch = tf.stack(grads_per_batch, axis=-1)
      local_grads.append(grads_per_batch)
    local_grads = tf.stack(local_grads, axis=0)
    log_local_element = self.log_local_area(local_grads)
    log_probs = transformed_log_prob(z)

    # Mask out NaN and Inf values that might happen due to numerically unstable
    # computations.
    # These are meant to represent the volume of a small hypercube transformed
    # via the bijector weighted by the new density.
    masked_elements = log_local_element + log_probs
    masked_elements = tf.where(
        tf.math.is_inf(masked_elements), -np.inf, masked_elements)
    masked_elements = tf.where(
        tf.math.is_nan(masked_elements), -np.inf, masked_elements)
    total_log_prob = self.evaluate(
        tf.reduce_logsumexp(masked_elements, axis=-1))
    # Make sure the total log prob is zero -> the probabilities sum to one under
    # the new parameterization.
    self.assertAllClose(
        np.zeros_like(total_log_prob),
        total_log_prob, atol=self.atol, rtol=1e-5)

  def _testSpecializations(self, bijector_class, event_ndims, bijector_params):
    coords = self.generate_coords()
    z = self.embed_coords(coords)
    bijector = bijector_class(**bijector_params)
    tangent_space = self.tangent_space(z)
    log_volume1, _ = tangent_space.transform_general(
        z, bijector, event_ndims=event_ndims)
    log_volume2, _ = tangent_space.transform_dimension_preserving(
        z, bijector, event_ndims=event_ndims)
    log_volume3, _ = tangent_space.transform_coordinatewise(
        z, bijector, event_ndims=event_ndims)

    (log_volume1, log_volume2, log_volume3) = (
        self.evaluate([log_volume1, log_volume2, log_volume3]))
    self.assertAllClose(log_volume1, log_volume2)
    self.assertAllClose(log_volume1, log_volume3)

  def flatten_bijector(self):
    """Set this if your event space is a non-vector."""
    return identity.Identity()

  def tangent_space(self, x):
    # Tangent Space at a point on the unit circle. Only one tangent vector.
    return spaces.GeneralSpace(self.tangent_basis(x))
