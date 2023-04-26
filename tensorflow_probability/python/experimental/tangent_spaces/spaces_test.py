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

"""Tests for Tangent Spaces."""

from absl.testing import parameterized
import numpy as np
import scipy.special as sp_special
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import exp
from tensorflow_probability.python.bijectors import identity
from tensorflow_probability.python.bijectors import reshape
from tensorflow_probability.python.bijectors import scale
from tensorflow_probability.python.bijectors import scale_matvec_tril
from tensorflow_probability.python.bijectors import transform_diagonal
from tensorflow_probability.python.experimental.tangent_spaces import spaces
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient


def _hypersphere_log_volume(dims):
  return (np.log(2.) + (dims + 1) / 2. * np.log(np.pi) -
          sp_special.gammaln((dims + 1) / 2.))


class _SpacesTest(test_util.TestCase):

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


class DiscreteZeroSpaceTest(_SpacesTest):

  def generate_coords(self):
    return tf.range(1., 11.)

  def embed_coords(self, coords):
    return tf.stack([coords, 2 * coords - 1], axis=-1)

  def log_local_area(self, local_grads):
    # Each coordinate is a 10th of the volume.
    return -np.log(10.) * tf.ones_like(local_grads[..., 0])

  def log_volume(self):
    # Total volume is 1.
    return 0.

  def tangent_space(self, x):
    return spaces.ZeroSpace()

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {}
       },
      {'testcase_name': 'Scale',
       'bijector_class': scale.Scale,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {'scale': [2., 20.]}
       },
  )
  def testZeroSpace(self, bijector_class, event_ndims, bijector_params):
    self._testSpace(bijector_class, event_ndims, bijector_params)


class CircleSpaceTest(_SpacesTest):
  """Test GeneralSpace works on a Circle."""

  delta = 3e-3

  def generate_coords(self):
    return tf.range(-np.pi, np.pi, self.delta)

  def embed_coords(self, coords):
    return tf.stack([tf.math.cos(coords), tf.math.sin(coords)], axis=-1)

  def tangent_basis(self, x):
    return spaces.DenseBasis(
        tf.stack([-x[..., 1], x[..., 0]], axis=-1)[tf.newaxis, ...])

  def log_local_area(self, local_grads):
    return np.log(self.delta) + 0.5 * tf.reduce_logsumexp(
        2 * tf.math.log(tf.math.abs(local_grads)), axis=-1)

  def log_volume(self):
    return np.log(2 * np.pi)

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {}
       },
      {'testcase_name': 'Scaling',
       'bijector_class': scale.Scale,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {'scale': [2., 20.]}
       },
      {'testcase_name': 'ScalingBatch',
       'bijector_class': scale.Scale,
       'event_ndims': 1,
       # batch_shape: [3, 1, 2]
       'bijector_params': {'scale': [[[2., 20.]], [[3., 5.]], [[1., -1.]]]}
       },
  )
  def testGeneralSpace(self, bijector_class, event_ndims, bijector_params):
    self._testSpace(bijector_class, event_ndims, bijector_params)

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {}
       },
      {'testcase_name': 'Scaling',
       'bijector_class': scale.Scale,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {'scale': [2., 20.]}
       },
      {'testcase_name': 'ScalingBatch',
       'bijector_class': scale.Scale,
       'event_ndims': 1,
       # batch_shape: [3, 1, 2]
       'bijector_params': {'scale': [[[2., 20.]], [[3., 5.]], [[1., -1.]]]}
       },
  )
  def testSpecializations(self, bijector_class, event_ndims, bijector_params):
    self._testSpecializations(bijector_class, event_ndims, bijector_params)


class HalfSphereSpaceTest(_SpacesTest):
  """Test GeneralSpace works on a Sphere."""

  delta = 3e-3

  def generate_coords(self):
    angle0 = tf.range(-np.pi, np.pi, self.delta)
    # Ensure that we are far away from zero so that the tangent vectors are well
    # defined.
    angle1 = tf.range(np.pi / 2., np.pi, self.delta)
    return tf.reshape(tf.stack(tf.meshgrid(angle0, angle1), axis=-1), [-1, 2])

  def embed_coords(self, coords):
    xy_angle = coords[..., 0]
    z_angle = coords[..., 1]
    return tf.stack(
        [tf.math.sin(z_angle) * tf.math.cos(xy_angle),
         tf.math.sin(z_angle) * tf.math.sin(xy_angle),
         tf.math.cos(z_angle)], axis=-1)

  def tangent_basis(self, x):
    # Hairy Ball Theorem kicks in so that this parameterization is not going to
    # be valid for the whole sphere. However, we are only computing this on the
    # half-sphere so we should have a well defined tangent space.
    return spaces.DenseBasis(tf.stack([
        tf.stack([-x[..., 1], x[..., 0], tf.zeros_like(x[..., 0])], axis=-1),
        tf.stack([tf.zeros_like(x[..., 1]), -x[..., 2], x[..., 1]], axis=-1)
    ], axis=0))

  def log_local_area(self, local_grads):
    grads_0, grads_1 = tf.unstack(local_grads, axis=-2)
    return 2 * np.log(self.delta) + tf.math.log(
        1e-6 + tf.linalg.norm(tf.linalg.cross(grads_0, grads_1), axis=-1))

  def log_volume(self):
    return np.log(2 * np.pi)

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {}
       },
      {'testcase_name': 'ScalingSphere',
       'bijector_class': scale.Scale,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {'scale': [2., 5., 0.5]}
       },
      {'testcase_name': 'ScalingSphereBatch',
       'bijector_class': scale.Scale,
       'event_ndims': 1,
       # batch_shape: [2, 1, 3]
       'bijector_params': {'scale': [[[2., 5., 0.5]], [[3., 5., 2.]]]}
       },
      {'testcase_name': 'AffineSphere',
       'bijector_class': scale_matvec_tril.ScaleMatvecTriL,
       'event_ndims': None,
       # batch_shape: []
       'bijector_params': {
           'scale_tril': [[3., 0., 0.], [-2., 2., 0.], [1., 1., 1.]]}
       },
  )
  def testGeneralSpace(self, bijector_class, event_ndims, bijector_params):
    self._testSpace(bijector_class, event_ndims, bijector_params)

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {}
       },
      {'testcase_name': 'Scaling',
       'bijector_class': scale.Scale,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {'scale': [2., 5., 0.5]}
       },
      {'testcase_name': 'ScalingBatch',
       'bijector_class': scale.Scale,
       'event_ndims': 1,
       # batch_shape: [2, 1, 3]
       'bijector_params': {'scale': [[[2., 5., 0.5]], [[3., 5., 2.]]]}
       },
  )
  def testSpecializations(self, bijector_class, event_ndims, bijector_params):
    self._testSpecializations(bijector_class, event_ndims, bijector_params)


class _SpheresTest(_SpacesTest):

  def generate_coords(self):
    # Generalized Spherical Coordinates
    angles = [tf.range(-np.pi, np.pi, self.delta)]
    for _ in range(self.dims - 1):
      angles.append(tf.range(0., np.pi, self.delta))
    return tf.reshape(
        tf.stack(tf.meshgrid(*angles), axis=-1), [-1, self.dims])

  def embed_coords(self, coords):
    # Use hyperspherical coordinates.
    result = []
    first_coord = 1.
    for i in range(self.dims):
      x = coords[..., i]
      result.append(first_coord * tf.math.cos(x))
      first_coord = first_coord * tf.math.sin(x)
    result.append(first_coord)
    # Ensure that the last coordinate is the 'z' coordinate.
    result = list(reversed(result))
    return tf.stack(result, axis=-1)

  def tangent_space(self, x):
    del x
    return spaces.SphericalSpace()

  def log_local_area(self, local_grads):
    jac_sq = tf.linalg.matmul(local_grads, local_grads, transpose_b=True)
    jac_sq = tf.linalg.set_diag(jac_sq, tf.linalg.diag_part(jac_sq) + 1e-5)
    log_volume = 0.5 * tf.linalg.logdet(jac_sq)
    return self.dims * np.log(self.delta) + log_volume

  def log_volume(self):
    return _hypersphere_log_volume(self.dims)


class OneSphereSpaceTest(_SpheresTest):
  delta = 3e-3
  dims = 1

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Scale',
       'bijector_class': scale.Scale,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {'scale': [2., 20.]}
       },
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {}
       },
      {'testcase_name': 'Affine',
       'bijector_class': scale_matvec_tril.ScaleMatvecTriL,
       'event_ndims': None,
       # batch_shape: [2, 1]
       'bijector_params': {
           'scale_tril': [[[[3., 0.], [-2., 5.]]], [[[1., 0.], [1., 1.]]]]}

       },
  )
  def testSphericalSpace(self, bijector_class, event_ndims, bijector_params):
    self._testSpace(bijector_class, event_ndims, bijector_params)

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Scale',
       'bijector_class': scale.Scale,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {'scale': [2., 20.]}
       },
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {}
       },
  )
  def testSpecializations(self, bijector_class, event_ndims, bijector_params):
    self._testSpecializations(bijector_class, event_ndims, bijector_params)


class TwoSphereSpaceTest(_SpheresTest):
  delta = 3e-3
  dims = 2

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Scale',
       'bijector_class': scale.Scale,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {'scale': [1.2, 0.8, 3.]}
       },
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {}
       },
  )
  def testSphericalSpace(self, bijector_class, event_ndims, bijector_params):
    self._testSpace(bijector_class, event_ndims, bijector_params)

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Scale',
       'bijector_class': scale.Scale,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {'scale': [1.2, 0.8, 3.]}
       },
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {}
       },
  )
  def testSpecializations(self, bijector_class, event_ndims, bijector_params):
    self._testSpecializations(bijector_class, event_ndims, bijector_params)


# Because we are gridding the space, we need to reduce the atol for
# higher dimensions due to not enough grid points.


class ThreeSphereSpaceTest(_SpheresTest):
  delta = 5e-2
  atol = 3.5e-3
  dims = 3

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Scale',
       'bijector_class': scale.Scale,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {'scale': [0.2, 1.8, 3., 2.]}
       },
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {}
       },
  )
  def testSphericalSpace(self, bijector_class, event_ndims, bijector_params):
    self._testSpace(bijector_class, event_ndims, bijector_params)

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Scale',
       'bijector_class': scale.Scale,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {'scale': [0.2, 1.8, 3., 2.]}
       },
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {}
       },
  )
  def testSpecializations(self, bijector_class, event_ndims, bijector_params):
    self._testSpecializations(bijector_class, event_ndims, bijector_params)


class FourSphereSpaceTest(_SpheresTest):
  delta = 1e-1
  atol = 2e-2
  dims = 4

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Scale',
       'bijector_class': scale.Scale,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {'scale': [0.2, 1.8, 3., 2., 5.]}
       },
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {}
       },
  )
  def testSphericalSpace(self, bijector_class, event_ndims, bijector_params):
    self._testSpace(bijector_class, event_ndims, bijector_params)

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Scale',
       'bijector_class': scale.Scale,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {'scale': [0.2, 1.8, 3., 2., 5.]}
       },
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {}
       },
  )
  def testSpecializations(self, bijector_class, event_ndims, bijector_params):
    self._testSpecializations(bijector_class, event_ndims, bijector_params)


class FiveSphereSpaceTest(_SpheresTest):
  delta = 2e-1
  atol = 2e-2
  dims = 5

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Scale',
       'bijector_class': scale.Scale,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {'scale': [0.2, 1.8, -2., 2., 1.7, 0.8]}
       },
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {}
       },
  )
  def testSphericalSpace(self, bijector_class, event_ndims, bijector_params):
    self._testSpace(bijector_class, event_ndims, bijector_params)

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Scale',
       'bijector_class': scale.Scale,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {'scale': [0.2, 1.8, -2., 2., 1.7, 0.8]}
       },
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {}
       },
  )
  def testSpecializations(self, bijector_class, event_ndims, bijector_params):
    self._testSpecializations(bijector_class, event_ndims, bijector_params)


class _SimplexTest(_SpacesTest):

  def generate_coords(self):
    # Ensure that this represents the positive orthant of the sphere.
    logits = []
    for _ in range(self.dims):
      logits.append(tf.range(-50., 50., self.delta))
    return tf.reshape(
        tf.stack(tf.meshgrid(*logits), axis=-1), [-1, self.dims])

  def embed_coords(self, coords):
    # Use logit embedding.
    logits = tf.concat([coords, tf.zeros_like(coords[..., -1:])], axis=-1)
    return tf.nn.softmax(logits)

  def tangent_space(self, x):
    del x
    return spaces.ProbabilitySimplexSpace()

  def log_local_area(self, local_grads):
    log_volume = 0.5 * tf.linalg.logdet(
        tf.linalg.matmul(local_grads, local_grads, transpose_b=True))
    # We need to account for the fact the measure on the probability simplex is
    # not the lebesgue measure, and differs by a constant.
    # For instance, the length of the line segment from (0, 1) to (1, 0) is
    # sqrt(2), but is 1 under the simplex measure.
    return (self.dims * np.log(self.delta) + log_volume -
            0.5 * np.log1p(self.dims))

  def log_volume(self):
    return -sp_special.gammaln(self.dims + 1)


class OneSimplexTest(_SimplexTest):
  delta = 1e-1
  dims = 1

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Affine',
       'bijector_class': scale_matvec_tril.ScaleMatvecTriL,
       'event_ndims': None,
       # batch_shape: [2, 1]
       'bijector_params': {
           'scale_tril': [[[[3., 0.], [-2., 5.]]], [[[1., 0.], [1., 1.]]]]}

       },
      {'testcase_name': 'Scale',
       'bijector_class': scale.Scale,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {'scale': [2., 20.]}
       },
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {}
       },
      {'testcase_name': 'Identity',
       'bijector_class': identity.Identity,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {}
       },
  )
  def testSimplex(self, bijector_class, event_ndims, bijector_params):
    self._testSpace(bijector_class, event_ndims, bijector_params)

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Scale',
       'bijector_class': scale.Scale,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {'scale': [2., 20.]}
       },
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {}
       },
      {'testcase_name': 'Identity',
       'bijector_class': identity.Identity,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {}
       },
  )
  def testSpecializations(self, bijector_class, event_ndims, bijector_params):
    self._testSpecializations(bijector_class, event_ndims, bijector_params)


class TwoSimplexTest(_SimplexTest):
  delta = 1e-1
  atol = 7e-3
  dims = 2

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Affine',
       'bijector_class': scale_matvec_tril.ScaleMatvecTriL,
       'event_ndims': None,
       # batch_shape: [2, 1]
       'bijector_params': {
           'scale_tril': [[[[3., 0., 0.], [-2., 5., 0.], [1., 1., 1.]]],
                          [[[1., 0., 0.], [1., 1., 0.], [2., 1., -1.]]]]}

       },
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {}
       },
      {'testcase_name': 'Identity',
       'bijector_class': identity.Identity,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {}
       },
      {'testcase_name': 'Scale',
       'bijector_class': scale.Scale,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {'scale': [1.2, 0.8, 3.]}
       },
  )
  def testSimplex(self, bijector_class, event_ndims, bijector_params):
    self._testSpace(bijector_class, event_ndims, bijector_params)

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Scale',
       'bijector_class': scale.Scale,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {'scale': [1.2, 0.8, 3.]}
       },
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {}
       },
  )
  def testSpecializations(self, bijector_class, event_ndims, bijector_params):
    self._testSpecializations(bijector_class, event_ndims, bijector_params)


# Because we are gridding the space, we need to reduce the atol for
# higher dimensions due to not enough grid points.


class ThreeSimplexTest(_SimplexTest):
  delta = 0.8
  atol = 1.1e-2
  dims = 3

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Scale',
       'bijector_class': scale.Scale,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {'scale': [0.2, 1.8, 3., 2.]}
       },
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {}
       },
      {'testcase_name': 'Identity',
       'bijector_class': identity.Identity,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {}
       },
  )
  def testSimplex(self, bijector_class, event_ndims, bijector_params):
    self._testSpace(bijector_class, event_ndims, bijector_params)

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Scale',
       'bijector_class': scale.Scale,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {'scale': [0.2, 1.8, 3., 2.]}
       },
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {}
       },
      {'testcase_name': 'Identity',
       'bijector_class': identity.Identity,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {}
       },
  )
  def testSpecializations(self, bijector_class, event_ndims, bijector_params):
    self._testSpecializations(bijector_class, event_ndims, bijector_params)


class FourSimplexTest(_SimplexTest):
  delta = 2.
  atol = 4e-2
  dims = 4

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Scale',
       'bijector_class': scale.Scale,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {'scale': [0.2, 1.8, 3., 2., 5.]}
       },
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {}
       },
  )
  def testSimplex(self, bijector_class, event_ndims, bijector_params):
    self._testSpace(bijector_class, event_ndims, bijector_params)

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Scale',
       'bijector_class': scale.Scale,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {'scale': [0.2, 1.8, 3., 2., 5.]}
       },
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 1,
       # batch_shape: []
       'bijector_params': {}
       },
  )
  def testSpecializations(self, bijector_class, event_ndims, bijector_params):
    self._testSpecializations(bijector_class, event_ndims, bijector_params)


class SymmetricMatrixTest(_SpacesTest):
  """Test GeneralSpace works on a Symmetric Matrix."""

  delta = 1e-2

  def generate_coords(self):
    # 3D square
    x = tf.range(0., 1., delta=self.delta)
    y = tf.range(0., 1., delta=self.delta)
    z = tf.range(0., 1., delta=self.delta)
    return tf.reshape(tf.stack(tf.meshgrid(x, y, z), axis=-1), [-1, 3])

  def embed_coords(self, coords):
    x, y, z = tf.unstack(coords, axis=-1)
    first_row = tf.stack([x, y], axis=-1)
    second_row = tf.stack([y, z], axis=-1)
    result = tf.stack([first_row, second_row], axis=-2)
    return result

  def tangent_basis(self, x):
    # Tangent vectors are just the embedded unit vectors that result in
    # symmetric matrices.
    # This tests a constant bases, and that broadcasting works appropriately.
    return spaces.DenseBasis(np.array([
        [[1., 0.], [0., 0.]],
        [[0., 1.], [1., 0.]],
        [[0., 0.], [0., 1.]]]).astype(np.float32))

  def log_local_area(self, local_grads):
    # We can elide out the third entry of each vector since it's the same.
    first_local_grads = local_grads[..., 0:2]
    last_local_grads = local_grads[..., -1:]
    local_grads = tf.concat([first_local_grads, last_local_grads], axis=-1)
    grads_0, grads_1, grads_2 = tf.unstack(local_grads, axis=-2)

    # Volume of parallelpiped.
    volume = tf.reduce_sum(tf.linalg.cross(grads_0, grads_1) * grads_2, axis=-1)
    return 3 * np.log(self.delta) + tf.math.log(1e-6 + volume)

  def log_volume(self):
    return 0.

  def flatten_bijector(self):
    return reshape.Reshape(event_shape_in=[2, 2], event_shape_out=[4])

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'ExpSymmetric',
       'bijector_class': exp.Exp,
       'event_ndims': 2,
       # batch_shape: []
       'bijector_params': {}
       },
      {'testcase_name': 'ScalingSymmetric',
       'bijector_class': scale.Scale,
       'event_ndims': 2,
       # batch_shape: []
       'bijector_params': {'scale': [[2., 1.], [1., 0.5]]}
       },
  )
  def testGeneralSpace(self, bijector_class, event_ndims, bijector_params):
    self._testSpace(bijector_class, event_ndims, bijector_params)

  @test_util.numpy_disable_gradient_test
  def testTransformGeneralTransformDiagonal(self):
    tangent_basis = np.array([
        [[1., 0.], [0., 0.]],
        [[0., 1.], [1., 0.]],
        [[0., 0.], [0., 1.]]]).astype(np.float32)
    x = np.array([
        [[3., 4.], [4., 3.]],
        [[1., 0.], [0., 1.]],
        [[4., 2.], [2., 3.]],
        [[-1., 2.], [2., 1.]]]).astype(np.float32)

    bijector = transform_diagonal.TransformDiagonal(exp.Exp())
    gs = spaces.GeneralSpace(spaces.DenseBasis(tangent_basis))
    correction, new_gs = gs.transform_general(x, bijector)

    # Test that we get a correction only from the diagonal elements.
    diag_x = np.diagonal(x, axis1=-2, axis2=-1)
    self.assertAllClose(self.evaluate(correction), np.sum(diag_x, axis=-1))
    # Test that the new basis retains the middle element. This is because the
    # bijector only modifies the diagonal elements.
    new_basis = self.evaluate(new_gs.basis.to_dense())
    middle_element = np.array([[0., 1.], [1., 0.]]).astype(np.float32)
    middle_element = np.broadcast_to(middle_element, [4, 2, 2])
    self.assertAllClose(new_basis[1], middle_element)
    # Test the new basis is correct.
    expected_bases_elems = np.exp(np.eye(2) * x)
    expected_elem_0 = expected_bases_elems * np.array(
        [[1., 0.], [0., 0]]).astype(np.float32)
    expected_elem_2 = expected_bases_elems * np.array(
        [[0., 0.], [0., 1]]).astype(np.float32)
    self.assertAllClose(new_basis[0], expected_elem_0)
    self.assertAllClose(new_basis[2], expected_elem_2)

    # Finally check that the log_volume
    reshaped_new_basis = np.transpose(new_basis.reshape([3, 4, -1]), [1, 0, 2])
    expected_log_volume = 0.5 * np.linalg.slogdet(
        np.matmul(reshaped_new_basis,
                  np.transpose(reshaped_new_basis, axes=[0, 2, 1])))[1]
    self.assertAllClose(self.evaluate(new_gs.computed_log_volume),
                        expected_log_volume)


del _SimplexTest
del _SpacesTest
del _SpheresTest


if __name__ == '__main__':
  test_util.main()
