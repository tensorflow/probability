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

"""Tests for Spherical Spaces."""


from absl.testing import parameterized
import numpy as np
import scipy.special as sp_special
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import exp
from tensorflow_probability.python.bijectors import scale
from tensorflow_probability.python.bijectors import scale_matvec_tril
from tensorflow_probability.python.experimental.tangent_spaces import spaces_test_util
from tensorflow_probability.python.experimental.tangent_spaces import spherical
from tensorflow_probability.python.internal import test_util


def _hypersphere_log_volume(dims):
  return (np.log(2.) + (dims + 1) / 2. * np.log(np.pi) -
          sp_special.gammaln((dims + 1) / 2.))


class _SpheresTest(spaces_test_util.SpacesTest):

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
    return spherical.SphericalSpace()

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


del _SpheresTest


if __name__ == '__main__':
  test_util.main()
