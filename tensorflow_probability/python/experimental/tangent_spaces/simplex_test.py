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

"""Tests for Simplex Spaces."""


from absl.testing import parameterized
import numpy as np
import scipy.special as sp_special
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import exp
from tensorflow_probability.python.bijectors import identity
from tensorflow_probability.python.bijectors import scale
from tensorflow_probability.python.bijectors import scale_matvec_tril
from tensorflow_probability.python.experimental.tangent_spaces import simplex
from tensorflow_probability.python.experimental.tangent_spaces import spaces_test_util
from tensorflow_probability.python.internal import test_util


class _SimplexTest(spaces_test_util.SpacesTest):

  def generate_coords(self):
    # Ensure that this represents the positive orthant of the sphere.
    logits = []
    for _ in range(self.dims):
      logits.append(tf.range(-7., 7., self.delta))
    return tf.reshape(
        tf.stack(tf.meshgrid(*logits), axis=-1), [-1, self.dims])

  def embed_coords(self, coords):
    # Use logit embedding.
    logits = tf.concat([coords, tf.zeros_like(coords[..., -1:])], axis=-1)
    return tf.nn.softmax(logits)

  def tangent_space(self, x):
    del x
    return simplex.ProbabilitySimplexSpace()

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


del _SimplexTest


if __name__ == '__main__':
  test_util.main()
