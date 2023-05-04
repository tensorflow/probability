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

"""Tests for Symmetric Matrix Spaces."""


from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import exp
from tensorflow_probability.python.bijectors import reshape
from tensorflow_probability.python.bijectors import scale
from tensorflow_probability.python.experimental.tangent_spaces import spaces_test_util
from tensorflow_probability.python.experimental.tangent_spaces import symmetric_matrix
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import linalg


class _SymmetricMatrixTest(spaces_test_util.SpacesTest):

  def generate_coords(self):
    coords = []
    for _ in range(self.dims):
      coords.append(tf.range(0., 1., self.delta))
    return tf.reshape(
        tf.stack(tf.meshgrid(*coords), axis=-1), [-1, self.dims])

  def embed_coords(self, coords):
    # Fill up a triangular matrix and then copy it to the other side.
    matrices = linalg.fill_triangular(coords)
    matrices = tf.linalg.matrix_transpose(matrices) + matrices
    matrices = tf.linalg.set_diag(matrices, tf.linalg.diag_part(matrices) / 2.)
    return matrices

  def tangent_space(self, x):
    del x
    return symmetric_matrix.SymmetricMatrixSpace()

  def log_local_area(self, local_grads):
    n = int((-1 + np.sqrt(1 + 8 * self.dims)) / 2)
    local_grads_shape = tf.shape(local_grads)
    local_grads = tf.reshape(
        local_grads,
        tf.concat([tf.shape(local_grads)[:-1], [n, n]], axis=0))
    # sqrt weight off diagonal terms so they appear with half the weight in the
    # jacobian matrix squared.
    local_grads = local_grads / np.sqrt(2.)
    local_grads = tf.linalg.set_diag(
        local_grads, tf.linalg.diag_part(local_grads) * np.sqrt(2.))
    local_grads = tf.reshape(local_grads, local_grads_shape)
    log_volume = 0.5 * tf.linalg.logdet(
        tf.linalg.matmul(local_grads, local_grads, transpose_b=True))
    return self.dims * np.log(self.delta) + log_volume

  def flatten_bijector(self):
    n = int((-1 + np.sqrt(1 + 8 * self.dims)) / 2)
    return reshape.Reshape(event_shape_in=[n, n], event_shape_out=[n**2])

  def log_volume(self):
    return 0.


class TwoDSymmetricMatrix(_SymmetricMatrixTest):
  delta = 5e-2
  dims = 3  # 3 independent coordinates

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Scale',
       'bijector_class': scale.Scale,
       'event_ndims': 2,
       # batch_shape: []
       'bijector_params': {
           'scale': [
               [1.2, 0.8],
               [0.8, 2.]]}
      },
      {'testcase_name': 'AsymmetricScale',
       'bijector_class': scale.Scale,
       'event_ndims': 2,
       # batch_shape: []
       'bijector_params': {
           'scale': [
               [1., 2.],
               [3., 2.]]}
      },
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 2,
       # batch_shape: []
       'bijector_params': {}
       },
  )
  def testSymmetric(self, bijector_class, event_ndims, bijector_params):
    self._testSpace(bijector_class, event_ndims, bijector_params)

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 2,
       'bijector_params': {}
       },
      {'testcase_name': 'Scaling',
       'bijector_class': scale.Scale,
       'event_ndims': 2,
       'bijector_params': {'scale': 2.}
       },
      {'testcase_name': 'ScalingMatrix',
       'bijector_class': scale.Scale,
       'event_ndims': 2,
       'bijector_params': {'scale': [[2., 1.], [1., 2.]]}
       },
      {'testcase_name': 'ScalingAsymmetricMatrix',
       'bijector_class': scale.Scale,
       'event_ndims': 2,
       'bijector_params': {'scale': [[2., 1.], [3., -2.]]}
       },

  )
  def testSpecializations(self, bijector_class, event_ndims, bijector_params):
    self._testSpecializations(bijector_class, event_ndims, bijector_params)


class ThreeDSymmetricMatrix(_SymmetricMatrixTest):
  delta = 2e-1
  dims = 6  # 6 independent coordinates

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Scale',
       'bijector_class': scale.Scale,
       'event_ndims': 2,
       # batch_shape: []
       'bijector_params': {
           'scale': [
               [1.2, 0.8, 3.],
               [0.8, 2., -1.],
               [3., -1., -1.]]}
      },
      {'testcase_name': 'AsymmetricScale',
       'bijector_class': scale.Scale,
       'event_ndims': 2,
       # batch_shape: []
       'bijector_params': {
           'scale': [
               [1., 2., 3.],
               [3., 2., 1.],
               [1., -3., -2.]]}
      },
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 2,
       # batch_shape: []
       'bijector_params': {}
       },
  )
  def testSymmetric(self, bijector_class, event_ndims, bijector_params):
    self._testSpace(bijector_class, event_ndims, bijector_params)

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 2,
       'bijector_params': {}
       },
      {'testcase_name': 'Scaling',
       'bijector_class': scale.Scale,
       'event_ndims': 2,
       'bijector_params': {'scale': 2.}
       },
      {'testcase_name': 'ScalingMatrix',
       'bijector_class': scale.Scale,
       'event_ndims': 2,
       'bijector_params': {
           'scale': [[2., 1., 0.5],
                     [1., 1., 1.],
                     [0.5, 1., -3.]]}
       },
  )
  def testSpecializations(self, bijector_class, event_ndims, bijector_params):
    self._testSpecializations(bijector_class, event_ndims, bijector_params)


class _ConstantDiagonalSymmetricMatrixTest(spaces_test_util.SpacesTest):

  def generate_coords(self):
    coords = []
    for _ in range(self.dims):
      coords.append(tf.range(0., 1., self.delta))
    return tf.reshape(
        tf.stack(tf.meshgrid(*coords), axis=-1), [-1, self.dims])

  def embed_coords(self, coords):
    # Fill up a triangular matrix, pad it with zeros and then copy it to the
    # other side. We'll assume the diagonal has zeros.
    n = int((-1 + np.sqrt(1 + 8 * self.dims)) / 2) + 1
    matrices = linalg.fill_triangular(coords)
    matrices_shape = tf.shape(matrices)
    row_shape = tf.concat([matrices_shape[:-2], [1, n - 1]], axis=0)
    matrices = tf.concat([
        tf.zeros(row_shape, dtype=coords.dtype),
        matrices], axis=-2)
    col_shape = tf.concat([matrices_shape[:-2], [n, 1]], axis=0)
    matrices = tf.concat([
        matrices, tf.zeros(col_shape, dtype=coords.dtype)], axis=-1)
    matrices = tf.linalg.matrix_transpose(matrices) + matrices
    return matrices

  def tangent_space(self, x):
    del x
    return symmetric_matrix.ConstantDiagonalSymmetricMatrixSpace()

  def log_local_area(self, local_grads):
    n = int((-1 + np.sqrt(1 + 8 * self.dims)) / 2) + 1
    local_grads_shape = tf.shape(local_grads)
    local_grads = tf.reshape(
        local_grads,
        tf.concat([tf.shape(local_grads)[:-1], [n, n]], axis=0))
    # sqrt weight off diagonal terms so they appear with half the weight in the
    # jacobian matrix squared.
    local_grads = local_grads / np.sqrt(2.)
    local_grads = tf.linalg.set_diag(
        local_grads, tf.zeros_like(tf.linalg.diag_part(local_grads)))
    local_grads = tf.reshape(local_grads, local_grads_shape)
    log_volume = 0.5 * tf.linalg.logdet(
        tf.linalg.matmul(local_grads, local_grads, transpose_b=True))
    return self.dims * np.log(self.delta) + log_volume

  def flatten_bijector(self):
    n = int((-1 + np.sqrt(1 + 8 * self.dims)) / 2) + 1
    return reshape.Reshape(
        event_shape_in=[n, n], event_shape_out=[n**2])

  def log_volume(self):
    return 0.


class ConstantDiagonalTwoDSymmetricMatrix(_ConstantDiagonalSymmetricMatrixTest):
  delta = 1e-3
  dims = 1  # 1 independent coordinate for a 2x2 matrix.

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Scale',
       'bijector_class': scale.Scale,
       'event_ndims': 2,
       # batch_shape: []
       'bijector_params': {
           'scale': [
               [1.2, 0.8],
               [0.8, 2.]]}
      },
      {'testcase_name': 'AsymmetricScale',
       'bijector_class': scale.Scale,
       'event_ndims': 2,
       # batch_shape: []
       'bijector_params': {
           'scale': [
               [1., 2.],
               [3., 2.]]}
      },
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 2,
       # batch_shape: []
       'bijector_params': {}
       },
  )
  def testSymmetric(self, bijector_class, event_ndims, bijector_params):
    self._testSpace(bijector_class, event_ndims, bijector_params)

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 2,
       'bijector_params': {}
       },
      {'testcase_name': 'Scaling',
       'bijector_class': scale.Scale,
       'event_ndims': 2,
       'bijector_params': {'scale': 2.}
       },
      {'testcase_name': 'ScalingMatrix',
       'bijector_class': scale.Scale,
       'event_ndims': 2,
       'bijector_params': {'scale': [[2., 1.], [1., 2.]]}
       },
      {'testcase_name': 'ScalingAsymmetricMatrix',
       'bijector_class': scale.Scale,
       'event_ndims': 2,
       'bijector_params': {'scale': [[2., 1.], [3., -2.]]}
       },

  )
  def testSpecializations(self, bijector_class, event_ndims, bijector_params):
    self._testSpecializations(bijector_class, event_ndims, bijector_params)


class ConstantDiagonalThreeDSymmetricMatrix(
    _ConstantDiagonalSymmetricMatrixTest):
  delta = 2e-1
  dims = 3  # 3 independent coordinate for a 2x2 matrix.

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Scale',
       'bijector_class': scale.Scale,
       'event_ndims': 2,
       # batch_shape: []
       'bijector_params': {
           'scale': [
               [1.2, 0.8, 3.],
               [0.8, 2., -1.],
               [3., -1., -1.]]}
      },
      {'testcase_name': 'AsymmetricScale',
       'bijector_class': scale.Scale,
       'event_ndims': 2,
       # batch_shape: []
       'bijector_params': {
           'scale': [
               [1., 2., 3.],
               [3., 2., 1.],
               [1., -3., -2.]]}
      },
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 2,
       # batch_shape: []
       'bijector_params': {}
       },
  )
  def testSymmetric(self, bijector_class, event_ndims, bijector_params):
    self._testSpace(bijector_class, event_ndims, bijector_params)

  @test_util.numpy_disable_gradient_test
  @parameterized.named_parameters(
      {'testcase_name': 'Exp',
       'bijector_class': exp.Exp,
       'event_ndims': 2,
       'bijector_params': {}
       },
      {'testcase_name': 'Scaling',
       'bijector_class': scale.Scale,
       'event_ndims': 2,
       'bijector_params': {'scale': 2.}
       },
      {'testcase_name': 'ScalingMatrix',
       'bijector_class': scale.Scale,
       'event_ndims': 2,
       'bijector_params': {
           'scale': [[2., 1., 0.5],
                     [1., 1., 1.],
                     [0.5, 1., -3.]]}
       },
  )
  def testSpecializations(self, bijector_class, event_ndims, bijector_params):
    self._testSpecializations(bijector_class, event_ndims, bijector_params)


del _ConstantDiagonalSymmetricMatrixTest
del _SymmetricMatrixTest


if __name__ == '__main__':
  test_util.main()
