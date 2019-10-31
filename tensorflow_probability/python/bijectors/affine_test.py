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
"""Affine Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

# Dependency imports
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class AffineBijectorTest(test_util.TestCase):
  """Tests correctness of the Y = scale @ x + shift transformation."""

  def testProperties(self):
    mu = -1.
    # scale corresponds to 1.
    bijector = tfb.Affine(shift=mu)
    self.assertStartsWith(bijector.name, "affine")

  def testNoBatchMultivariateIdentity(self):
    def static_run(fun, x, **kwargs):
      return self.evaluate(fun(x, **kwargs))

    def dynamic_run(fun, x_value, **kwargs):
      x_value = np.array(x_value, dtype=np.float32)
      placeholder = tf1.placeholder_with_default(x_value, shape=None)
      return self.evaluate(fun(placeholder, **kwargs))

    for run in (static_run, dynamic_run):
      mu = [1., -1]
      # Multivariate
      # Corresponds to scale = [[1., 0], [0, 1.]]
      bijector = tfb.Affine(shift=mu)
      x = [1., 1]
      # matmul(sigma, x) + shift
      # = [-1, -1] + [1, -1]
      self.assertAllClose([2., 0], run(bijector.forward, x))
      self.assertAllClose([0., 2], run(bijector.inverse, x))

      # x is a 2-batch of 2-vectors.
      # The first vector is [1, 1], the second is [-1, -1].
      # Each undergoes matmul(sigma, x) + shift.
      x = [[1., 1], [-1., -1]]
      self.assertAllClose([[2., 0], [0., -2]], run(bijector.forward, x))
      self.assertAllClose([[0., 2], [-2., 0]], run(bijector.inverse, x))
      self.assertAllClose(
          0., run(bijector.inverse_log_det_jacobian, x, event_ndims=1))

  def testNoBatchMultivariateDiag(self):
    def static_run(fun, x, **kwargs):
      return self.evaluate(fun(x, **kwargs))

    def dynamic_run(fun, x_value, **kwargs):
      x_value = np.array(x_value, dtype=np.float32)
      placeholder = tf1.placeholder_with_default(x_value, shape=None)
      return self.evaluate(fun(placeholder, **kwargs))

    for run in (static_run, dynamic_run):
      mu = [1., -1]
      # Multivariate
      # Corresponds to scale = [[2., 0], [0, 1.]]
      bijector = tfb.Affine(shift=mu, scale_diag=[2., 1])
      x = [1., 1]
      # matmul(sigma, x) + shift
      # = [-1, -1] + [1, -1]
      self.assertAllClose([3., 0], run(bijector.forward, x))
      self.assertAllClose([0., 2], run(bijector.inverse, x))
      self.assertAllClose(
          -np.log(2.),
          run(bijector.inverse_log_det_jacobian, x, event_ndims=1))

      # Reset bijector.
      bijector = tfb.Affine(shift=mu, scale_diag=[2., 1])
      # x is a 2-batch of 2-vectors.
      # The first vector is [1, 1], the second is [-1, -1].
      # Each undergoes matmul(sigma, x) + shift.
      x = [[1., 1],
           [-1., -1]]
      self.assertAllClose([[3., 0],
                           [-1., -2]],
                          run(bijector.forward, x))
      self.assertAllClose([[0., 2],
                           [-1., 0]],
                          run(bijector.inverse, x))
      self.assertAllClose(
          -np.log(2.),
          run(bijector.inverse_log_det_jacobian, x, event_ndims=1))

  def testNoBatchMultivariateFullDynamic(self):
    x_value = np.array([[1., 1]], dtype=np.float32)
    mu_value = np.array([1., -1], dtype=np.float32)
    scale_diag_value = np.array([2., 2], dtype=np.float32)

    x = tf1.placeholder_with_default(x_value, shape=None)
    mu = tf1.placeholder_with_default(mu_value, shape=None)
    scale_diag = tf1.placeholder_with_default(
        scale_diag_value, shape=None)

    bijector = tfb.Affine(shift=mu, scale_diag=scale_diag)
    self.assertAllClose([[3., 1]], self.evaluate(bijector.forward(x)))
    self.assertAllClose([[0., 1]], self.evaluate(bijector.inverse(x)))
    self.assertAllClose(
        -np.log(4),
        self.evaluate(bijector.inverse_log_det_jacobian(x, event_ndims=1)))

  def testBatchMultivariateIdentity(self):
    def static_run(fun, x, **kwargs):
      return self.evaluate(fun(x, **kwargs))

    def dynamic_run(fun, x_value, **kwargs):
      x_value = np.array(x_value, dtype=np.float32)
      placeholder = tf1.placeholder_with_default(x_value, shape=None)
      return self.evaluate(fun(placeholder, **kwargs))

    for run in (static_run, dynamic_run):
      mu = [[1., -1]]
      # Corresponds to 1 2x2 matrix, with twos on the diagonal.
      scale = 2.
      bijector = tfb.Affine(shift=mu, scale_identity_multiplier=scale)
      x = [[[1., 1]]]
      self.assertAllClose([[[3., 1]]], run(bijector.forward, x))
      self.assertAllClose([[[0., 1]]], run(bijector.inverse, x))
      self.assertAllClose(
          -np.log(4),
          run(bijector.inverse_log_det_jacobian, x, event_ndims=1))

  def testBatchMultivariateDiag(self):
    def static_run(fun, x, **kwargs):
      return self.evaluate(fun(x, **kwargs))

    def dynamic_run(fun, x_value, **kwargs):
      x_value = np.array(x_value, dtype=np.float32)
      placeholder = tf1.placeholder_with_default(x_value, shape=None)
      return self.evaluate(fun(placeholder, **kwargs))

    for run in (static_run, dynamic_run):
      mu = [[1., -1]]
      # Corresponds to 1 2x2 matrix, with twos on the diagonal.
      scale_diag = [[2., 2]]
      bijector = tfb.Affine(shift=mu, scale_diag=scale_diag)
      x = [[[1., 1]]]
      self.assertAllClose([[[3., 1]]], run(bijector.forward, x))
      self.assertAllClose([[[0., 1]]], run(bijector.inverse, x))
      self.assertAllClose(
          [-np.log(4)],
          run(bijector.inverse_log_det_jacobian, x, event_ndims=1))

  def testBatchMultivariateFullDynamic(self):
    x_value = np.array([[[1., 1]]], dtype=np.float32)
    mu_value = np.array([[1., -1]], dtype=np.float32)
    scale_diag_value = np.array([[2., 2]], dtype=np.float32)

    x = tf1.placeholder_with_default(x_value, shape=None)
    mu = tf1.placeholder_with_default(mu_value, shape=None)
    scale_diag = tf1.placeholder_with_default(
        scale_diag_value, shape=None)

    bijector = tfb.Affine(shift=mu, scale_diag=scale_diag)
    self.assertAllClose([[[3., 1]]], self.evaluate(bijector.forward(x)))
    self.assertAllClose([[[0., 1]]], self.evaluate(bijector.inverse(x)))
    self.assertAllClose(
        [-np.log(4)],
        self.evaluate(bijector.inverse_log_det_jacobian(
            x, event_ndims=1)))

  def testIdentityWithDiagUpdate(self):
    def static_run(fun, x, **kwargs):
      return self.evaluate(fun(x, **kwargs))

    def dynamic_run(fun, x_value, **kwargs):
      x_value = np.array(x_value, dtype=np.float32)
      placeholder = tf1.placeholder_with_default(x_value, shape=None)
      return self.evaluate(fun(placeholder, **kwargs))

    for run in (static_run, dynamic_run):
      mu = -1.
      # Corresponds to scale = 2
      bijector = tfb.Affine(
          shift=mu, scale_identity_multiplier=1., scale_diag=[1., 1., 1.])
      x = [1., 2, 3]  # Three scalar samples (no batches).
      self.assertAllClose([1., 3, 5], run(bijector.forward, x))
      self.assertAllClose([1., 1.5, 2.], run(bijector.inverse, x))
      self.assertAllClose(
          -np.log(2.**3),
          run(bijector.inverse_log_det_jacobian, x, event_ndims=1))

  def testIdentityWithTriL(self):
    def static_run(fun, x, **kwargs):
      return self.evaluate(fun(x, **kwargs))

    def dynamic_run(fun, x_value, **kwargs):
      x_value = np.array(x_value, dtype=np.float32)
      placeholder = tf1.placeholder_with_default(x_value, shape=None)
      return self.evaluate(fun(placeholder, **kwargs))

    for run in (static_run, dynamic_run):
      mu = -1.
      # scale = [[2., 0], [2, 2]]
      bijector = tfb.Affine(
          shift=mu,
          scale_identity_multiplier=1.,
          scale_tril=[[1., 0], [2., 1]])
      x = [[1., 2]]  # One multivariate sample.
      self.assertAllClose([[1., 5]], run(bijector.forward, x))
      self.assertAllClose([[1., 0.5]], run(bijector.inverse, x))
      self.assertAllClose(
          -np.log(4.),
          run(bijector.inverse_log_det_jacobian, x, event_ndims=1))

  def testDiagWithTriL(self):
    def static_run(fun, x, **kwargs):
      return self.evaluate(fun(x, **kwargs))

    def dynamic_run(fun, x_value, **kwargs):
      x_value = np.array(x_value, dtype=np.float32)
      placeholder = tf1.placeholder_with_default(x_value, shape=None)
      return self.evaluate(fun(placeholder, **kwargs))

    for run in (static_run, dynamic_run):
      mu = -1.
      # scale = [[2., 0], [2, 3]]
      bijector = tfb.Affine(
          shift=mu, scale_diag=[1., 2.], scale_tril=[[1., 0], [2., 1]])
      x = [[1., 2]]  # One multivariate sample.
      self.assertAllClose([[1., 7]], run(bijector.forward, x))
      self.assertAllClose([[1., 1 / 3.]], run(bijector.inverse, x))
      self.assertAllClose(
          -np.log(6.),
          run(bijector.inverse_log_det_jacobian, x, event_ndims=1))

  def testIdentityAndDiagWithTriL(self):
    def static_run(fun, x, **kwargs):
      return self.evaluate(fun(x, **kwargs))

    def dynamic_run(fun, x_value, **kwargs):
      x_value = np.array(x_value, dtype=np.float32)
      placeholder = tf1.placeholder_with_default(x_value, shape=None)
      return self.evaluate(fun(placeholder, **kwargs))

    for run in (static_run, dynamic_run):
      mu = -1.
      # scale = [[3., 0], [2, 4]]
      bijector = tfb.Affine(
          shift=mu,
          scale_identity_multiplier=1.0,
          scale_diag=[1., 2.],
          scale_tril=[[1., 0], [2., 1]])
      x = [[1., 2]]  # One multivariate sample.
      self.assertAllClose([[2., 9]], run(bijector.forward, x))
      self.assertAllClose([[2 / 3., 5 / 12.]], run(bijector.inverse, x))
      self.assertAllClose(
          -np.log(12.),
          run(bijector.inverse_log_det_jacobian, x, event_ndims=1))

  def testIdentityWithVDVTUpdate(self):
    def static_run(fun, x, **kwargs):
      return self.evaluate(fun(x, **kwargs))

    def dynamic_run(fun, x_value, **kwargs):
      x_value = np.array(x_value, dtype=np.float32)
      placeholder = tf1.placeholder_with_default(x_value, shape=None)
      return self.evaluate(fun(placeholder, **kwargs))

    for run in (static_run, dynamic_run):
      mu = -1.
      # Corresponds to scale = [[10, 0, 0], [0, 2, 0], [0, 0, 3]]
      bijector = tfb.Affine(
          shift=mu,
          scale_identity_multiplier=2.,
          scale_perturb_diag=[2., 1],
          scale_perturb_factor=[[2., 0], [0., 0], [0, 1]])
      bijector_ref = tfb.Affine(shift=mu, scale_diag=[10., 2, 3])

      x = [1., 2, 3]  # Vector.
      self.assertAllClose([9., 3, 8], run(bijector.forward, x))
      self.assertAllClose(
          run(bijector_ref.forward, x), run(bijector.forward, x))

      self.assertAllClose([0.2, 1.5, 4 / 3.], run(bijector.inverse, x))
      self.assertAllClose(
          run(bijector_ref.inverse, x), run(bijector.inverse, x))
      self.assertAllClose(
          -np.log(60.),
          run(bijector.inverse_log_det_jacobian, x, event_ndims=1))
      self.assertAllClose(
          run(bijector.inverse_log_det_jacobian, x, event_ndims=1),
          run(bijector_ref.inverse_log_det_jacobian, x, event_ndims=1))

  def testDiagWithVDVTUpdate(self):
    def static_run(fun, x, **kwargs):
      return self.evaluate(fun(x, **kwargs))

    def dynamic_run(fun, x_value, **kwargs):
      x_value = np.array(x_value, dtype=np.float32)
      placeholder = tf1.placeholder_with_default(x_value, shape=None)
      return self.evaluate(fun(placeholder, **kwargs))

    for run in (static_run, dynamic_run):
      mu = -1.
      # Corresponds to scale = [[10, 0, 0], [0, 3, 0], [0, 0, 5]]
      bijector = tfb.Affine(
          shift=mu,
          scale_diag=[2., 3, 4],
          scale_perturb_diag=[2., 1],
          scale_perturb_factor=[[2., 0], [0., 0], [0, 1]])
      bijector_ref = tfb.Affine(shift=mu, scale_diag=[10., 3, 5])

      x = [1., 2, 3]  # Vector.
      self.assertAllClose([9., 5, 14], run(bijector.forward, x))
      self.assertAllClose(
          run(bijector_ref.forward, x), run(bijector.forward, x))
      self.assertAllClose([0.2, 1., 0.8], run(bijector.inverse, x))
      self.assertAllClose(
          run(bijector_ref.inverse, x), run(bijector.inverse, x))
      self.assertAllClose(
          -np.log(150.),
          run(bijector.inverse_log_det_jacobian, x, event_ndims=1))
      self.assertAllClose(
          run(bijector.inverse_log_det_jacobian, x, event_ndims=1),
          run(bijector_ref.inverse_log_det_jacobian, x, event_ndims=1))

  def testTriLWithVDVTUpdate(self):
    def static_run(fun, x, **kwargs):
      return self.evaluate(fun(x, **kwargs))

    def dynamic_run(fun, x_value, **kwargs):
      x_value = np.array(x_value, dtype=np.float32)
      placeholder = tf1.placeholder_with_default(x_value, shape=None)
      return self.evaluate(fun(placeholder, **kwargs))

    for run in (static_run, dynamic_run):
      mu = -1.
      # Corresponds to scale = [[10, 0, 0], [1, 3, 0], [2, 3, 5]]
      bijector = tfb.Affine(
          shift=mu,
          scale_tril=[[2., 0, 0], [1, 3, 0], [2, 3, 4]],
          scale_perturb_diag=[2., 1],
          scale_perturb_factor=[[2., 0], [0., 0], [0, 1]])
      bijector_ref = tfb.Affine(
          shift=mu, scale_tril=[[10., 0, 0], [1, 3, 0], [2, 3, 5]])

      x = [1., 2, 3]  # Vector.
      self.assertAllClose([9., 6, 22], run(bijector.forward, x))
      self.assertAllClose(
          run(bijector_ref.forward, x), run(bijector.forward, x))
      self.assertAllClose([0.2, 14 / 15., 4 / 25.], run(bijector.inverse, x))
      self.assertAllClose(
          run(bijector_ref.inverse, x), run(bijector.inverse, x))
      self.assertAllClose(
          -np.log(150.),
          run(bijector.inverse_log_det_jacobian, x, event_ndims=1))
      self.assertAllClose(
          run(bijector.inverse_log_det_jacobian, x, event_ndims=1),
          run(bijector_ref.inverse_log_det_jacobian, x, event_ndims=1))

  def testTriLWithVDVTUpdateNoDiagonal(self):
    def static_run(fun, x, **kwargs):
      return self.evaluate(fun(x, **kwargs))

    def dynamic_run(fun, x_value, **kwargs):
      x_value = np.array(x_value, dtype=np.float32)
      placeholder = tf1.placeholder_with_default(x_value, shape=None)
      return self.evaluate(fun(placeholder, **kwargs))

    for run in (static_run, dynamic_run):
      mu = -1.
      # Corresponds to scale = [[6, 0, 0], [1, 3, 0], [2, 3, 5]]
      bijector = tfb.Affine(
          shift=mu,
          scale_tril=[[2., 0, 0], [1, 3, 0], [2, 3, 4]],
          scale_perturb_diag=None,
          scale_perturb_factor=[[2., 0], [0., 0], [0, 1]])
      bijector_ref = tfb.Affine(
          shift=mu, scale_tril=[[6., 0, 0], [1, 3, 0], [2, 3, 5]])

      x = [1., 2, 3]  # Vector.
      self.assertAllClose([5., 6, 22], run(bijector.forward, x))
      self.assertAllClose(
          run(bijector_ref.forward, x), run(bijector.forward, x))
      self.assertAllClose([1 / 3., 8 / 9., 4 / 30.], run(bijector.inverse, x))
      self.assertAllClose(
          run(bijector_ref.inverse, x), run(bijector.inverse, x))
      self.assertAllClose(
          -np.log(90.),
          run(bijector.inverse_log_det_jacobian, x, event_ndims=1))
      self.assertAllClose(
          run(bijector.inverse_log_det_jacobian, x, event_ndims=1),
          run(bijector_ref.inverse_log_det_jacobian, x, event_ndims=1))

  def testNoBatchMultivariateRaisesWhenSingular(self):
    mu = [1., -1]
    with self.assertRaisesRegexp(
        Exception,
        ".*Singular operator:  Diagonal contained zero values.*"):
      bijector = tfb.Affine(
          shift=mu,
          # Has zero on the diagonal.
          scale_diag=[0., 1],
          validate_args=True)
      self.evaluate(bijector.forward([1., 1.]))

  def _makeScale(self,
                 x,
                 scale_identity_multiplier=None,
                 scale_diag=None,
                 scale_tril=None,
                 scale_perturb_factor=None,
                 scale_perturb_diag=None):
    """Create a scale matrix. Return None if it can not be created."""
    c = scale_identity_multiplier
    d1 = scale_diag
    tril = scale_tril
    v = scale_perturb_factor
    d2 = scale_perturb_diag

    # Ambiguous low rank update.
    if v is None and d2 is not None:
      return None

    if c is None and d1 is None and tril is None:
      # Special case when no scale args are passed in. This means use an
      # identity matrix.
      c = 1.

    matrix = np.float32(0.)
    if c is not None:
      # Infer the dimension from x.
      matrix += c * self._matrix_diag(np.ones_like(x))
    if d1 is not None:
      matrix += self._matrix_diag(np.array(d1, dtype=np.float32))
    if tril is not None:
      matrix += np.array(tril, dtype=np.float32)
    if v is not None:
      v = np.array(v, dtype=np.float32)
      if v.ndim < 2:
        vt = v.T
      else:
        vt = np.swapaxes(v, axis1=v.ndim - 2, axis2=v.ndim - 1)
      if d2 is not None:
        d2 = self._matrix_diag(np.array(d2, dtype=np.float32))
        right = np.matmul(d2, vt)
      else:
        right = vt
      matrix += np.matmul(v, right)
    return matrix

  def _matrix_diag(self, d):
    """Batch version of np.diag."""
    orig_shape = d.shape
    d = np.reshape(d, (int(np.prod(d.shape[:-1])), d.shape[-1]))
    diag_list = []
    for i in range(d.shape[0]):
      diag_list.append(np.diag(d[i, ...]))
    return np.reshape(diag_list, orig_shape + (d.shape[-1],))

  def _testLegalInputs(self, shift=None, scale_params=None, x=None):

    def _powerset(x):
      s = list(x)
      return itertools.chain.from_iterable(
          itertools.combinations(s, r) for r in range(len(s) + 1))

    for args in _powerset(scale_params.items()):
      args = dict(args)

      scale_args = dict({"x": x}, **args)
      scale = self._makeScale(**scale_args)

      # We haven't specified enough information for the scale.
      if scale is None:
        with self.assertRaisesRegexp(ValueError, ("must be specified.")):
          bijector = tfb.Affine(shift=shift, **args)
      else:
        bijector = tfb.Affine(shift=shift, **args)
        np_x = x
        # For the case a vector is passed in, we need to make the shape
        # match the matrix for matmul to work.
        if x.ndim == scale.ndim - 1:
          np_x = np.expand_dims(x, axis=-1)

        forward = np.matmul(scale, np_x) + shift
        if x.ndim == scale.ndim - 1:
          forward = np.squeeze(forward, axis=-1)
        self.assertAllClose(forward, self.evaluate(bijector.forward(x)))

        backward = np.linalg.solve(scale, np_x - shift)
        if x.ndim == scale.ndim - 1:
          backward = np.squeeze(backward, axis=-1)
        self.assertAllClose(backward, self.evaluate(bijector.inverse(x)))

        scale *= np.ones(shape=x.shape[:-1], dtype=scale.dtype)
        ildj = -np.log(np.abs(np.linalg.det(scale)))
        # TODO(jvdillon): We need to make it so the scale_identity_multiplier
        # case does not deviate in expected shape. Fixing this will get rid of
        # these special cases.
        if (ildj.ndim > 0 and (len(scale_args) == 1 or (
            len(scale_args) == 2 and
            scale_args.get("scale_identity_multiplier", None) is not None))):
          ildj = np.squeeze(ildj[0])
        elif ildj.ndim < scale.ndim - 2:
          ildj = np.reshape(ildj, scale.shape[0:-2])
        self.assertAllClose(
            ildj,
            self.evaluate(bijector.inverse_log_det_jacobian(
                x, event_ndims=1)))

  def testLegalInputs(self):
    self._testLegalInputs(
        shift=np.float32(-1),
        scale_params={
            "scale_identity_multiplier": 2.,
            "scale_diag": [2., 3.],
            "scale_tril": [[1., 0.],
                           [-3., 3.]],
            "scale_perturb_factor": [[1., 0],
                                     [1.5, 3.]],
            "scale_perturb_diag": [3., 1.]
        },
        x=np.array(
            [1., 2], dtype=np.float32))

  def testLegalInputsWithBatch(self):
    # Shape of scale is [2, 1, 2, 2]
    self._testLegalInputs(
        shift=np.float32(-1),
        scale_params={
            "scale_identity_multiplier": 2.,
            "scale_diag": [[[2., 3.]], [[1., 2]]],
            "scale_tril": [[[[1., 0.], [-3., 3.]]], [[[0.5, 0.], [1., 1.]]]],
            "scale_perturb_factor": [[[[1., 0], [1.5, 3.]]],
                                     [[[1., 0], [1., 1.]]]],
            "scale_perturb_diag": [[[3., 1.]], [[0.5, 1.]]]
        },
        x=np.array(
            [[[1., 2]], [[3., 4]]], dtype=np.float32))

  def testNegativeDetTrilPlusVDVT(self):
    # scale = [[3.7, 2.7],
    #          [-0.3, -1.3]]
    # inv(scale) = [[0.325, 0.675],
    #               [-0.075, -0.925]]
    # eig(scale) = [3.5324, -1.1324]
    self._testLegalInputs(
        shift=np.float32(-1),
        scale_params={
            "scale_tril": [[1., 0], [-3, -4]],
            "scale_perturb_factor": [[0.1, 0], [0.5, 0.3]],
            "scale_perturb_diag": [3., 1]
        },
        x=np.array(
            [1., 2], dtype=np.float32))

  def testTriLWithVDVTUpdateAdjoint(self):
    def static_run(fun, x, **kwargs):
      return self.evaluate(fun(x, **kwargs))

    def dynamic_run(fun, x_value, **kwargs):
      x_value = np.array(x_value, dtype=np.float32)
      placeholder = tf1.placeholder_with_default(x_value, shape=None)
      return self.evaluate(fun(placeholder, **kwargs))

    for run in (static_run, dynamic_run):
      mu = -1.
      # Corresponds to scale = [[10, 0, 0], [1, 3, 0], [2, 3, 5]]
      bijector = tfb.Affine(
          shift=mu,
          scale_tril=[[2., 0, 0], [1, 3, 0], [2, 3, 4]],
          scale_perturb_diag=[2., 1],
          scale_perturb_factor=[[2., 0], [0., 0], [0, 1]],
          adjoint=True,
          validate_args=True)
      scale_ref = np.array([[10., 0, 0],
                            [1, 3, 0],
                            [2, 3, 5]], dtype=np.float32)
      x = np.array([1., 2, 3], dtype=np.float32)
      expected_forward = np.matmul(scale_ref.T, x) + mu
      self.assertAllClose(expected_forward,
                          run(bijector.forward, x))
      expected_inverse = np.linalg.solve(scale_ref.T, x - mu)
      self.assertAllClose(expected_inverse,
                          run(bijector.inverse, x))
      self.assertAllClose(x,
                          run(bijector.inverse, expected_forward))
      expected_fldj = np.log(np.prod(np.diagonal(scale_ref)))
      self.assertAllClose(
          -expected_fldj,
          run(bijector.inverse_log_det_jacobian, x, event_ndims=1))
      self.assertAllClose(
          expected_fldj,
          run(bijector.forward_log_det_jacobian, x, event_ndims=1))

  def _testScaledIdentityComplexAdjoint(self, is_dynamic):
    shift_ = np.array(-0.5, dtype=np.complex)
    scale_ = np.array(4 + 2j, dtype=np.complex)
    shift = tf1.placeholder_with_default(
        shift_, shape=None if is_dynamic else [])
    scale = tf1.placeholder_with_default(
        scale_, shape=None if is_dynamic else [])
    bijector = tfb.Affine(
        shift=shift,
        scale_identity_multiplier=scale,
        adjoint=True,
        validate_args=True)
    z = np.array([1., 2, 3], dtype=np.complex)
    y = bijector.forward(z)
    x = bijector.inverse(z)
    inv_fwd_z = bijector.inverse(tf.identity(y))
    ildj = bijector.inverse_log_det_jacobian(z, event_ndims=1)
    fldj = bijector.forward_log_det_jacobian(z, event_ndims=1)
    [x_, y_, inv_fwd_z_, ildj_, fldj_] = self.evaluate([
        x, y, inv_fwd_z, ildj, fldj])
    self.assertAllClose(np.conj(scale_) * z + shift_, y_)
    self.assertAllClose((z - shift_) / np.conj(scale_), x_)
    self.assertAllClose(z, inv_fwd_z_)
    self.assertAllClose(z.shape[-1] * np.log(np.abs(scale_)), fldj_)
    self.assertAllClose(-z.shape[-1] * np.log(np.abs(scale_)), ildj_)

  def testScaledIdentityComplexAdjointDynamic(self):
    return self._testScaledIdentityComplexAdjoint(is_dynamic=True)

  def testScaledIdentityComplexAdjointStatic(self):
    return self._testScaledIdentityComplexAdjoint(is_dynamic=False)


if __name__ == "__main__":
  tf.test.main()
