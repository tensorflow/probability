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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions


def _set_seed(seed):
  """Helper which uses graph seed if using eager."""
  # TODO(b/68017812): Deprecate once eager correctly supports seed.
  if tf.executing_eagerly():
    tf.random.set_seed(seed)
    return None
  return seed


@test_util.test_all_tf_execution_regimes
class BlockwiseTest(test_util.TestCase):

  def testDocstring1(self):
    d = tfd.Blockwise(
        [
            tfd.Independent(
                tfd.Normal(
                    loc=tf1.placeholder_with_default(
                        tf.zeros(4, dtype=tf.float64),
                        shape=None,
                    ),
                    scale=1),
                reinterpreted_batch_ndims=1),
            tfd.MultivariateNormalTriL(
                scale_tril=tf1.placeholder_with_default(
                    tf.eye(2, dtype=tf.float32), shape=None)),
        ],
        dtype_override=tf.float32,
        validate_args=True,
    )
    x = d.sample([2, 1], seed=42)
    y = d.log_prob(x)
    x_, y_ = self.evaluate([x, y])
    self.assertEqual((2, 1, 4 + 2), x_.shape)
    self.assertIs(tf.float32, x.dtype)
    self.assertEqual((2, 1), y_.shape)
    self.assertIs(tf.float32, y.dtype)

    self.assertAllClose(
        np.zeros((6,), dtype=np.float32), self.evaluate(d.mean()))

  def testDocstring2(self):
    Root = tfd.JointDistributionCoroutine.Root  # pylint: disable=invalid-name

    def model():
      e = yield Root(tfd.Independent(tfd.Exponential(rate=[100, 120]), 1))
      g = yield tfd.Gamma(concentration=e[..., 0], rate=e[..., 1])
      n = yield Root(tfd.Normal(loc=0, scale=2.))
      yield tfd.Normal(loc=n, scale=g)

    joint = tfd.JointDistributionCoroutine(model)
    d = tfd.Blockwise(joint, validate_args=True)

    x = d.sample([2, 1], seed=42)
    y = d.log_prob(x)
    x_, y_ = self.evaluate([x, y])
    self.assertEqual((2, 1, 2 + 1 + 1 + 1), x_.shape)
    self.assertIs(tf.float32, x.dtype)
    self.assertEqual((2, 1), y_.shape)
    self.assertIs(tf.float32, y.dtype)

  def testSampleReproducible(self):
    Root = tfd.JointDistributionCoroutine.Root  # pylint: disable=invalid-name

    def model():
      e = yield Root(tfd.Independent(tfd.Exponential(rate=[100, 120]), 1))
      g = yield tfd.Gamma(concentration=e[..., 0], rate=e[..., 1])
      n = yield Root(tfd.Normal(loc=0, scale=2.))
      yield tfd.Normal(loc=n, scale=g)

    joint = tfd.JointDistributionCoroutine(model)
    d = tfd.Blockwise(joint, validate_args=True)

    x = d.sample([2, 1], seed=_set_seed(42))
    y = d.sample([2, 1], seed=_set_seed(42))
    x_, y_ = self.evaluate([x, y])
    self.assertAllClose(x_, y_)

  def testVaryingBatchShapeErrorStatic(self):
    with self.assertRaisesRegexp(
        ValueError, 'Distributions must have the same `batch_shape`'):
      tfd.Blockwise(
          [
              tfd.Normal(tf.zeros(2), tf.ones(2)),
              tfd.Normal(0., 1.),
          ],
          validate_args=True,
      )

  def testVaryingBatchShapeErrorDynamicRank(self):
    if tf.executing_eagerly():
      return
    with self.assertRaisesOpError(
        'Distributions must have the same `batch_shape`'):
      loc = tf1.placeholder_with_default(tf.zeros([2]), shape=None)
      dist = tfd.Blockwise(
          [
              tfd.Normal(loc, tf.ones_like(loc)),
              tfd.Independent(tfd.Normal(loc, tf.ones_like(loc)), 1),
          ],
          validate_args=True,
      )
      self.evaluate(dist.mean())

  def testVaryingBatchShapeErrorDynamicDims(self):
    if tf.executing_eagerly():
      return
    with self.assertRaisesOpError(
        'Distributions must have the same `batch_shape`'):
      loc1 = tf1.placeholder_with_default(tf.zeros([1]), shape=None)
      loc2 = tf1.placeholder_with_default(tf.zeros([2]), shape=None)
      dist = tfd.Blockwise(
          [
              tfd.Normal(loc1, tf.ones_like(loc1)),
              tfd.Normal(loc2, tf.ones_like(loc2)),
          ],
          validate_args=True,
      )
      self.evaluate(dist.mean())

  def testAssertValidSample(self):
    loc1 = tf1.placeholder_with_default(tf.zeros([2]), shape=None)
    loc2 = tf1.placeholder_with_default(tf.zeros([2]), shape=None)
    dist = tfd.Blockwise(
        [
            tfd.Normal(loc1, tf.ones_like(loc1)),
            tfd.Normal(loc2, tf.ones_like(loc2)),
        ],
        validate_args=True,
    )

    with self.assertRaisesRegexp(
        ValueError, 'must have at least one dimension'):
      self.evaluate(dist.prob(3.))

  def testKlBlockwiseIsSum(self):

    gamma0 = tfd.Gamma(concentration=[1., 2., 3.], rate=1.)
    gamma1 = tfd.Gamma(concentration=[3., 4., 5.], rate=1.)

    normal0 = tfd.Normal(loc=tf.zeros(3), scale=2.)
    normal1 = tfd.Normal(loc=tf.ones(3), scale=[2., 3., 4.])

    d0 = tfd.Blockwise([
        tfd.Independent(gamma0, reinterpreted_batch_ndims=1),
        tfd.Independent(normal0, reinterpreted_batch_ndims=1)
    ],
                       validate_args=True)

    d1 = tfd.Blockwise([
        tfd.Independent(gamma1, reinterpreted_batch_ndims=1),
        tfd.Independent(normal1, reinterpreted_batch_ndims=1)
    ],
                       validate_args=True)

    kl_sum = tf.reduce_sum(
        (tfd.kl_divergence(gamma0, gamma1) +
         tfd.kl_divergence(normal0, normal1)))

    blockwise_kl = tfd.kl_divergence(d0, d1)

    kl_sum_, blockwise_kl_ = self.evaluate([kl_sum, blockwise_kl])

    self.assertAllClose(kl_sum_, blockwise_kl_)

  def testKLBlockwise(self):
    # d0 and d1 are two MVN's that are 6 dimensional. Construct the
    # corresponding MVNs, and ensure that the KL between the MVNs is close to
    # the Blockwise ones.
    # In both cases the scale matrix has a block diag structure, owing to
    # independence of the component distributions.
    d0 = tfd.Blockwise([
        tfd.Independent(
            tfd.Normal(loc=tf.zeros(4, dtype=tf.float64), scale=1.),
            reinterpreted_batch_ndims=1),
        tfd.MultivariateNormalTriL(
            scale_tril=tf1.placeholder_with_default(
                tf.eye(2, dtype=tf.float64), shape=None)),
    ],
                       validate_args=True)

    d0_mvn = tfd.MultivariateNormalLinearOperator(
        loc=np.float64([0.] * 6),
        scale=tf.linalg.LinearOperatorBlockDiag([
            tf.linalg.LinearOperatorIdentity(num_rows=4, dtype=tf.float64),
            tf.linalg.LinearOperatorLowerTriangular(
                tf.eye(2, dtype=tf.float64))
        ]))

    d1 = tfd.Blockwise([
        tfd.Independent(
            tfd.Normal(loc=tf.ones(4, dtype=tf.float64), scale=1),
            reinterpreted_batch_ndims=1),
        tfd.MultivariateNormalTriL(
            loc=tf.ones(2, dtype=tf.float64),
            scale_tril=tf1.placeholder_with_default(
                np.float64([[1., 0.], [2., 3.]]), shape=None)),
    ],
                       validate_args=True)
    d1_mvn = tfd.MultivariateNormalLinearOperator(
        loc=np.float64([1.] * 6),
        scale=tf.linalg.LinearOperatorBlockDiag([
            tf.linalg.LinearOperatorIdentity(num_rows=4, dtype=tf.float64),
            tf.linalg.LinearOperatorLowerTriangular(
                np.float64([[1., 0.], [2., 3.]]))
        ]))

    blockwise_kl = tfd.kl_divergence(d0, d1)
    mvn_kl = tfd.kl_divergence(d0_mvn, d1_mvn)
    blockwise_kl_, mvn_kl_ = self.evaluate([blockwise_kl, mvn_kl])
    self.assertAllClose(blockwise_kl_, mvn_kl_)

  def testUnconstrainingBijector(self):
    dist = tfd.Exponential(rate=[1., 2., 6.], validate_args=True)
    blockwise_dist = tfd.Blockwise(dist, validate_args=True)
    x = self.evaluate(
        dist._experimental_default_event_space_bijector()(
            tf.ones(dist.batch_shape)))
    x_blockwise = self.evaluate(
        blockwise_dist._experimental_default_event_space_bijector()(
            tf.ones(blockwise_dist.batch_shape)))
    self.assertAllEqual(x, x_blockwise)


@test_util.test_all_tf_execution_regimes
class BlockwiseTestStaticParams(test_util.TestCase):
  use_static_shape = True

  def _input(self, value):
    """Helper to create inputs with optional static shapes."""
    value = tf.convert_to_tensor(value)
    return tf1.placeholder_with_default(
        value, shape=value.shape if self.use_static_shape else None)

  @parameterized.named_parameters(
      (
          'Scalar',
          lambda self: tfd.Normal(self._input(0.), self._input(1.)),
          1,
          [],
      ),
      (
          'Vector',
          lambda self: tfd.Independent(  # pylint: disable=g-long-lambda
              tfd.Normal(self._input(tf.zeros(2)), self._input(tf.ones(2))), 1),
          2,
          [],
      ),
      (
          'Matrix',
          lambda self: tfd.Independent(  # pylint: disable=g-long-lambda
              tfd.Normal(
                  self._input(tf.zeros([2, 3])), self._input(tf.ones([2, 3]))),
              2),
          6,
          [],
      ),
      (
          'VectorBatch',
          lambda self: tfd.Independent(  # pylint: disable=g-long-lambda
              tfd.Normal(
                  self._input(tf.zeros([2, 2])), self._input(tf.ones([2, 2]))),
              1),
          2,
          [2],
      ),
  )
  def testSingleTensor(self, dist_fn, num_elements, batch_shape):
    """Checks that basic properties work with single Tensor distributions."""
    base = dist_fn(self)

    flat = tfd.Blockwise(base, validate_args=True)
    if self.use_static_shape:
      self.assertAllEqual([num_elements], flat.event_shape)
    self.assertAllEqual([num_elements],
                        self.evaluate(flat.event_shape_tensor()))
    if self.use_static_shape:
      self.assertAllEqual(batch_shape, flat.batch_shape)
    self.assertAllEqual(batch_shape, self.evaluate(flat.batch_shape_tensor()))

    base_sample = self.evaluate(base.sample(3, seed=test_util.test_seed()))
    flat_sample = self.evaluate(flat.sample(3, seed=test_util.test_seed()))
    self.assertAllEqual([3] + batch_shape + [num_elements], flat_sample.shape)
    base_sample = flat_sample.reshape(base_sample.shape)

    base_log_prob = self.evaluate(base.log_prob(base_sample))
    flat_log_prob = self.evaluate(flat.log_prob(flat_sample))
    self.assertAllEqual([3] + batch_shape, flat_log_prob.shape)
    self.assertAllClose(base_log_prob, flat_log_prob)

  def _MakeModelFn(self):
    Root = tfd.JointDistributionCoroutine.Root  # pylint: disable=invalid-name

    def model_fn():
      yield Root(tfd.Normal(self._input(0.), self._input(1.)))
      yield Root(
          tfd.Independent(
              tfd.Normal(self._input(tf.zeros(2)), self._input(tf.ones(2))), 1))

    return model_fn

  @parameterized.named_parameters(
      (
          'Sequential',
          lambda self: tfd.JointDistributionSequential([  # pylint: disable=g-long-lambda
              tfd.Normal(self._input(0.), self._input(1.)),
              tfd.Independent(
                  tfd.Normal(self._input(tf.zeros(2)), self._input(tf.ones(2))),
                  1),
          ]),
          [1, 2],
          [],
      ),
      (
          'Named',
          lambda self: tfd.JointDistributionNamed({  # pylint: disable=g-long-lambda
              'a':
                  tfd.Normal(self._input(0.), self._input(1.)),
              'b':
                  tfd.Independent(
                      tfd.Normal(
                          self._input(tf.zeros(2)), self._input(tf.ones(2))), 1
                  ),
          }),
          [1, 2],
          [],
      ),
      (
          'Coroutine',
          lambda self: tfd.JointDistributionCoroutine(self._MakeModelFn()),
          [1, 2],
          [],
      ),
      (
          'SequentialMixedStaticShape',
          lambda self: tfd.JointDistributionSequential([  # pylint: disable=g-long-lambda
              tfd.Normal(0., 1.),
              tfd.Independent(
                  tfd.Normal(self._input(tf.zeros(2)), self._input(tf.ones(2))),
                  1),
          ]),
          [1, 2],
          [],
      ),
      (
          'SequentialBatch',
          lambda self: tfd.JointDistributionSequential([  # pylint: disable=g-long-lambda
              tfd.Normal(tf.zeros(2), tf.ones(2)),
              tfd.Independent(
                  tfd.Normal(
                      self._input(tf.zeros([2, 2])), self._input(
                          tf.ones([2, 2]))), 1),
          ]),
          [1, 2],
          [2],
      ),
  )
  def testJointDistribution(self, dist_fn, nums_elements, batch_shape):
    """Checks that basic properties work with JointDistribution."""
    base = dist_fn(self)
    num_elements = sum(nums_elements)

    flat = tfd.Blockwise(base, validate_args=True)
    if self.use_static_shape:
      self.assertAllEqual([num_elements], flat.event_shape)
    self.assertAllEqual([num_elements],
                        self.evaluate(flat.event_shape_tensor()))
    if self.use_static_shape:
      self.assertAllEqual(batch_shape, flat.batch_shape)
    self.assertAllEqual(batch_shape, self.evaluate(flat.batch_shape_tensor()))

    base_sample = self.evaluate(base.sample(3, seed=test_util.test_seed()))
    base_sample_list = tf.nest.flatten(base_sample)
    flat_sample = self.evaluate(flat.sample(3, seed=test_util.test_seed()))
    self.assertAllEqual([3] + batch_shape + [num_elements], flat_sample.shape)

    split_points = np.cumsum([0] + nums_elements)
    base_sample_list = [
        flat_sample[..., start:end].reshape(base_sample_part.shape)
        for start, end, base_sample_part in zip(
            split_points[:-1], split_points[1:], base_sample_list)
    ]

    base_sample = tf.nest.pack_sequence_as(base_sample, base_sample_list)

    base_log_prob = self.evaluate(base.log_prob(base_sample))
    flat_log_prob = self.evaluate(flat.log_prob(flat_sample))
    self.assertAllEqual([3] + batch_shape, flat_log_prob.shape)
    self.assertAllClose(base_log_prob, flat_log_prob)

  @parameterized.named_parameters(
      (
          'NoBatch',
          lambda self: [  # pylint: disable=g-long-lambda
              tfd.Normal(self._input(0.), self._input(1.)),
              tfd.Independent(
                  tfd.Normal(self._input(tf.zeros(2)), self._input(tf.ones(2))),
                  1),
          ],
          [1, 2],
          [],
      ),
      (
          'MixedStaticShape',
          lambda self: [  # pylint: disable=g-long-lambda
              tfd.Normal(0., 1.),
              tfd.Independent(
                  tfd.Normal(self._input(tf.zeros(2)), self._input(tf.ones(2))),
                  1),
          ],
          [1, 2],
          [],
      ),
      (
          'Batch',
          lambda self: [  # pylint: disable=g-long-lambda
              tfd.Normal(tf.zeros(2), tf.ones(2)),
              tfd.Independent(
                  tfd.Normal(
                      self._input(tf.zeros([2, 2])), self._input(
                          tf.ones([2, 2]))), 1),
          ],
          [1, 2],
          [2],
      ),
  )
  def testDistributionList(self, dists_fn, nums_elements, batch_shape):
    """Checks that basic properties work with a list of distributions."""
    bases = dists_fn(self)
    num_elements = sum(nums_elements)

    flat = tfd.Blockwise(bases, validate_args=True)
    if self.use_static_shape:
      self.assertAllEqual([num_elements], flat.event_shape)
    self.assertAllEqual([num_elements],
                        self.evaluate(flat.event_shape_tensor()))
    if self.use_static_shape:
      self.assertAllEqual(batch_shape, flat.batch_shape)
    self.assertAllEqual(batch_shape, self.evaluate(flat.batch_shape_tensor()))

    base_sample_list = self.evaluate(
        [base.sample(3, seed=test_util.test_seed()) for base in bases])
    flat_sample = self.evaluate(flat.sample(3, seed=test_util.test_seed()))
    self.assertAllEqual([3] + batch_shape + [num_elements], flat_sample.shape)

    split_points = np.cumsum([0] + nums_elements)
    base_sample_list = [
        flat_sample[..., start:end].reshape(base_sample_part.shape)
        for start, end, base_sample_part in zip(
            split_points[:-1], split_points[1:], base_sample_list)
    ]

    base_log_prob = sum(
        self.evaluate([
            base.log_prob(base_sample)
            for base, base_sample in zip(bases, base_sample_list)
        ]))
    flat_log_prob = self.evaluate(flat.log_prob(flat_sample))
    self.assertAllEqual([3] + batch_shape, flat_log_prob.shape)
    self.assertAllClose(base_log_prob, flat_log_prob)


class BlockwiseTestDynamicParams(BlockwiseTestStaticParams):
  use_static_shape = False


if __name__ == '__main__':
  tf.test.main()
