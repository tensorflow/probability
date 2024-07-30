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
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import blockwise
from tensorflow_probability.python.distributions import exponential
from tensorflow_probability.python.distributions import gamma
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import joint_distribution_coroutine as jdc
from tensorflow_probability.python.distributions import joint_distribution_named as jdn
from tensorflow_probability.python.distributions import joint_distribution_sequential as jds
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import mvn_linear_operator
from tensorflow_probability.python.distributions import mvn_tril
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.internal import test_util


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
    d = blockwise.Blockwise(
        [
            independent.Independent(
                normal.Normal(
                    loc=tf1.placeholder_with_default(
                        tf.zeros(4, dtype=tf.float64),
                        shape=None,
                    ),
                    scale=1),
                reinterpreted_batch_ndims=1),
            mvn_tril.MultivariateNormalTriL(
                scale_tril=tf1.placeholder_with_default(
                    tf.eye(2, dtype=tf.float32), shape=None)),
        ],
        dtype_override=tf.float32,
        validate_args=True,
    )
    x = d.sample([2, 1], seed=test_util.test_seed())
    y = d.log_prob(x)
    x_, y_ = self.evaluate([x, y])
    self.assertEqual((2, 1, 4 + 2), x_.shape)
    self.assertDTypeEqual(x, np.float32)
    self.assertEqual((2, 1), y_.shape)
    self.assertDTypeEqual(y, np.float32)

    self.assertAllClose(
        np.zeros((6,), dtype=np.float32), self.evaluate(d.mean()))

  def testDocstring2(self):
    Root = jdc.JointDistributionCoroutine.Root  # pylint: disable=invalid-name

    def model():
      e = yield Root(
          independent.Independent(exponential.Exponential(rate=[100, 120]), 1))
      g = yield gamma.Gamma(concentration=e[..., 0], rate=e[..., 1])
      n = yield Root(normal.Normal(loc=0, scale=2.))
      yield normal.Normal(loc=n, scale=g)

    joint = jdc.JointDistributionCoroutine(model)
    d = blockwise.Blockwise(joint, validate_args=True)

    x = d.sample([2, 1], seed=test_util.test_seed())
    y = d.log_prob(x)
    x_, y_ = self.evaluate([x, y])
    self.assertEqual((2, 1, 2 + 1 + 1 + 1), x_.shape)
    self.assertDTypeEqual(x, np.float32)
    self.assertEqual((2, 1), y_.shape)
    self.assertDTypeEqual(y, np.float32)

  def testSampleReproducible(self):
    Root = jdc.JointDistributionCoroutine.Root  # pylint: disable=invalid-name

    def model():
      e = yield Root(
          independent.Independent(exponential.Exponential(rate=[100, 120]), 1))
      g = yield gamma.Gamma(concentration=e[..., 0], rate=e[..., 1])
      n = yield Root(normal.Normal(loc=0, scale=2.))
      yield normal.Normal(loc=n, scale=g)

    joint = jdc.JointDistributionCoroutine(model)
    d = blockwise.Blockwise(joint, validate_args=True)
    seed = test_util.test_seed()

    tf.random.set_seed(seed)
    x = d.sample([2, 1], seed=seed)
    tf.random.set_seed(seed)
    y = d.sample([2, 1], seed=seed)
    x_, y_ = self.evaluate([x, y])
    self.assertAllClose(x_, y_)

  def testVaryingBatchShapeErrorStatic(self):
    with self.assertRaisesRegex(
        ValueError, 'Distributions must have the same `batch_shape`'):
      blockwise.Blockwise(
          [
              normal.Normal(tf.zeros(2), tf.ones(2)),
              normal.Normal(0., 1.),
          ],
          validate_args=True,
      )

  def testVaryingBatchShapeErrorDynamicRank(self):
    if tf.executing_eagerly():
      return
    with self.assertRaisesOpError(
        'Distributions must have the same `batch_shape`'):
      loc = tf1.placeholder_with_default(tf.zeros([2]), shape=None)
      dist = blockwise.Blockwise(
          [
              normal.Normal(loc, tf.ones_like(loc)),
              independent.Independent(normal.Normal(loc, tf.ones_like(loc)), 1),
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
      dist = blockwise.Blockwise(
          [
              normal.Normal(loc1, tf.ones_like(loc1)),
              normal.Normal(loc2, tf.ones_like(loc2)),
          ],
          validate_args=True,
      )
      self.evaluate(dist.mean())

  def testAssertValidSample(self):
    loc1 = tf1.placeholder_with_default(tf.zeros([2]), shape=None)
    loc2 = tf1.placeholder_with_default(tf.zeros([2]), shape=None)
    dist = blockwise.Blockwise(
        [
            normal.Normal(loc1, tf.ones_like(loc1)),
            normal.Normal(loc2, tf.ones_like(loc2)),
        ],
        validate_args=True,
    )

    with self.assertRaisesRegex(
        ValueError, 'must have at least one dimension'):
      self.evaluate(dist.prob(3.))

  def testKlBlockwiseIsSum(self):

    gamma0 = gamma.Gamma(concentration=[1., 2., 3.], rate=1.)
    gamma1 = gamma.Gamma(concentration=[3., 4., 5.], rate=1.)

    normal0 = normal.Normal(loc=tf.zeros(3), scale=2.)
    normal1 = normal.Normal(loc=tf.ones(3), scale=[2., 3., 4.])

    d0 = blockwise.Blockwise([
        independent.Independent(gamma0, reinterpreted_batch_ndims=1),
        independent.Independent(normal0, reinterpreted_batch_ndims=1)
    ],
                             validate_args=True)

    d1 = blockwise.Blockwise([
        independent.Independent(gamma1, reinterpreted_batch_ndims=1),
        independent.Independent(normal1, reinterpreted_batch_ndims=1)
    ],
                             validate_args=True)

    kl_sum = tf.reduce_sum((kullback_leibler.kl_divergence(gamma0, gamma1) +
                            kullback_leibler.kl_divergence(normal0, normal1)))

    blockwise_kl = kullback_leibler.kl_divergence(d0, d1)

    kl_sum_, blockwise_kl_ = self.evaluate([kl_sum, blockwise_kl])

    self.assertAllClose(kl_sum_, blockwise_kl_)

  def testKLBlockwise(self):
    # d0 and d1 are two MVN's that are 6 dimensional. Construct the
    # corresponding MVNs, and ensure that the KL between the MVNs is close to
    # the Blockwise ones.
    # In both cases the scale matrix has a block diag structure, owing to
    # independence of the component distributions.
    d0 = blockwise.Blockwise([
        independent.Independent(
            normal.Normal(loc=tf.zeros(4, dtype=tf.float64), scale=1.),
            reinterpreted_batch_ndims=1),
        mvn_tril.MultivariateNormalTriL(
            scale_tril=tf1.placeholder_with_default(
                tf.eye(2, dtype=tf.float64), shape=None)),
    ],
                             validate_args=True)

    d0_mvn = mvn_linear_operator.MultivariateNormalLinearOperator(
        loc=np.float64([0.] * 6),
        scale=tf.linalg.LinearOperatorBlockDiag([
            tf.linalg.LinearOperatorIdentity(num_rows=4, dtype=tf.float64),
            tf.linalg.LinearOperatorLowerTriangular(
                tf.eye(2, dtype=tf.float64))
        ]))

    d1 = blockwise.Blockwise([
        independent.Independent(
            normal.Normal(loc=tf.ones(4, dtype=tf.float64), scale=1),
            reinterpreted_batch_ndims=1),
        mvn_tril.MultivariateNormalTriL(
            loc=tf.ones(2, dtype=tf.float64),
            scale_tril=tf1.placeholder_with_default(
                np.float64([[1., 0.], [2., 3.]]), shape=None)),
    ],
                             validate_args=True)
    d1_mvn = mvn_linear_operator.MultivariateNormalLinearOperator(
        loc=np.float64([1.] * 6),
        scale=tf.linalg.LinearOperatorBlockDiag([
            tf.linalg.LinearOperatorIdentity(num_rows=4, dtype=tf.float64),
            tf.linalg.LinearOperatorLowerTriangular(
                np.float64([[1., 0.], [2., 3.]]))
        ]))

    blockwise_kl = kullback_leibler.kl_divergence(d0, d1)
    mvn_kl = kullback_leibler.kl_divergence(d0_mvn, d1_mvn)
    blockwise_kl_, mvn_kl_ = self.evaluate([blockwise_kl, mvn_kl])
    self.assertAllClose(blockwise_kl_, mvn_kl_)

  def testUnconstrainingBijector(self):
    dist = exponential.Exponential(rate=[1., 2., 6.], validate_args=True)
    blockwise_dist = blockwise.Blockwise(dist, validate_args=True)
    x = self.evaluate(
        dist.experimental_default_event_space_bijector()(
            tf.ones(dist.batch_shape)))
    x_blockwise = self.evaluate(
        blockwise_dist.experimental_default_event_space_bijector()(
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
          lambda self: normal.Normal(self._input(0.), self._input(1.)),
          1,
          [],
      ),
      (
          'Vector',
          lambda self: independent.Independent(  # pylint: disable=g-long-lambda
              normal.Normal(self._input(tf.zeros(2)), self._input(tf.ones(2))),
              1),
          2,
          [],
      ),
      (
          'Matrix',
          lambda self: independent.Independent(  # pylint: disable=g-long-lambda
              normal.Normal(
                  self._input(tf.zeros([2, 3])), self._input(tf.ones([2, 3]))),
              2),
          6,
          [],
      ),
      (
          'VectorBatch',
          lambda self: independent.Independent(  # pylint: disable=g-long-lambda
              normal.Normal(
                  self._input(tf.zeros([2, 2])), self._input(tf.ones([2, 2]))),
              1),
          2,
          [2],
      ),
  )
  def testSingleTensor(self, dist_fn, num_elements, batch_shape):
    """Checks that basic properties work with single Tensor distributions."""
    base = dist_fn(self)

    flat = blockwise.Blockwise(base, validate_args=True)
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
    Root = jdc.JointDistributionCoroutine.Root  # pylint: disable=invalid-name

    def model_fn():
      yield Root(normal.Normal(self._input(0.), self._input(1.)))
      yield Root(
          independent.Independent(
              normal.Normal(self._input(tf.zeros(2)), self._input(tf.ones(2))),
              1))

    return model_fn

  @parameterized.named_parameters(
      (
          'Sequential',
          lambda self: jds.JointDistributionSequential([  # pylint: disable=g-long-lambda
              normal.Normal(self._input(0.), self._input(1.)),
              independent.Independent(
                  normal.Normal(self._input(tf.zeros(2)), self._input(tf.ones(2))),
                  1),
          ]),
          [1, 2],
          [],
      ),
      (
          'Named',
          lambda self: jdn.JointDistributionNamed({  # pylint: disable=g-long-lambda
              'a':
                  normal.Normal(self._input(0.), self._input(1.)),
              'b':
                  independent.Independent(
                      normal.Normal(
                          self._input(tf.zeros(2)), self._input(tf.ones(2))), 1
                  ),
          }),
          [1, 2],
          [],
      ),
      (
          'Coroutine',
          lambda self: jdc.JointDistributionCoroutine(self._MakeModelFn()),
          [1, 2],
          [],
      ),
      (
          'SequentialMixedStaticShape',
          lambda self: jds.JointDistributionSequential([  # pylint: disable=g-long-lambda
              normal.Normal(0., 1.),
              independent.Independent(
                  normal.Normal(self._input(tf.zeros(2)), self._input(tf.ones(2))),
                  1),
          ]),
          [1, 2],
          [],
      ),
      (
          'SequentialBatch',
          lambda self: jds.JointDistributionSequential([  # pylint: disable=g-long-lambda
              normal.Normal(tf.zeros(2), tf.ones(2)),
              independent.Independent(
                  normal.Normal(
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

    flat = blockwise.Blockwise(base, validate_args=True)
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
              normal.Normal(self._input(0.), self._input(1.)),
              independent.Independent(
                  normal.Normal(
                      self._input(tf.zeros(2)), self._input(tf.ones(2))), 1),
          ],
          [1, 2],
          [],
      ),
      (
          'MixedStaticShape',
          lambda self: [  # pylint: disable=g-long-lambda
              normal.Normal(0., 1.),
              independent.Independent(
                  normal.Normal(
                      self._input(tf.zeros(2)), self._input(tf.ones(2))), 1),
          ],
          [1, 2],
          [],
      ),
      (
          'Batch',
          lambda self: [  # pylint: disable=g-long-lambda
              normal.Normal(tf.zeros(2), tf.ones(2)),
              independent.Independent(
                  normal.Normal(
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

    flat = blockwise.Blockwise(bases, validate_args=True)
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
  test_util.main()
