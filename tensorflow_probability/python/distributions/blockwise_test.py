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

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_case
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top

tfd = tfp.distributions


@test_util.run_all_in_graph_and_eager_modes
class BlockwiseTest(test_case.TestCase):

  # TODO(b/126735233): Add better unit-tests.

  def test_works_correctly(self):
    d = tfd.Blockwise(
        [
            tfd.Independent(
                tfd.Normal(
                    loc=tf.compat.v1.placeholder_with_default(
                        tf.zeros(4, dtype=tf.float64),
                        shape=None),
                    scale=1),
                reinterpreted_batch_ndims=1),
            tfd.MultivariateNormalTriL(
                scale_tril=tf.compat.v1.placeholder_with_default(
                    tf.eye(2, dtype=tf.float32),
                    shape=None)),
        ],
        dtype_override=tf.float32,
        validate_args=True)
    x = d.sample([2, 1], seed=42)
    y = d.log_prob(x)
    x_, y_ = self.evaluate([x, y])
    self.assertEqual((2, 1, 4 + 2), x_.shape)
    self.assertIs(tf.float32, x.dtype)
    self.assertEqual((2, 1), y_.shape)
    self.assertIs(tf.float32, y.dtype)

    self.assertAllClose(np.zeros((6,), dtype=np.float32),
                        self.evaluate(d.mean()))

  def testKlBlockwiseError(self):
    d0 = tfd.Blockwise(
        [
            tfd.Independent(
                tfd.Normal(loc=tf.zeros(3., dtype=tf.float64), scale=1.),
                reinterpreted_batch_ndims=1),
            tfd.MultivariateNormalTriL(
                scale_tril=tf.eye(2, dtype=tf.float64))
        ])

    # Same event_shape as d0.
    d1 = tfd.Blockwise(
        [
            tfd.Independent(
                tfd.Normal(loc=tf.zeros(5., tf.float64), scale=1.),
                reinterpreted_batch_ndims=1),
        ])

    with self.assertRaisesRegexp(ValueError, 'same number of component'):
      tfd.kl_divergence(d0, d1)

    # Fix d1 to have the same event_shape, but components have different shapes.
    d1 = tfd.Blockwise(
        [
            tfd.MultivariateNormalTriL(
                scale_tril=tf.eye(2, dtype=tf.float64)),
            tfd.Independent(
                tfd.Normal(loc=tf.zeros(3., dtype=tf.float64), scale=1.),
                reinterpreted_batch_ndims=1),
        ])
    with self.assertRaisesRegexp(ValueError, 'same pairwise event shapes'):
      tfd.kl_divergence(d0, d1)

  def testKlBlockwiseIsSum(self):

    gamma0 = tfd.Gamma(concentration=[1., 2., 3.], rate=1.)
    gamma1 = tfd.Gamma(concentration=[3., 4., 5.], rate=1.)

    normal0 = tfd.Normal(loc=tf.zeros(3.), scale=2.)
    normal1 = tfd.Normal(loc=tf.ones(3.), scale=[2., 3., 4.])

    d0 = tfd.Blockwise([
        tfd.Independent(gamma0, reinterpreted_batch_ndims=1),
        tfd.Independent(normal0, reinterpreted_batch_ndims=1)])

    d1 = tfd.Blockwise([
        tfd.Independent(gamma1, reinterpreted_batch_ndims=1),
        tfd.Independent(normal1, reinterpreted_batch_ndims=1)])

    kl_sum = tf.reduce_sum(
        input_tensor=(tfd.kl_divergence(gamma0, gamma1) +
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
    d0 = tfd.Blockwise(
        [
            tfd.Independent(
                tfd.Normal(loc=tf.zeros(4, dtype=tf.float64), scale=1.),
                reinterpreted_batch_ndims=1),
            tfd.MultivariateNormalTriL(
                scale_tril=tf.compat.v1.placeholder_with_default(
                    tf.eye(2, dtype=tf.float64),
                    shape=None)),
        ])

    d0_mvn = tfd.MultivariateNormalLinearOperator(
        loc=np.float64([0.] * 6),
        scale=tf.linalg.LinearOperatorBlockDiag([
            tf.linalg.LinearOperatorIdentity(
                num_rows=4,
                dtype=tf.float64),
            tf.linalg.LinearOperatorLowerTriangular(
                tf.eye(2, dtype=tf.float64))]))

    d1 = tfd.Blockwise(
        [
            tfd.Independent(
                tfd.Normal(loc=tf.ones(4, dtype=tf.float64), scale=1),
                reinterpreted_batch_ndims=1),
            tfd.MultivariateNormalTriL(
                loc=tf.ones(2, dtype=tf.float64),
                scale_tril=tf.compat.v1.placeholder_with_default(
                    np.float64([[1., 0.], [2., 3.]]),
                    shape=None)),
        ])
    d1_mvn = tfd.MultivariateNormalLinearOperator(
        loc=np.float64([1.] * 6),
        scale=tf.linalg.LinearOperatorBlockDiag([
            tf.linalg.LinearOperatorIdentity(
                num_rows=4,
                dtype=tf.float64),
            tf.linalg.LinearOperatorLowerTriangular(
                np.float64([[1., 0.], [2., 3.]]))]))

    blockwise_kl = tfd.kl_divergence(d0, d1)
    mvn_kl = tfd.kl_divergence(d0_mvn, d1_mvn)
    blockwise_kl_, mvn_kl_ = self.evaluate([blockwise_kl, mvn_kl])
    self.assertAllClose(blockwise_kl_, mvn_kl_)


if __name__ == '__main__':
  tf.test.main()
