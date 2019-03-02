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


if __name__ == '__main__':
  tf.test.main()
