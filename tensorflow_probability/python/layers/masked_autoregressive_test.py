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

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python import layers as tfpl
from tensorflow_probability.python.internal import test_util

tfk = tf.keras
tfkl = tf.keras.layers


@test_util.test_all_tf_execution_regimes
class AutoregressiveTransformTest(test_util.TestCase):

  def test_doc_string(self):
    # Generate data -- as in Figure 1 in [Papamakarios et al. (2017)][1]).
    n = 2000
    x2 = np.random.randn(n) * 2.
    x1 = np.random.randn(n) + (x2 * x2 / 4.)
    data = np.stack([x1, x2], axis=-1)

    # Density estimation with MADE.
    model = tfk.Sequential([
        # NOTE: This model takes no input and outputs a Distribution.  (We use
        # the batch_size and type of the input, but there are no actual input
        # values because the last dimension of the shape is 0.)
        #
        # For conditional density estimation, the model would take the
        # conditioning values as input.)
        tfkl.InputLayer(input_shape=(0,), dtype=tf.float32),

        # Given the empty input, return a standard normal distribution with
        # matching batch_shape and event_shape of [2].
        # pylint: disable=g-long-lambda
        tfpl.DistributionLambda(lambda t: tfd.MultivariateNormalDiag(
            loc=tf.zeros(tf.concat([tf.shape(t)[:-1], [2]], axis=0)),
            scale_diag=[1., 1.])),

        # Transform the standard normal distribution with event_shape of [2] to
        # the target distribution with event_shape of [2].
        tfpl.AutoregressiveTransform(tfb.AutoregressiveNetwork(
            params=2, hidden_units=[10], activation='relu')),
    ])

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=lambda y, rv_y: -rv_y.log_prob(y))

    model.fit(x=np.zeros((n, 0)),
              y=data,
              batch_size=25,
              epochs=1,
              steps_per_epoch=1,  # Usually n // 25,
              verbose=True)

    distribution = model(np.zeros((0,)))
    self.assertEqual((4, 2), self.evaluate(distribution.sample(4)).shape)
    self.assertEqual((5, 3), self.evaluate(distribution.log_prob(
        np.zeros((5, 3, 2), dtype=np.float32))).shape)


if __name__ == '__main__':
  tf.test.main()
