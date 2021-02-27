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
"""Tests for Monte Carlo dropout layer."""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util

@test_util.test_all_tf_execution_regimes
class MonteCarloDropoutLayerTest(test_util.TestCase):

  def test_end_to_end(self):
    # Make the test reproducible.
    np.random.seed(468)
    tf.random.set_seed(575)
    # Generate dataset.
    x_train = np.random.rand(100, 10)
    y_train = np.sum(x_train, axis=-1)
    # Create model.
    model = tf.keras.Sequential([
      tfp.layers.MonteCarloDropout(0.2),
      tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam',
                  loss='mse')
    # Train model.
    model.fit(x=x_train, y=y_train)
    # Compare two predictions.
    x_test = np.random.rand(1, 10)
    prediction1 = model.predict(x_test)
    prediction2 = model.predict(x_test)
    assert prediction1[0] != prediction2[0]

if __name__ == '__main__':
  tf.test.main()
