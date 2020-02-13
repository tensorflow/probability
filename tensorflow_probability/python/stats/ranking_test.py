# Copyright 2019 The TensorFlow Probability Authors.
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
"""Tests for tensorflow_probability.python.stats.ranking_stats."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl.testing import parameterized

import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import test_util


tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class AurocAuprcTest(test_util.TestCase):

  @parameterized.parameters(
      ('ROC', ((.7, .8), (.6, .5)), ((.2, .4), (.6, .7))),
      ('ROC', .6, .3),
      ('PR', (.8, .6), (.4, .7)),
      ('PR', .5, .4)
      )
  def testAurocAuprc(self, curve, positive_means, negative_means):

    num_positive_trials = 4001
    num_negative_trials = 5156
    total_trials = num_positive_trials + num_negative_trials
    num_positive_quantiles = 445
    num_negative_quantiles = 393

    dist_positive = tfd.TruncatedNormal(
        positive_means, scale=0.2, low=0., high=1.)
    dist_negative = tfd.TruncatedNormal(
        negative_means, scale=0.2, low=0., high=1.)

    positive_trials = dist_positive.sample(
        num_positive_trials, seed=test_util.test_seed())
    negative_trials = dist_negative.sample(
        num_negative_trials, seed=test_util.test_seed())

    positive_trials_, negative_trials_ = self.evaluate(
        [positive_trials, negative_trials])
    q1 = tfp.stats.quantiles(positive_trials_, num_positive_quantiles, axis=0)
    q0 = tfp.stats.quantiles(negative_trials_, num_negative_quantiles, axis=0)

    y_true = np.array([1] * num_positive_trials + [0] * num_negative_trials)
    y_pred = np.concatenate([positive_trials_, negative_trials_])

    def auc_fn(y_pred):
      m = tf.keras.metrics.AUC(num_thresholds=total_trials, curve=curve)
      self.evaluate([v.initializer for v in m.variables])
      self.evaluate(m.update_state(y_true, y_pred))
      out = self.evaluate(m.result())
      return out

    batch_shape = np.array(positive_means).shape
    batch_rank = len(batch_shape)
    if batch_rank > 0:
      # Transpose so that batch dimensions are first and data dimension is last
      transpose_axes = list(range(1, batch_rank + 1)) + [0]
      q0 = tf.transpose(q0, transpose_axes)
      q1 = tf.transpose(q1, transpose_axes)

    true_auc = np.apply_along_axis(auc_fn, 0, y_pred)

    auc = tfp.stats.quantile_auc(
        q0, num_negative_trials, q1, num_positive_trials, curve=curve)
    auc_ = self.evaluate(auc)

    self.assertAllClose(auc_, true_auc, atol=5e-3, rtol=0.)
    self.assertAllEqual(batch_shape, auc_.shape)


if __name__ == '__main__':
  tf.test.main()
