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
"""Testing the TFP Hypothesis strategies.

(As opposed to using them to test other things).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import hypothesis as hp
from hypothesis import strategies as hps
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class HypothesisTestlibTest(test_util.TestCase):

  @parameterized.parameters((support,) for support in tfp_hps.ALL_SUPPORTS)
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings()
  def testTensorsInSupportsAlwaysFinite(self, support, data):
    try:
      result_ = data.draw(tfp_hps.tensors_in_support(support))
    except NotImplementedError:
      # Constraint class doesn't have a constrainer function at all, so this
      # test is moot.
      return
    result = self.evaluate(result_)
    self.assertTrue(np.all(np.isfinite(result)))


if __name__ == '__main__':
  tf.test.main()
