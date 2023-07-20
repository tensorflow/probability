# Copyright 2021 The TensorFlow Probability Authors.
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
import tensorflow.compat.v2 as tf
from fun_mc import using_tensorflow as fun_mc
from absl.testing import absltest


class TensorFlowIntegrationTest(absltest.TestCase):

  def testBasic(self):

    def fun(x):
      return x + 1., 2 * x

    x, _ = fun_mc.trace(state=0., fn=fun, num_steps=5)

    self.assertIsInstance(x, tf.Tensor)


if __name__ == '__main__':
  absltest.main()
