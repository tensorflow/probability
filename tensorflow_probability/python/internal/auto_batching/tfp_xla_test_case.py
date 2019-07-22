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
"""A TestCase wrapper for TF Probability, inspired in part by XLATestCase."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports
from absl import flags
import tensorflow as tf
from tensorflow_probability.python.internal.auto_batching import xla
from tensorflow.python.ops import control_flow_util  # pylint: disable=g-direct-tensorflow-import

flags.DEFINE_string('test_device', None,
                    'TensorFlow device on which to place operators under test')
flags.DEFINE_string('tf_xla_flags', None,
                    'Value to set the TF_XLA_FLAGS environment variable to')
FLAGS = flags.FLAGS


class TFPXLATestCase(tf.test.TestCase):
  """TFP+XLA test harness."""

  def __init__(self, method_name='runTest'):
    super(TFPXLATestCase, self).__init__(method_name)
    self.device = FLAGS.test_device
    if FLAGS.tf_xla_flags is not None:
      os.environ['TF_XLA_FLAGS'] = FLAGS.tf_xla_flags

  def setUp(self):
    self._orig_cfv2 = control_flow_util.ENABLE_CONTROL_FLOW_V2
    # We require control flow v2 for XLA CPU.
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
    super(TFPXLATestCase, self).setUp()

  def tearDown(self):
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = self._orig_cfv2
    super(TFPXLATestCase, self).tearDown()

  def wrap_fn(self, f):
    return xla.compile_nested_output(
        f, (tf.compat.v1.tpu.rewrite if 'TPU' in self.device
            else tf.xla.experimental.compile))
