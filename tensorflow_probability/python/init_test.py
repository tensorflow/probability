# Copyright 2026 The TensorFlow Probability Authors.
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
"""Tests for the TensorFlow environment check in tensorflow_probability."""

import sys
from unittest import mock

import tensorflow as tf

from tensorflow_probability import python as tfp_python
from tensorflow_probability.python.internal import test_util


class ValidateTfEnvironmentTest(test_util.TestCase):

  def testDoesNotNeedDistutils(self):
    # distutils was removed in Python 3.12, and TensorFlow no longer installs
    # setuptools, which used to provide a replacement.
    with mock.patch.dict(sys.modules, {'distutils': None,
                                       'distutils.version': None}):
      tfp_python._validate_tf_environment('distributions')

  def testAcceptsNonPep440Version(self):
    with mock.patch.object(tf, '__version__', '2.19.0-custom-build'):
      tfp_python._validate_tf_environment('distributions')

  def testRejectsOldNonPep440Version(self):
    with mock.patch.object(tf, '__version__', '2.17.0-custom-build'):
      with self.assertRaisesRegex(ImportError, 'requires TensorFlow'):
        tfp_python._validate_tf_environment('distributions')


if __name__ == '__main__':
  test_util.main()
