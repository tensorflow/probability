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
"""Tests for StructuralTimeSeries utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.sts.internal import missing_values_util


class _MissingValuesUtilityTests(test_util.TestCase):

  def testMoments(self):
    series = np.random.randn(2, 4)
    mask = np.array([[False, True, False, True],
                     [True, False, True, False]])
    unmasked_entries = [[series[0, 0], series[0, 2]],
                        [series[1, 1], series[1, 3]]]
    expected_mean = np.mean(unmasked_entries, axis=-1).astype(self.dtype)
    expected_variance = np.var(
        unmasked_entries, axis=-1).astype(self.dtype)

    mean, variance = missing_values_util.moments_of_masked_time_series(
        self._build_tensor(series),
        broadcast_mask=self._build_tensor(mask, dtype=np.bool))

    mean_, variance_ = self.evaluate((mean, variance))
    self.assertAllClose(mean_, expected_mean)
    self.assertAllClose(variance_, expected_variance)

  def testInitialValueOfMaskedTimeSeries(self):

    if not self.use_static_shape:
      return  # Dynamic rank is not currently supported.

    series = np.random.randn(2, 4)
    mask = np.array([[False, True, False, True],
                     [True, False, True, False]])
    expected_initial_values = [series[0, 0], series[1, 1]]

    initial_values = missing_values_util.initial_value_of_masked_time_series(
        self._build_tensor(series),
        broadcast_mask=self._build_tensor(mask, dtype=np.bool))

    self.assertAllClose(self.evaluate(initial_values), expected_initial_values)

  def _build_tensor(self, ndarray, dtype=None):
    """Convert a numpy array to a TF placeholder.

    Args:
      ndarray: any object convertible to a numpy array via `np.asarray()`.
      dtype: optional `dtype`.

    Returns:
      placeholder: a TensorFlow `placeholder` with default value given by the
      provided `ndarray`, dtype given by `self.dtype` (if not specified), and
      shape specified statically only if `self.use_static_shape` is `True`.
    """

    ndarray = np.asarray(ndarray).astype(self.dtype if dtype is None else dtype)
    return tf1.placeholder_with_default(
        ndarray, shape=ndarray.shape if self.use_static_shape else None)


class _MissingValuesUtilityTestsDynamicFloat32(_MissingValuesUtilityTests):
  use_static_shape = False
  dtype = np.float32


@test_util.test_all_tf_execution_regimes
class _MissingValuesUtilityTestsStaticFloat64(_MissingValuesUtilityTests):
  use_static_shape = True
  dtype = np.float64

del _MissingValuesUtilityTests

if __name__ == "__main__":
  tf.test.main()
