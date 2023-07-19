# Copyright 2022 The TensorFlow Probability Authors.
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
"""Tests for Positive-Semidefinite Kernels test utilities."""

import numpy as np
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math.psd_kernels import exponentiated_quadratic
from tensorflow_probability.python.math.psd_kernels.internal import test_util as psd_kernel_test_util


class TestUtilTest(test_util.TestCase):

  def testMulipartTestKernel(self):
    base_kernel = exponentiated_quadratic.ExponentiatedQuadratic()
    multipart_kernel = psd_kernel_test_util.MultipartTestKernel(base_kernel)

    x = np.random.normal(size=(10, 5))
    x_multipart = dict(zip(('foo', 'bar'), np.split(x, [2, 3], axis=-1)))

    self.assertSameStructure(multipart_kernel.feature_ndims,
                             {'foo': 1, 'bar': 1})
    self.assertSameElements(multipart_kernel.dtype.keys(), ('foo', 'bar'))
    self.assertAllClose(base_kernel.apply(x, x),
                        multipart_kernel.apply(x_multipart, x_multipart))


if __name__ == '__main__':
  test_util.main()
