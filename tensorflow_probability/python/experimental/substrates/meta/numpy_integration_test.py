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
# ============================================================================
"""Integration test TFP+Numpy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

import numpy as np

import tensorflow_probability as tfp
tfp = tfp.experimental.substrates.numpy

tfb = tfp.bijectors
tfd = tfp.distributions


class NumpyIntegrationTest(absltest.TestCase):

  def testBijector(self):

    def f(x):
      return tfb.GumbelCDF(loc=np.arange(3.)).forward(x)

    f(0.)
    f(np.array([1, 2, 3.]))

  def testDistribution(self):

    def f(s):
      return tfd.Normal(loc=np.arange(3.), scale=s).sample()

    f(1.)
    f(np.array([1, 2, 3.]))


if __name__ == '__main__':
  absltest.main()
