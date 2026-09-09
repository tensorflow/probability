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
"""Tests for poisson binomial distribution."""

# Dependency imports

from importlib.metadata import distribution
import numpy as np
from scipy import stats
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import discrete_rejection_sampling
from tensorflow_probability.python.distributions import geometric
from tensorflow_probability.python.distributions.poisson_binomial import PoissonBinomial
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class PoissonBinomialTest(test_util.TestCase):

  def testFunctionalityPoissonBinomial(self):

    probs = np.float32([0.2, 0.5, 0.3, 0.4])
    dist = PoissonBinomial(probs=probs)
    
    self.evaluate(dist.cdf(3))
    self.evaluate(dist.pdf(3))


if __name__ == '__main__':
  test_util.main()