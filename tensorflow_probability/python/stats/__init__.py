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
"""Statistical functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,line-too-long,g-importing-member

from tensorflow_probability.python.stats.quantiles import find_bins
from tensorflow_probability.python.stats.quantiles import percentile
from tensorflow_probability.python.stats.quantiles import quantiles
from tensorflow_probability.python.stats.sample_stats import auto_correlation
from tensorflow_probability.python.stats.sample_stats import cholesky_covariance
from tensorflow_probability.python.stats.sample_stats import correlation
from tensorflow_probability.python.stats.sample_stats import covariance
from tensorflow_probability.python.stats.sample_stats import stddev
from tensorflow_probability.python.stats.sample_stats import variance

# pylint: enable=unused-import,wildcard-import,line-too-long,g-importing-member

from tensorflow.python.util.all_util import remove_undocumented

__all__ = [
    'auto_correlation',
    'cholesky_covariance',
    'correlation',
    'covariance',
    'find_bins',
    'percentile',
    'quantiles',
    'stddev',
    'variance',
]

remove_undocumented(__name__, __all__)
