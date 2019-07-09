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

from tensorflow_probability.python.stats.leave_one_out import log_loomean_exp
from tensorflow_probability.python.stats.leave_one_out import log_loosum_exp
from tensorflow_probability.python.stats.leave_one_out import log_soomean_exp
from tensorflow_probability.python.stats.leave_one_out import log_soosum_exp
from tensorflow_probability.python.stats.quantiles import count_integers
from tensorflow_probability.python.stats.quantiles import find_bins
from tensorflow_probability.python.stats.quantiles import histogram
from tensorflow_probability.python.stats.quantiles import percentile
from tensorflow_probability.python.stats.quantiles import quantiles
from tensorflow_probability.python.stats.sample_stats import auto_correlation
from tensorflow_probability.python.stats.sample_stats import cholesky_covariance
from tensorflow_probability.python.stats.sample_stats import correlation
from tensorflow_probability.python.stats.sample_stats import covariance
from tensorflow_probability.python.stats.sample_stats import stddev
from tensorflow_probability.python.stats.sample_stats import variance

# pylint: enable=unused-import,wildcard-import,line-too-long,g-importing-member

from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'auto_correlation',
    'cholesky_covariance',
    'correlation',
    'count_integers',
    'covariance',
    'find_bins',
    'histogram',
    'log_loomean_exp',
    'log_loosum_exp',
    'log_soomean_exp',
    'log_soosum_exp',
    'percentile',
    'quantiles',
    'stddev',
    'variance',
]

remove_undocumented(__name__, __all__)
