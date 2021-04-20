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

from tensorflow_probability.python.internal import all_util
from tensorflow_probability.python.stats.calibration import brier_decomposition
from tensorflow_probability.python.stats.calibration import brier_score
from tensorflow_probability.python.stats.calibration import expected_calibration_error
from tensorflow_probability.python.stats.calibration import expected_calibration_error_quantiles
from tensorflow_probability.python.stats.kendalls_tau import iterative_mergesort
from tensorflow_probability.python.stats.kendalls_tau import kendalls_tau
from tensorflow_probability.python.stats.kendalls_tau import lexicographical_indirect_sort
from tensorflow_probability.python.stats.leave_one_out import log_loomean_exp
from tensorflow_probability.python.stats.leave_one_out import log_loosum_exp
from tensorflow_probability.python.stats.leave_one_out import log_soomean_exp
from tensorflow_probability.python.stats.leave_one_out import log_soosum_exp
from tensorflow_probability.python.stats.moving_stats import assign_log_moving_mean_exp
from tensorflow_probability.python.stats.moving_stats import assign_moving_mean_variance
from tensorflow_probability.python.stats.moving_stats import moving_mean_variance_zero_debiased
from tensorflow_probability.python.stats.quantiles import count_integers
from tensorflow_probability.python.stats.quantiles import find_bins
from tensorflow_probability.python.stats.quantiles import histogram
from tensorflow_probability.python.stats.quantiles import percentile
from tensorflow_probability.python.stats.quantiles import quantiles
from tensorflow_probability.python.stats.ranking import quantile_auc
from tensorflow_probability.python.stats.sample_stats import auto_correlation
from tensorflow_probability.python.stats.sample_stats import cholesky_covariance
from tensorflow_probability.python.stats.sample_stats import correlation
from tensorflow_probability.python.stats.sample_stats import covariance
from tensorflow_probability.python.stats.sample_stats import cumulative_variance
from tensorflow_probability.python.stats.sample_stats import log_average_probs
from tensorflow_probability.python.stats.sample_stats import stddev
from tensorflow_probability.python.stats.sample_stats import variance

# pylint: enable=unused-import,wildcard-import,line-too-long,g-importing-member


__all__ = [
    'assign_log_moving_mean_exp',
    'assign_moving_mean_variance',
    'auto_correlation',
    'brier_decomposition',
    'brier_score',
    'cholesky_covariance',
    'correlation',
    'count_integers',
    'covariance',
    'cumulative_variance',
    'expected_calibration_error',
    'expected_calibration_error_quantiles',
    'find_bins',
    'histogram',
    'iterative_mergesort',
    'kendalls_tau',
    'lexicographical_indirect_sort',
    'log_average_probs',
    'log_loomean_exp',
    'log_loosum_exp',
    'log_soomean_exp',
    'log_soosum_exp',
    'moving_mean_variance_zero_debiased',
    'percentile',
    'quantile_auc',
    'quantiles',
    'stddev',
    'variance',
]

all_util.remove_undocumented(__name__, __all__)
