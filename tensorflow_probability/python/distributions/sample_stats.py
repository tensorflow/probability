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
"""Functions for computing statistics of samples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf
from tensorflow_probability.python import stats

__all__ = [
    "auto_correlation",
    "percentile",
]


auto_correlation_deprecator = tf.contrib.framework.deprecated(
    "2018-10-01",
    "auto_correlation is moved to the `stats` namespace.  Access it via: "
    "`tfp.stats.auto_correlation`.",
    warn_once=True)
auto_correlation = auto_correlation_deprecator(stats.auto_correlation)


percentile_deprecator = tf.contrib.framework.deprecated(
    "2018-10-01",
    "percentile is moved to the `stats` namespace.  Access it via: "
    "`tfp.stats.percentile`.",
    warn_once=True)
percentile = percentile_deprecator(stats.percentile)
