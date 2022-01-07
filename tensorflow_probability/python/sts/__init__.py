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
"""Framework for Bayesian structural time series models.

See the [blog post](
https://blog.tensorflow.org/2019/03/structural-time-series-modeling-in.html)
for an introductory example.
"""

from tensorflow_probability.python.internal import all_util
from tensorflow_probability.python.sts.components import *
from tensorflow_probability.python.sts.decomposition import decompose_by_component
from tensorflow_probability.python.sts.decomposition import decompose_forecast_by_component
from tensorflow_probability.python.sts.fitting import build_factored_surrogate_posterior
from tensorflow_probability.python.sts.fitting import fit_with_hmc
from tensorflow_probability.python.sts.fitting import sample_uniform_initial_state
from tensorflow_probability.python.sts.forecast import forecast
from tensorflow_probability.python.sts.forecast import impute_missing_values
from tensorflow_probability.python.sts.forecast import one_step_predictive
from tensorflow_probability.python.sts.internal.missing_values_util import MaskedTimeSeries
from tensorflow_probability.python.sts.regularization import MissingValuesTolerance
from tensorflow_probability.python.sts.regularization import regularize_series
from tensorflow_probability.python.sts.structural_time_series import StructuralTimeSeries


_allowed_symbols = [
    'AdditiveStateSpaceModel',
    'Autoregressive',
    'AutoregressiveIntegratedMovingAverage',
    'AutoregressiveMovingAverageStateSpaceModel',
    'AutoregressiveStateSpaceModel',
    'ConstrainedSeasonalStateSpaceModel',
    'DynamicLinearRegression',
    'DynamicLinearRegressionStateSpaceModel',
    'IntegratedStateSpaceModel',
    'LinearRegression',
    'LocalLevel',
    'LocalLevelStateSpaceModel',
    'LocalLinearTrend',
    'LocalLinearTrendStateSpaceModel',
    'MaskedTimeSeries',
    'MissingValuesTolerance',
    'Seasonal',
    'SeasonalStateSpaceModel',
    'SemiLocalLinearTrend',
    'SemiLocalLinearTrendStateSpaceModel',
    'SmoothSeasonal',
    'SmoothSeasonalStateSpaceModel',
    'SparseLinearRegression',
    'StructuralTimeSeries',
    'Sum',
    'build_factored_surrogate_posterior',
    'decompose_by_component',
    'decompose_forecast_by_component',
    'fit_with_hmc',
    'forecast',
    'impute_missing_values',
    'one_step_predictive',
    'regularize_series',
    'sample_uniform_initial_state'
]

all_util.remove_undocumented(__name__, _allowed_symbols)
