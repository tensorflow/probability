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
"""Framework for Bayesian structural time series models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_probability.python.sts.autoregressive import Autoregressive
from tensorflow_probability.python.sts.autoregressive import AutoregressiveStateSpaceModel
from tensorflow_probability.python.sts.decomposition import decompose_by_component
from tensorflow_probability.python.sts.decomposition import decompose_forecast_by_component
from tensorflow_probability.python.sts.dynamic_regression import DynamicLinearRegression
from tensorflow_probability.python.sts.dynamic_regression import DynamicLinearRegressionStateSpaceModel
from tensorflow_probability.python.sts.fitting import build_factored_variational_loss
from tensorflow_probability.python.sts.fitting import fit_with_hmc
from tensorflow_probability.python.sts.fitting import sample_uniform_initial_state
from tensorflow_probability.python.sts.forecast import forecast
from tensorflow_probability.python.sts.forecast import one_step_predictive
from tensorflow_probability.python.sts.internal.missing_values_util import MaskedTimeSeries
from tensorflow_probability.python.sts.local_level import LocalLevel
from tensorflow_probability.python.sts.local_level import LocalLevelStateSpaceModel
from tensorflow_probability.python.sts.local_linear_trend import LocalLinearTrend
from tensorflow_probability.python.sts.local_linear_trend import LocalLinearTrendStateSpaceModel
from tensorflow_probability.python.sts.regression import LinearRegression
from tensorflow_probability.python.sts.regression import SparseLinearRegression
from tensorflow_probability.python.sts.seasonal import ConstrainedSeasonalStateSpaceModel
from tensorflow_probability.python.sts.seasonal import Seasonal
from tensorflow_probability.python.sts.seasonal import SeasonalStateSpaceModel
from tensorflow_probability.python.sts.semilocal_linear_trend import SemiLocalLinearTrend
from tensorflow_probability.python.sts.semilocal_linear_trend import SemiLocalLinearTrendStateSpaceModel
from tensorflow_probability.python.sts.structural_time_series import StructuralTimeSeries
from tensorflow_probability.python.sts.sum import AdditiveStateSpaceModel
from tensorflow_probability.python.sts.sum import Sum

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    'AdditiveStateSpaceModel',
    'Autoregressive',
    'AutoregressiveStateSpaceModel',
    'ConstrainedSeasonalStateSpaceModel',
    'DynamicLinearRegression',
    'DynamicLinearRegressionStateSpaceModel',
    'LinearRegression',
    'LocalLevel',
    'LocalLevelStateSpaceModel',
    'LocalLinearTrend',
    'LocalLinearTrendStateSpaceModel',
    'MaskedTimeSeries',
    'Seasonal',
    'SeasonalStateSpaceModel',
    'SemiLocalLinearTrend',
    'SemiLocalLinearTrendStateSpaceModel',
    'SparseLinearRegression',
    'StructuralTimeSeries',
    'Sum',
    'build_factored_variational_loss',
    'decompose_by_component',
    'decompose_forecast_by_component',
    'fit_with_hmc',
    'forecast',
    'one_step_predictive',
    'sample_uniform_initial_state'
]

remove_undocumented(__name__, _allowed_symbols)
