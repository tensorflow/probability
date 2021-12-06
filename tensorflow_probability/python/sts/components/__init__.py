# Copyright 2021 The TensorFlow Probability Authors.
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
"""Components for Bayesian structural time series models."""


from tensorflow_probability.python.internal import all_util
from tensorflow_probability.python.sts.components.autoregressive import Autoregressive
from tensorflow_probability.python.sts.components.autoregressive import AutoregressiveStateSpaceModel
from tensorflow_probability.python.sts.components.autoregressive_moving_average import AutoregressiveMovingAverageStateSpaceModel
from tensorflow_probability.python.sts.components.dynamic_regression import DynamicLinearRegression
from tensorflow_probability.python.sts.components.dynamic_regression import DynamicLinearRegressionStateSpaceModel
from tensorflow_probability.python.sts.components.local_level import LocalLevel
from tensorflow_probability.python.sts.components.local_level import LocalLevelStateSpaceModel
from tensorflow_probability.python.sts.components.local_linear_trend import LocalLinearTrend
from tensorflow_probability.python.sts.components.local_linear_trend import LocalLinearTrendStateSpaceModel
from tensorflow_probability.python.sts.components.regression import LinearRegression
from tensorflow_probability.python.sts.components.regression import SparseLinearRegression
from tensorflow_probability.python.sts.components.seasonal import ConstrainedSeasonalStateSpaceModel
from tensorflow_probability.python.sts.components.seasonal import Seasonal
from tensorflow_probability.python.sts.components.seasonal import SeasonalStateSpaceModel
from tensorflow_probability.python.sts.components.semilocal_linear_trend import SemiLocalLinearTrend
from tensorflow_probability.python.sts.components.semilocal_linear_trend import SemiLocalLinearTrendStateSpaceModel
from tensorflow_probability.python.sts.components.smooth_seasonal import SmoothSeasonal
from tensorflow_probability.python.sts.components.smooth_seasonal import SmoothSeasonalStateSpaceModel
from tensorflow_probability.python.sts.components.sum import AdditiveStateSpaceModel
from tensorflow_probability.python.sts.components.sum import Sum


_allowed_symbols = [
    'AdditiveStateSpaceModel',
    'Autoregressive',
    'AutoregressiveStateSpaceModel',
    'AutoregressiveMovingAverageStateSpaceModel',
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
    'SmoothSeasonal',
    'SmoothSeasonalStateSpaceModel',
    'SparseLinearRegression',
    'StructuralTimeSeries',
    'Sum',
]

all_util.remove_undocumented(__name__, _allowed_symbols)
