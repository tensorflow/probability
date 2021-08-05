# Copyright 2020 The TensorFlow Probability Authors.
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
"""Anomaly detection with structural time series models."""

from tensorflow_probability.python.internal import all_util
from tensorflow_probability.python.sts.anomaly_detection.anomaly_detection_lib import detect_anomalies
from tensorflow_probability.python.sts.anomaly_detection.anomaly_detection_lib import plot_predictions
from tensorflow_probability.python.sts.anomaly_detection.anomaly_detection_lib import PredictionOutput

_allowed_symbols = [
    'detect_anomalies',
    'plot_predictions',
    'PredictionOutput'
]

all_util.remove_undocumented(__name__, _allowed_symbols)

