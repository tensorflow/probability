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
"""Line-search optimizers package."""

from tensorflow_probability.python.internal import all_util
from tensorflow_probability.python.optimizer.linesearch.hager_zhang import hager_zhang
from tensorflow_probability.python.optimizer.linesearch.backtracking import backtracking


_allowed_symbols = [
    'hager_zhang',
    'backtracking',
]

all_util.remove_undocumented(__name__, _allowed_symbols)
