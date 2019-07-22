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
"""TensorFlow Probability auto-batching package.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Wildcard import is appropriate here because the purpose is to re-export all
# auto_batching symbols so they can be imported as from
# tensorflow_probability.experimental import auto_batching.
from tensorflow_probability.python.internal.auto_batching import *  # pylint: disable=wildcard-import
