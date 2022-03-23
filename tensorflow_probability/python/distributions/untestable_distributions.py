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
"""Script reporting which Distributions are not being tested by Hypothesis."""

from tensorflow_probability.python.distributions import hypothesis_testlib  # pylint: disable=unused-import

# Don't need a run block, because the desired information comes from warnings
# emitted when `hypothesis_testlib` is loaded.
