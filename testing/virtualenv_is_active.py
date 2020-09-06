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
"""A script to detect if we're inside a virtualenv."""
import sys

# This is apparently the most reliable way to check whether the script is
# being run in a virtualenv. Source:
# https://stackoverflow.com/questions/1871549/determine-if-python-is-running-inside-virtualenv
if (hasattr(sys, 'real_prefix') or
    (hasattr(sys, 'base_prefix') and sys.prefix != sys.base_prefix)):
  sys.exit(0)
else:
  sys.exit(1)
