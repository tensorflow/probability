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

"""Central place to hold the `setup.py` dependencies.

This way, our developer automation shell scripts can read them.
"""

REQUIRED_PACKAGES = [
    'absl-py',
    'six>=1.10.0',
    'numpy>=1.13.3',
    'decorator',
    'cloudpickle>=1.3',
    'gast>=0.3.2',  # For autobatching
    'dm-tree',  # For NumPy/JAX backends (hence, also for prefer_static)
]

if __name__ == '__main__':
  # For using `pip` in shell scripts.
  print(' '.join(REQUIRED_PACKAGES))
