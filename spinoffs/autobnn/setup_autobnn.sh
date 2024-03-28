#!/usr/bin/env bash
# Copyright 2023 The TensorFlow Probability Authors.
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

set -v  # print commands as they are executed
set -e  # fail and exit on any command erroring

user_flag_is_set() {
  # We could use getopts but it's annoying and arguably overkill here. We should
  # consider it if ever this script grows more complicated.
  [[ "$SCRIPT_ARGS" =~ --user ]]
}

if user_flag_set; then
  PIP_FLAGS="--user"
else
  PIP_FLAGS=""
fi

python -m pip install $PIP_FLAGS bayeux-ml chex flax jaxtyping matplotlib pandas scipy
