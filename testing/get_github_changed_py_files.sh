#!/usr/bin/env bash
# Copyright 2022 The TensorFlow Probability Authors.
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

if [ $GITHUB_BASE_REF ]; then
  git fetch origin ${GITHUB_BASE_REF} --depth=1
  git diff \
      --name-only \
      --diff-filter=AM origin/${GITHUB_BASE_REF} \
    | { grep '^tensorflow_probability.*\.py$' || true; }
fi
