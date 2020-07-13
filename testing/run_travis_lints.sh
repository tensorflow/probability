#!/usr/bin/env bash
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

set -v  # print commands as they are executed
set -e  # fail and exit on any command erroring

get_changed_py_files() {
  # Need to fetch the base branch in case it is not master.
  git remote set-branches --add origin ${TRAVIS_BRANCH}
  git fetch --depth=20 --quiet
  git diff \
      --name-only \
      --diff-filter=AM origin/${TRAVIS_BRANCH}...HEAD \
    | grep '^tensorflow_probability.*\.py$' || true
}

pip install --quiet pylint

# Run lints on added/changed python files.
changed_py_files=$(get_changed_py_files)
if [[ -n "${changed_py_files}" ]]; then
  echo "Running pylint on ${changed_py_files}"
  pylint -j2 --rcfile=testing/pylintrc ${changed_py_files}
else
  echo "No files to lint."
fi
