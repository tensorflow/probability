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

# Make sure the environment variables are set.
if [ -z "${SHARD}" ]; then
  echo "SHARD is unset."
  exit -1
fi

if [ -z "${NUM_SHARDS}" ]; then
  echo "NUM_SHARDS is unset."
  exit -1
fi

install_bazel() {
  # Install Bazel for tests. Based on instructions at
  # https://docs.bazel.build/versions/master/install-ubuntu.html#install-on-ubuntu
  # (We skip the openjdk8 install step, since travis lets us have that by
  # default).

  # Add Bazel distribution URI as a package source
  echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" \
    | sudo tee /etc/apt/sources.list.d/bazel.list
  curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -

  # Update apt and install bazel (use -qq to minimize log cruft)
  sudo apt-get update -qq
  sudo apt-get install bazel -qq
}

install_python_packages() {
  # Some tests use methods from these libraries so we install them although
  # they're not official dependencies of the TFP library. pylint is used for
  # linting.
  # NB: tensorflow pulls in our other deps, like numpy and six, transitively.
  pip install --quiet tf-nightly scipy hypothesis matplotlib

  # Upgrade numpy to the latest to address issues that happen when testing with
  # Python 3 (https://github.com/tensorflow/tensorflow/issues/16488).
  pip install -U numpy
}

# Do these in parallel
install_bazel &
install_python_packages &
wait

# Get a shard of tests.
shard_tests=$(bazel query 'tests(//tensorflow_probability/...)' |
  awk -v n=${NUM_SHARDS} -v s=${SHARD} 'NR%n == s' )

# Run tests. Notes on less obvious options:
#   --build_tests_only -- only build test targets and dependencies, instead of
#     everything captured by "//tensorflow_probability/..." (this *doesn't* mean
#     "build the tests but don't run them" which is how it sounds)
#   --test_timeout -- comma separated values correspond to various test sizes
#     (short, moderate, long or eternal)
#   --test_tag_filters -- skip tests whose 'tags' arg (if present) includes any
#     of the comma-separated entries
#   --action_env -- specify environment vars to pass through to action
#     environment. (We need these in order to run inside a virtualenv.)
#     See https://github.com/bazelbuild/bazel/issues/6648 and b/121259040.
echo "${shard_tests}" \
  | xargs bazel test \
    --copt=-O3 \
    --copt=-march=native \
    --notest_keep_going \
    --test_tag_filters=-gpu,-requires-gpu-sm35 \
    --test_timeout 300,450,1200,3600 \
    --build_tests_only \
    --action_env=PATH \
    --action_env=LD_LIBRARY_PATH \
    --test_output=errors
