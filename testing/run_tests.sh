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

set -x  # print commands as they are executed
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

call_with_log_folding() {
  local command=$1
  echo "travis_fold:start:$command"
  $command
  echo "travis_fold:end:$command"
}

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
  sudo apt-get update
  sudo apt-get install bazel
}

install_python_packages() {
  # NB: tf-nightly pulls in other deps, like numpy, absl, and six, transitively.
  pip install tf-nightly

  # The following unofficial dependencies are used only by tests.
  pip install scipy hypothesis matplotlib mock

  # Install additional TFP dependencies.
  pip install decorator cloudpickle

  # Upgrade numpy to the latest to address issues that happen when testing with
  # Python 3 (https://github.com/tensorflow/tensorflow/issues/16488).
  pip install -U numpy
}

call_with_log_folding install_bazel
call_with_log_folding install_python_packages

# Get a shard of tests.
shard_tests=$(bazel query 'tests(//tensorflow_probability/...)' |
  awk -v n=${NUM_SHARDS} -v s=${SHARD} 'NR%n == s' )

# Run tests. Notes on less obvious options:
#   --notest_keep_going -- stop running tests as soon as anything fails. This is
#     to minimize load on Travis, where we share a limited number of concurrent
#     jobs with a bunch of other TensorFlow projects.
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
    --test_tag_filters=-gpu,-requires-gpu-sm35,-no-oss-ci \
    --test_timeout 300,450,1200,3600 \
    --action_env=PATH \
    --action_env=LD_LIBRARY_PATH \
    --test_output=errors
