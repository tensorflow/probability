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

# Include seconds since script execution as a prefix for all command echo logs.
PS4='+ $SECONDS s\011 '  # Octal 9, for tab
set -x  # print commands as they are executed
set -e  # fail and exit on any command erroring
set -u  # fail and exit on any undefined variable reference

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

find_version_str() {
  PKG_NAME=$1
  # These are nightly builds we'd like to avoid for some reason; separated by
  # regex OR operator.
  BAD_NIGHTLY_DATES="20200112\|20200113"
  # This will fail to find version 'X" and log available version strings to
  # stderr. We then sort, remove bad versions and take the last entry. This
  # allows us to avoid hardcoding the main version number, which would then need
  # to be updated on every new TF release.
  pip install $PKG_NAME==X 2>&1 \
    | grep -o "[0-9.]\+dev[0-9]\{8\}" \
    | sort \
    | grep -v "$BAD_NIGHTLY_DATES" \
    | tail -n1
}

install_python_packages() {
  # Ensure newer than 18.x pip version, which is necessary after tf-nightly
  # switched to manylinux2010.
  pip install --upgrade 'pip>=19.2'

  # NB: tf-nightly pulls in other deps, like numpy, absl, and six, transitively.
  TF_VERSION_STR=$(find_version_str tf-nightly)
  pip install tf-nightly==$TF_VERSION_STR

  # The following unofficial dependencies are used only by tests.
  # TODO(b/148685448): Unpin Hypothesis and coverage versions.
  pip install hypothesis==3.56.5 coverage==4.4.2 matplotlib mock scipy

  # Install additional TFP dependencies.
  pip install decorator cloudpickle

  # Upgrade numpy to the latest to address issues that happen when testing with
  # Python 3 (https://github.com/tensorflow/tensorflow/issues/16488).
  pip install -U numpy

  # Print out all versions, as an FYI in the logs.
  python --version
  pip --version
  pip freeze
}

call_with_log_folding install_bazel
call_with_log_folding install_python_packages

test_tags_to_skip="(gpu|requires-gpu-nvidia|notap|no-oss-ci|tfp_jax|tf2-broken|tf2-kokoro-broken)"

# Given a test size (small, medium, large), a number of shards and a shard ID,
# query and print a list of tests of the given size to run in the given shard.
query_and_shard_tests_by_size() {
  size=$1
  bazel_query="attr(size, ${size}, tests(//tensorflow_probability/...)) \
               except \
               attr(tags, \"${test_tags_to_skip}\", \
                    tests(//tensorflow_probability/...))"
  bazel query ${bazel_query} \
    | awk -v n=${NUM_SHARDS} -v s=${SHARD} 'NR%n == s'
}

# Generate a list of tests for this shard, consisting of a subset of tests of
# each size (small, medium and large). By evenly splitting the various test
# sizes across shards, we help ensure the shards have comparable runtimes.
sharded_tests="$(query_and_shard_tests_by_size small)"
sharded_tests="${sharded_tests} $(query_and_shard_tests_by_size medium)"
sharded_tests="${sharded_tests} $(query_and_shard_tests_by_size large)"

# Run tests. Notes on less obvious options:
#   --notest_keep_going -- stop running tests as soon as anything fails. This is
#     to minimize load on Travis, where we share a limited number of concurrent
#     jobs with a bunch of other TensorFlow projects.
#   --test_timeout -- comma separated values correspond to various test sizes
#     (short, moderate, long or eternal)
#   --action_env -- specify environment vars to pass through to action
#     environment. (We need these in order to run inside a virtualenv.)
#     See https://github.com/bazelbuild/bazel/issues/6648 and b/121259040.
echo "${sharded_tests}" \
  | xargs bazel test \
    --compilation_mode=opt \
    --copt=-O3 \
    --copt=-march=native \
    --notest_keep_going \
    --test_timeout 300,450,1200,3600 \
    --test_env=TFP_HYPOTHESIS_MAX_EXAMPLES=2 \
    --action_env=PATH \
    --action_env=LD_LIBRARY_PATH \
    --test_output=errors
