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

# Get the absolute path to the directory containing this script.
DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)

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
  sudo apt-get update
  sudo apt-get install bazel
}

install_python_packages() {
  ${DIR}/install_test_dependencies.sh
}

# Only install bazel if not already present (useful for locally testing this
# script).
which bazel || install_bazel
install_python_packages

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

# Run tests using run_tfp_test.sh script.
echo "${sharded_tests}" \
  | xargs $DIR/run_tfp_test.sh
