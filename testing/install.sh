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
if [ -z "${TF_VERSION}" ]; then
  echo "TF_VERSION is unset."
  exit -1
fi

# NB: tensorflow pulls in our other deps, like numpy and six, transitively.
pip install "${TF_VERSION}"

# Some tests use methods from these libraries so we install them although
# they are not official dependencies of the TFP library. pylint is used for
# linting.
pip install scipy hypothesis matplotlib pylint

# Upgrade numpy to the latest to address issues that happen when testing with
# Python 3 (https://github.com/tensorflow/tensorflow/issues/16488).
pip install -U numpy

# Install Bazel for tests.
# Step 1: Install the JDK
sudo apt-get install openjdk-8-jdk

# Step 2: Add Bazel distribution URI as a package source
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" |
  sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -

# Step 3: Install and update Bazel
sudo apt-get update && sudo apt-get install bazel
