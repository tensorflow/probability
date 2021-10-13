#!/usr/bin/env bash
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

#
# This script creates a fresh virtualenv containing the TFP test dependencies
# (but not TFP itself).  This replicates the environment our continuous
# integration uses to test TFP.
#
# Caveat: This script relies on the system Python interpreter.  If you are
# hunting Python-version-dependent problems, you may want to use pyenv to
# install a specific Python version.

# Get the absolute path to the directory containing this script.
DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)

# Start a virtualenv
tmpdir=$(mktemp -d kokoro_virtualenv.XXXXXXXXXX --tmpdir)
echo "Creating virtual environment in ${tmpdir}"
(cd $tmpdir && virtualenv -p python3 .)
source "${tmpdir}/bin/activate"

# Install TFP dependencies in it
"${DIR}/install_test_dependencies.sh"

echo "To reproduce any desired test:"
echo "git clone https://github.com/tensorflow/probability && cd probability if you haven't already"
echo "source ${tmpdir}/bin/activate if needed"
echo "./testing/run_tfp_test.sh <your bazel target>"
