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
# Define convenient alias to the testing/run_tfp_lints.sh script. Source this
# file in your terminal session to define the alias `tfp_lints`:
#
# ```bash
# source testing/define_linting_alias.sh
# alias tfp_lints
# # ==> /absolute/path/to/testing/run_tfp_lints.sh
# ```
#
# Optionally, provide a positional arguemnt to this script to create an
# alternate alias name:
#
# ```bash
# source testing/define_linting_alias.sh my_tfp_lints_alias
# alias my_tfp_lints_alias
# # ==> /absolute/path/to/testing/run_tfp_lints.sh

# Get the absolute path to the directory containing this script
DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)

# Read alias name from the first argument to the script, with a default of
# `tfp_lints`.
ALIAS_NAME=${1:-tfp_lints}

alias $ALIAS_NAME="$DIR/testing/run_tfp_lints.sh"
