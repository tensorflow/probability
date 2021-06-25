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
# This script installs TFP test dependencies using pip, and does some sanity
# checks beforehand to make sure things look like they should work. Namely:
#   1. Make sure a virtualenv is active, unless user has explicitly asked not
#      to (by passing the --user flag to this script).
#   2. Make sure no invalid tensorflow packages are already installed:
#      a. no non-nightly versions (i.e. the stable `tensorflow` pip package)
#      b. if a nightly version is present, it is consistent with the presence
#         or absence of the --enable_gpu flag.
#
# Simple usage (from the root of the TFP repo, with a virtualenv active):
#
# ```bash
#   $ ./testing/install_python_packages.sh
# ```
#
# Usage outside of a virtualenv (not recommended):
#
# ```bash
#   $ ./testing/install_python_packages.sh --user
# ```
#
# The --user variant will simply forward the --user flag to pip install
# commands, telling pip to install the dependencies in the user install
# directory.

set -v  # print commands as they are executed
set -e  # fail and exit on any command erroring

# Get the absolute path to the directory containing this script.
DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)

virtualenv_is_active() {
  python ${DIR}/virtualenv_is_active.py
}

SCRIPT_ARGS=$@
user_flag_is_set() {
  # We could use getopts but it's annoying and arguably overkill here. We should
  # consider it if ever this script grows more complicated.
  [[ "$SCRIPT_ARGS" =~ --user ]]
}

enable_gpu_flag_is_set() {
  [[ "$SCRIPT_ARGS" =~ --enable_gpu ]]
}

if ! virtualenv_is_active && ! user_flag_is_set; then
  echo "Looks like you're not in a virtualenv. We strongly recommend running"
  echo "TFP tests inside of a virtualenv to avoid conflicts with preinstalled"
  echo "libraries. If you're sure you want to proceed outside of a virtualenv,"
  echo "rerun this script with the --user flag, which will indicate that we"
  echo "should install TFP dependencies using \`pip install --user ...\`"
  exit 1
elif ! virtualenv_is_active && user_flag_is_set; then
  PIP_FLAGS="--user"
else
  PIP_FLAGS=""
fi

if enable_gpu_flag_is_set; then
  TF_NIGHTLY_PACKAGE=tf-nightly
else
  TF_NIGHTLY_PACKAGE=tf-nightly-cpu
fi

PYTHON_PARSE_PACKAGE_JSON="
import sys
import json
import argparse
import sysconfig


parser = argparse.ArgumentParser(description='Parse pypi json')
parser.add_argument(
    '--bad_dates', type=str, nargs='*',
    help='Bad dates to be avoided. Space-separated')
args = parser.parse_args()

pypi_version_str = 'cp' + sysconfig.get_config_var('py_version_nodot')

package_data = json.loads(sys.stdin.read())
releases = []
for release, release_info in package_data['releases'].items():
  # Skip bad dates.
  if any(bad_date in release for bad_date in args.bad_dates):
    continue

  # Make sure there's a manylinux wheel file for given python_version.
  if not any(('manylinux' in wheel_info['filename'] and
              wheel_info['python_version'] in pypi_version_str)
             for wheel_info in release_info):
    continue
  releases.append(release)
print(sorted(releases)[-1])
"

find_good_tf_nightly_version_str() {
  curl -s https://pypi.org/pypi/tf-nightly/json \
    | python -c "$PYTHON_PARSE_PACKAGE_JSON" \
        --bad_dates 20210519 20210619
}

has_tensorflow_packages() {
  python -m pip list | grep -v tensorflow-metadata | grep tensorflow &> /dev/null
}

has_tf_nightly_cpu_package() {
  python -m pip list | grep tf-nightly-cpu &> /dev/null
}

has_tf_nightly_package() {
  python -m pip list | grep -v tf-nightly-cpu | grep tf-nightly &> /dev/null
}

check_for_common_package_conflicts() {
  if has_tensorflow_packages; then
    echo "Looks like you have a non-nightly version of tensorflow (or a \
          related non-nightly TF dependency like tensorflow-estimator). We \
          recommend purging all non-nightly versions from your virtualenv \
          before trying to run TFP tests. The safest way to do this is usually \
          to deactivate and delete your current virtualenv, create and \
          activate a new one, then rerun this script. These are the offending \
          packages:"
    echo
    python -m pip list | grep tensorflow
    exit 1
  fi

  if enable_gpu_flag_is_set && has_tf_nightly_cpu_package; then
    echo "Looks like you have requested to install TF with GPU support, but \
          also have tf-nightly-cpu installed. Having these present \
          may not have the intended effect: which one gets used will depend on \
          the order in which they were installed. We recommend removing the \
          version you don't want used for testing. The safest way to do this \
          is usually to deactivate and delete your current virtualenv, create \
          and activate a new one, then rerun this script."
    exit 1
  fi

  if ! enable_gpu_flag_is_set && has_tf_nightly_package; then
    echo "Looks like you have a version of TF installed which includes GPU, \
          but you haven't explictly run this script with the --enable_gpu \
          flag. This is usually a mistake, so we're letting you know. If this \
          was indeed a mistake, you can simply uninstall and rerun this \
          script. If you *did* intend to run tests against the GPU-enabled TF \
          installation, you'll just need to run this script with the explicit \
          --enable_gpu flag to proceed. NOTE: you will probably also need to \
          invoke tests using the --jobs=1 flag to \`bazel\`, otherwise it will \
          try to run lots of jobs in parallel and they will all try to claim \
          all the GPU RAM simultaneously."
    exit 1
  fi
}

install_tensorflow() {
  # NB: tf-nightly pulls in other deps, like numpy, absl, and six, transitively.
  TF_VERSION_STR=$(find_good_tf_nightly_version_str $TF_NIGHTLY_PACKAGE)
  python -m pip install $PIP_FLAGS $TF_NIGHTLY_PACKAGE==$TF_VERSION_STR
}

install_jax() {
  # For the JAX backend.
  python -m pip install jax jaxlib
}

install_python_packages() {
  install_tensorflow
  install_jax

  # The following unofficial dependencies are used only by tests.
  # TODO(b/148685448): Unpin Hypothesis and coverage versions.
  python -m pip install $PIP_FLAGS hypothesis==3.56.5 coverage==4.4.2 matplotlib mock mpmath scipy

  # Install additional TFP dependencies.
  python -m pip install $PIP_FLAGS decorator 'cloudpickle>=1.3' dm-tree

  # Print out all versions, as an FYI in the logs.
  python --version
  python -m pip --version
  python -m pip list
}

# Ensure newer than 18.x pip version, which is necessary after tf-nightly
# switched to manylinux2010.
python -m pip install $PIP_FLAGS --upgrade 'pip>=19.2'
python -m pip install --upgrade setuptools

check_for_common_package_conflicts
install_python_packages
