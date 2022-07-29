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

# This file contains functions common between developer-facing and
# continuous-integration-tool-facing environment setups for TFP.

# TODO(b/191965268): Move this to a proper python script, maybe even test it.
PYTHON_PARSE_PACKAGE_JSON="
import re
import sys
import json
import argparse
import pkg_resources
import sysconfig


parser = argparse.ArgumentParser(description='Parse pypi json')
parser.add_argument(
    '--bad_dates', type=str, nargs='*',
    help='Bad dates to be avoided. Space-separated')
args = parser.parse_args()

pypi_version_str = 'cp' + sysconfig.get_config_var('py_version_nodot')
release_pattern = re.compile('.+dev[0-9]+$')

package_data = json.loads(sys.stdin.read())
releases = []
for release, release_info in package_data['releases'].items():
  # Skip bad releases like '2.11.0'.
  if not release_pattern.match(release):
    continue

  # Skip bad dates.
  if any(bad_date in release for bad_date in args.bad_dates):
    continue

  # Make sure there's a manylinux wheel file for given python_version.
  if not any(('manylinux' in wheel_info['filename'] and
              wheel_info['python_version'] in pypi_version_str)
             for wheel_info in release_info):
    continue
  releases.append(release)
print(sorted(releases, key=pkg_resources.parse_version)[-1])
"

find_good_tf_nightly_version_str() {
  VERSION=$1
  curl -s "https://pypi.org/pypi/${VERSION}/json" \
    | python -c "$PYTHON_PARSE_PACKAGE_JSON" \
        --bad_dates 20210519 20210619 20220727
}

install_tensorflow() {
  TF_NIGHTLY_PACKAGE=$1
  PIP_FLAGS=${2-}
  # NB: tf-nightly pulls in other deps, like numpy, absl, and six, transitively.
  TF_VERSION_STR=$(find_good_tf_nightly_version_str $TF_NIGHTLY_PACKAGE)
  python -m pip install $PIP_FLAGS $TF_NIGHTLY_PACKAGE==$TF_VERSION_STR
}

install_jax() {
  # For the JAX backend.
  PIP_FLAGS=${1-}
  python -m pip install $PIP_FLAGS jax jaxlib
}

install_common_packages() {
  # Install additional TFP dependencies (other than TensorFlow or JAX).
  PIP_FLAGS=${1-}
  ROOT=$(cd $(dirname $(dirname "${BASH_SOURCE[0]}")) && pwd)
  python -m pip install $PIP_FLAGS $(python "${ROOT}/required_packages.py")
}

install_test_only_packages() {
  # The following unofficial dependencies are used only by tests.
  PIP_FLAGS=${1-}
  python -m pip install $PIP_FLAGS hypothesis matplotlib mock mpmath scipy pandas optax holidays
}

dump_versions() {
  python --version
  python -m pip --version
  python -m pip list
}
