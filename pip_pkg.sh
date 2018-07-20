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
# ==============================================================================
set -e
set -x

PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"

function main() {
  if [ $# -lt 1 ] ; then
    echo "No destination dir provided"
    exit 1
  fi

  # Create the directory, then do dirname on a non-existent file inside it to
  # give us an absolute paths with tilde characters resolved to the destination
  # directory. Readlink -f is a cleaner way of doing this but is not available
  # on a fresh macOS install.
  mkdir -p "$1"
  DEST="$(dirname "${1}/does_not_exist")"
  echo "=== destination directory: ${DEST}"

  TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)

  echo $(date) : "=== Using tmpdir: ${TMPDIR}"

  echo "=== Copy TensorFlow Probability files"
  # Here is the bazel-bin/pip_pkg.runfiles directory structure.
  # bazel-bin/pip_pkg.runfiles
  # |- MANIFEST
  # |- tensorflow_probability
  #   |- pip_pkg
  #   |- pip_pkg.sh
  #   |- MANIFEST.in (needed)
  #   |- setup.py (needed)
  #   |- tensorflow_probability (needed)
  #
  # To build tensorflow probability wheel, we only need setup.py, MANIFEST.in,
  # .py and .so files under tensorflow_probability/tensorflow_probability, so we
  # extract those to ${TMPDIR}.
  cp bazel-bin/pip_pkg.runfiles/tensorflow_probability/setup.py "${TMPDIR}"
  cp bazel-bin/pip_pkg.runfiles/tensorflow_probability/MANIFEST.in "${TMPDIR}"
  cp -R \
    bazel-bin/pip_pkg.runfiles/tensorflow_probability/tensorflow_probability \
    "${TMPDIR}"

  echo "=== Copy TensorFlow Probability root files"
  cp README.md ${TMPDIR}
  cp LICENSE ${TMPDIR}

  pushd ${TMPDIR}
  echo $(date) : "=== Building wheel"

  # Pass through remaining arguments (following the first argument, which
  # specifies the output dir) to setup.py, e.g.,
  #  ./pip_pkg /tmp/tensorflow_probability_pkg --gpu --release
  # passes `--gpu --release` to setup.py.
  python setup.py bdist_wheel --universal ${@:2} >/dev/null

  cp dist/* "${DEST}"
  popd
  rm -rf ${TMPDIR}
  echo $(date) : "=== Output tar ball and wheel file are in: ${DEST}"
}

main "$@"
