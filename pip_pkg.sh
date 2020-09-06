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

if [[ $# -lt 1 ]] ; then
  echo "Usage:"
  echo "pip_pkg /path/to/destination/directory"
  exit 1
fi

# Create the destination directory, then do dirname on a non-existent file
# inside it to give us a path with tilde characters resolved (readlink -f is
# another way of doing this but is not available on a fresh macOS install).
# Finally, use cd and pwd to get an absolute path, in case a relative one was
# given.
mkdir -p "$1"
DEST=$(dirname "${1}/does_not_exist")
DEST=$(cd "$DEST" && pwd)

# Use cp -L to resolve symlinks added as part of bazel build. Otherwise wheel
# does not follow them and fails to include TFP on JAX/Numpy.
pip_tmp="$(mktemp -d)"
cp -L -R bazel-bin/pip_pkg.runfiles/tensorflow_probability "${pip_tmp}"
cd "${pip_tmp}/tensorflow_probability"

echo >>tensorflow_probability/python/version.py \
  "if __version__.endswith('dev'): __version__ += '$(date --utc +%Y%m%d)'"
# Pass through remaining arguments (following the first argument, which
# specifies the output dir) to setup.py, e.g.,
#  ./pip_pkg /tmp/tensorflow_probability_pkg --gpu --release
# passes `--gpu --release` to setup.py.
python setup.py bdist_wheel --universal ${@:2} --dist-dir="$DEST" >/dev/null

set +x
echo -e "\nBuild complete. Wheel files are in $DEST"
