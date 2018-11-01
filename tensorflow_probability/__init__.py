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
"""Tools for probabilistic reasoning in TensorFlow."""

# Contributors to the `python/` dir should not alter this file; instead update
# `python/__init__.py` as necessary.


# We need to put some imports inside a function call below, and the function
# call needs to come before the *actual* imports that populate the
# tensorflow_probability namespace. Hence, we disable this lint check throughout
# the file.
#
# pylint: disable=g-import-not-at-top


# Ensure TensorFlow is importable and its version is sufficiently recent. This
# needs to happen before anything else, since the imports below will try to
# import tensorflow, too.
def _ensure_tf_install():  # pylint: disable=g-statement-before-imports
  """Attempt to import tensorflow, and ensure its version is sufficient.

  Raises:
    ImportError: if either tensorflow is not importable or its version is
    inadequate.
  """
  try:
    import tensorflow as tf
  except ImportError:
    # Re-raise with more informative error message.
    raise ImportError(
        "Failed to import TensorFlow. Please note that TensorFlow is not "
        "installed by default when you install TensorFlow Probability. This is "
        "so that users can decide whether to install the GPU-enabled "
        "TensorFlow package. To use TensorFlow Probability, please install the "
        "most recent version of TensorFlow, by following instructions at "
        "https://tensorflow.org/install.")

  import distutils.version

  #
  # Update this whenever we need to depend on a newer TensorFlow release.
  #
  required_tensorflow_version = "1.11.0"

  if (distutils.version.LooseVersion(tf.__version__) <
      distutils.version.LooseVersion(required_tensorflow_version)):
    raise ImportError(
        "This version of TensorFlow Probability requires TensorFlow "
        "version >= {required}; Detected an installation of version {present}. "
        "Please upgrade TensorFlow to proceed.".format(
            required=required_tensorflow_version,
            present=tf.__version__))


_ensure_tf_install()


# Cleanup symbols to avoid polluting namespace.
import sys as _sys
for symbol in ["_ensure_tf_install", "_sys"]:
  delattr(_sys.modules[__name__], symbol)


# from tensorflow_probability.google import staging  # DisableOnExport
from tensorflow_probability.python import *  # pylint: disable=wildcard-import
from tensorflow_probability.python.version import __version__
# pylint: enable=g-import-not-at-top
