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

from tensorflow_probability import substrates
# from tensorflow_probability.google import staging  # DisableOnExport
# from tensorflow_probability.google import tfp_google  # DisableOnExport
from tensorflow_probability.python import *  # pylint: disable=wildcard-import
from tensorflow_probability.python.version import __version__

# tfp_google.bind(globals())  # DisableOnExport
# del tfp_google  # DisableOnExport
