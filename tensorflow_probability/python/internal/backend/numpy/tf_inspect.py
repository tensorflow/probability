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
"""Reimplementation of tensorflow.python.util.tf_inspect."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
  from tensorflow.python.util import tf_inspect as inspect  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
except ImportError:
  import inspect  # pylint: disable=g-import-not-at-top


# Although `inspect` is different between Python 2 and 3, we should only ever
# be using Python 3's inspect because JAX is Python 3 only and if TF is present
# we will use `tf_inspect` which is compatible with both Python 2 and 3.
Parameter = inspect.Parameter
getfullargspec = inspect.getfullargspec
getcallargs = inspect.getcallargs
getframeinfo = inspect.getframeinfo
getdoc = inspect.getdoc
getfile = inspect.getfile
getmembers = inspect.getmembers
getmodule = inspect.getmodule
getmro = inspect.getmro
getsource = inspect.getsource
getsourcefile = inspect.getsourcefile
isbuiltin = inspect.isbuiltin
isclass = inspect.isclass
isfunction = inspect.isfunction
isframe = inspect.isframe
isgenerator = inspect.isgenerator
isgeneratorfunction = inspect.isgeneratorfunction
ismethod = inspect.ismethod
ismodule = inspect.ismodule
isroutine = inspect.isroutine
signature = inspect.signature
stack = inspect.stack
