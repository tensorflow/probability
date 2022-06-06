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
"""Backend-specific utilities."""

import contextlib

__all__ = [
    'silence_nonrewritten_import_errors',
]

BACKEND = None  # Rewritten by backends/rewrite.py.


@contextlib.contextmanager
def silence_nonrewritten_import_errors():
  """Context manager to silence import errors if `BACKEND is None`.

  Sometimes un-rewritten (i.e. `BACKEND is None`) modules must be executed.
  Doing so can cause unintended external imports to be triggered. This context
  manager bluntly silences such errors, leaving a partially broken module in its
  wake. This is usually fine, since in such cases we won't actually surface the
  broken module to the user (and if an advanced user does actually need those
  modules to function, they can resolve the import errors to unbreak the
  module).

  The primary use case for this module is in `__init__.py` files which re-export
  API.

  Yields:
    Nothing.
  """
  if BACKEND is None:
    try:
      yield
    except ImportError:
      pass
  else:
    yield
