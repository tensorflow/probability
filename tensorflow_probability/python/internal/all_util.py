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
"""Prune module attributes (API) down to a specified list (e.g. `__all__`)."""

import sys


_HIDDEN_ATTRIBUTES = {}


def remove_undocumented(module_name, allowed_symbols):
  """Removes symbols in a module that are not referenced by a docstring.

  Args:
    module_name: the name of the module (usually `__name__`).
    allowed_symbols: which symbols are allowed for this module.

  Returns:
    None
  """
  current_symbols = set(dir(sys.modules[module_name]))
  extra_symbols = current_symbols - set(allowed_symbols)
  target_module = sys.modules[module_name]
  for extra_symbol in extra_symbols:
    # Skip over __file__, etc. Also preserves internal symbols.
    if extra_symbol.startswith('_'): continue
    fully_qualified_name = module_name + '.' + extra_symbol
    _HIDDEN_ATTRIBUTES[fully_qualified_name] = (target_module,
                                                getattr(target_module,
                                                        extra_symbol))
    delattr(target_module, extra_symbol)
