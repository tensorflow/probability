# Lint as: python3
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
"""System for dynamically rewriting modules to support multiple backends.

The theory of operation is as follows. In general, when requesting a module to
be imported Python first accesses finders in `sys.meta_path`, which assign a
loader to a module.

We hook into this system by providing a custom finder which trigers when a
certain module name is loaded, and then specifies a custom loader which
manipulates the source of the module before executing it. The source is
manipulated primarily to alter the import lines to point to a new backend
module.

By modifiying the loader, we make this process resilent to reloads via IPython's
`autoreload` magic, as well as `importlib.reload` function.

Concretely, modules of the name `fun_mcmc.dynamic.<backend>.<mod>` are specially
handled to:

1. Load the source from the `fun_mcmc.<mod>` module instead.
2. Rewrite the `from fun_mcmc import backend` imports to import
   `fun_mcmc.<backend>` instead.
3. Rewrite imports of the form `fun_mcmc.<mod2>` inside the modules to be
   `fun_mcmc.dynamic.<backend>.<mod2>`, which repeats this process for <mod2>.

The various name/filename manipulations are done relative to a 'root module'
which is `fun_mcmc` in this case.

As a special note, `fun_mcmc.dynamic.<backend>` modules already exist on the
filesystem to simplify logic in this module, and are not rewritten. This works
fine because the custom finder we install is only queried if the regular Python
finders fail to find anything.
"""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import re
import sys

# For debugging, you can turn this on which prints a large number of messages to
# stdout. We don't use regular absl.logging because most of this happens before
# it is set up in `main`.
DEBUG = False


def _root_name_comps():
  """Return the name components of the root module."""
  # This assumes `rewrite` is a top-level submodule of `fun_mcmc`.
  # If we move it, we'd change the `-1` to equal the (negative) nesting level.
  return __name__.split('.')[:-1]


def _root_name():
  """Return the name of the root module."""
  return '.'.join(_root_name_comps())


class Loader(importlib.abc.SourceLoader):
  """Custom loader which rewrites the source before loading it."""

  def __init__(self, orig_module_name, orig_loader, backend):
    self._backend = backend
    self._orig_module_name = orig_module_name
    self._orig_loader = orig_loader

  def get_filename(self, fullname):
    del fullname
    return self._orig_loader.get_filename(self._orig_module_name)

  def get_data(self, path):
    if DEBUG:
      print('Rewriting ', path)

    data = self._orig_loader.get_data(path).decode('utf-8')

    backend_name = self._backend.__name__.split('.')[-1]
    root_name = _root_name()

    if DEBUG:
      print('root_name ', root_name)
    re_backend = re.compile('from {} import backend'.format(root_name))
    re_import = re.compile('import {}'.format(root_name))
    re_from = re.compile('from {}'.format(root_name))

    lines = []
    for line in data.split('\n'):
      line, n = re_backend.subn(
          'from {} import {} as backend'.format(root_name, backend_name), line)
      if n == 0:
        line = re_import.sub(
            'import {}.dynamic.{}'.format(root_name, backend_name), line)
        line = re_from.sub('from {}.dynamic.{}'.format(root_name, backend_name),
                           line)
      lines.append(line)
    ret = '\n'.join(lines)
    if DEBUG:
      print(ret)
      print()
    return ret.encode('utf-8')


class Finder(importlib.abc.MetaPathFinder):
  """Custom finder which handles modules like `fun_mcmc.dynamic.<backend>`."""
  _FUN_MCMC_FINDER = True  # This is here so we can detect stale references to

  # this class in sys.meta_path.

  def find_spec(self, fullname, path, target=None):
    """See base class."""
    del target  # We don't use this hint.
    root_name_comps = _root_name_comps()
    root_name = _root_name() + '.dynamic'

    # Only handle things starting with .dynamic.
    if not fullname.startswith(root_name):
      return

    if DEBUG:
      print('fullname: ', fullname)
      print('path: ', path)

    # We cut out the leading components (including 'dynamic', hence the + 1),
    # leaving us with [<backend>, ...].
    module_name_comps = fullname.split('.')[len(root_name_comps) + 1:]

    if DEBUG:
      print('module_name_comps: ', module_name_comps)

    # This shouldn't really happen, but to be safe we don't handle this case
    # either. This would correspond to doing `import
    # fun_mcmc.dynamic.<backend>`, which doesn't need rewriting.
    if len(module_name_comps) < 2:
      return

    backend = importlib.import_module('.'.join(
        [_root_name(), module_name_comps[0]]))

    orig_module_name = '.'.join(root_name_comps + module_name_comps[1:])
    if DEBUG:
      print('backend: ', backend)
      print('orig_module_name: ', orig_module_name)
    orig_spec = importlib.util.find_spec(orig_module_name)
    if orig_spec is None:
      raise ImportError('Cannot import ' + orig_module_name)
    orig_loader = orig_spec.loader
    # We use duck-typing here because we don't necesarily need this to be a
    # SourceFileLoader, just that it has these methods.
    if not (hasattr(orig_loader, 'get_filename') and hasattr(
        orig_loader, 'is_package') and hasattr(orig_loader, 'get_data')):
      raise TypeError('{} has an abnormal loader: {}'.format(
          orig_module_name, orig_loader))

    spec = importlib.machinery.ModuleSpec(
        fullname,
        Loader(orig_module_name, orig_loader, backend),  # pylint: disable=abstract-class-instantiated
        origin=orig_loader.get_filename(orig_module_name),
        is_package=orig_loader.is_package(orig_module_name),
    )

    # We need to modify the spec after construction to set a few attributes.
    # This is allowed as per ModuleSpec docstring.
    if orig_loader.is_package(orig_module_name):
      # Otherwise importing from packages fails to work.
      spec.submodule_search_locations = [
          '.'.join([path[0], module_name_comps[-1]])
      ]
    # Helps with pdb integration.
    spec.has_location = True
    # We don't cache these rewritten modules.
    spec.cached = False
    if DEBUG:
      print()
    return spec


def enable_backends():
  """Enables the backends."""
  if DEBUG:
    print(sys.meta_path)

  # We try to be robust to reloading this module, and remove the old instance of
  # the Finder from sys.meta_path.
  found = False
  i = 0
  for (i, finder) in enumerate(sys.meta_path):
    if hasattr(finder, '_FUN_MCMC_FINDER'):
      found = True
      break

  if found:
    sys.meta_path[i] = Finder()
  else:
    # We insert it at the end, so the regular finders get first dibs. This is
    # significantly more robust than pre-pending it.
    sys.meta_path.append(Finder())


enable_backends()
