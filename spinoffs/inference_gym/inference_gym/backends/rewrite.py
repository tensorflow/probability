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

Concretely, modules of the name `inference_gym.dynamic.<backend>.<mod>` are
specially handled to:

1. Load the source from the `inference_gym.<mod>` module instead.
2. Rewrite external imports, like TensorFlow and TensorFlow Probability.
3. Rewrite imports of the form `inference_gym.<mod2>` inside the modules to be
   `inference_gym.dynamic.<backend>.<mod2>`, which repeats this process
   for <mod2>.
4. Rewrite `BACKEND = None` to `BACKEND = '<backend>'.

The various name/filename manipulations are done relative to a 'root module'
which is `inference_gym` in this case.

As a special note, `inference_gym.dynamic.backend_<backend>` modules already
exist on the filesystem to simplify logic in this module, and are not rewritten.
This works fine because the custom finder we install is only queried if the
regular Python finders fail to find anything.
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


def _external_rewrites(backend, data):
  """Rewrite external imports."""
  # TFP
  if backend == 'backend_tensorflow':
    sub = 'import tensorflow_probability as tfp'
  elif backend == 'backend_jax':
    sub = 'import tensorflow_probability.substrates.jax as tfp'
  elif backend == 'backend_numpy':
    sub = 'import tensorflow_probability.substrates.numpy as tfp'
  else:
    raise ValueError(f'Unknown backend {backend}')
  data = re.sub('import tensorflow_probability as tfp', sub, data)

  # TensorFlow
  if backend == 'backend_tensorflow':
    sub = 'import tensorflow.compat.v2 as tf'
  elif backend == 'backend_jax':
    sub = ('from tensorflow_probability.substrates import jax as _tfp_jax; tf ='
           ' _tfp_jax.tf2jax; del _tfp_jax')
  elif backend == 'backend_numpy':
    sub = ('from tensorflow_probability.substrates import numpy as _tfp_numpy; '
           'tf = _tfp_numpy.tf2numpy; del _tfp_numpy')
  else:
    raise ValueError(f'Unknown backend {backend}')
  data = re.sub('import tensorflow.compat.v2 as tf', sub, data)

  # TFP's direct imports
  if backend == 'backend_tensorflow':
    sub = 'from tensorflow_probability.python'
  elif backend == 'backend_jax':
    sub = 'from tensorflow_probability.substrates.jax'
  elif backend == 'backend_numpy':
    sub = 'from tensorflow_probability.substrates.numpy'
  else:
    raise ValueError(f'Unknown backend {backend}')
  data = re.sub('from tensorflow_probability.python',
                sub, data)

  # TF's internal nest module
  if backend == 'backend_tensorflow':
    sub = 'from tensorflow.python.util import nest'
  elif backend == 'backend_jax':
    sub = 'from tensorflow_probability.python.internal.backend.jax import nest'
  elif backend == 'backend_numpy':
    sub = (
        'from tensorflow_probability.python.internal.backend.numpy import nest')
  else:
    raise ValueError(f'Unknown backend {backend}')
  data = re.sub('from tensorflow.python.util import nest',
                sub, data)

  return data


def _root_name_comps():
  """Return the name components of the root module."""
  # This assumes `rewrite` is a 2 levels deep inside of `inference_gym`. If we
  # move it, we'd change the `-2` to equal the (negative) nesting level.
  return __name__.split('.')[:-2]


def _root_name():
  """Return the name of the root module."""
  return '.'.join(_root_name_comps())


class Loader(importlib.abc.SourceLoader):
  """Custom loader which rewrites the source before loading it."""

  def __init__(self, orig_module_name, orig_loader, orig_filename,
               backend_name):
    self._backend_name = backend_name
    self._orig_module_name = orig_module_name
    self._orig_loader = orig_loader
    self._orig_filename = orig_filename

  def get_filename(self, fullname):
    del fullname
    return self._orig_filename

  def get_data(self, path):
    if DEBUG:
      print('Rewriting ', path)

    data = self._orig_loader.get_data(path).decode('utf-8')

    root_name = _root_name()

    if DEBUG:
      print('root_name ', root_name)

    data = re.sub(
        'import {}'.format(root_name),
        'import {}.dynamic.{}'.format(root_name, self._backend_name),
        data,
    )
    data = re.sub(
        'from {}'.format(root_name),
        'from {}.dynamic.{}'.format(root_name, self._backend_name),
        data,
    )
    data = re.sub(
        'BACKEND = None',
        'BACKEND = \'{}\''.format(self._backend_name),
        data,
    )
    data = _external_rewrites(self._backend_name, data)

    if DEBUG:
      print(data)
      print()
    return data.encode('utf-8')


class Finder(importlib.abc.MetaPathFinder):
  """Custom finder for the dynamic rewrite system.

  It handles modules like `inference_gym.dynamic.<backend>`.
  """
  # This is here so we can detect stale references to this class in
  # sys.meta_path.
  _INFERENCE_GYM_FINDER = True

  def find_spec(self, fullname, path, target=None):
    """See base class."""
    del target  # We don't use this hint.
    root_name_comps = _root_name_comps()
    root_name = _root_name() + '.dynamic'

    if DEBUG:
      print('candidate: ', fullname, path)
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
    # inference_gym.dynamic.<backend>`, which doesn't need rewriting.
    if len(module_name_comps) < 2:
      return

    backend_name = module_name_comps[0]
    orig_module_name = '.'.join(root_name_comps + module_name_comps[1:])
    if DEBUG:
      print('backend: ', backend_name)
      print('orig_module_name: ', orig_module_name)
    # N.B. this will actually execute package __init__.py files. If those import
    # backend-specific modules, those imports will fail. That's why the
    # __init__.py files in question disable import errors using the
    # backends.util.silence_nonrewritten_import_errors utility.
    orig_spec = importlib.util.find_spec(orig_module_name)
    if orig_spec is None:
      raise ImportError('Cannot import ' + orig_module_name)
    is_package = bool(orig_spec.submodule_search_locations)
    orig_loader = orig_spec.loader
    # We use duck-typing here because we don't necesarily need this to be a
    # SourceFileLoader, just that it has this method.
    if not hasattr(orig_loader, 'get_data'):
      raise TypeError('{} has an unsupported loader: {}'.format(
          orig_module_name, orig_loader))

    spec = importlib.machinery.ModuleSpec(
        fullname,
        Loader(orig_module_name, orig_loader, orig_spec.origin, backend_name),  # pylint: disable=abstract-class-instantiated
        origin=orig_spec.origin,
        is_package=is_package,
    )

    # We need to modify the spec after construction to set a few attributes.
    # This is allowed as per ModuleSpec docstring.
    if is_package:
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
    print('sys.meta_path: ', sys.meta_path)

  # We try to be robust to reloading this module, and remove the old instance of
  # the Finder from sys.meta_path.
  found = False
  i = 0
  for (i, finder) in enumerate(sys.meta_path):
    if hasattr(finder, '_INFERENCE_GYM_FINDER'):
      found = True
      break

  if found:
    sys.meta_path[i] = Finder()
  else:
    # We insert it at the beginning. This enables us to intercept the dynamic
    # modules before any other finder tries and (mistakenly) succeeds in somehow
    # handling them. This does force us to be careful about letting all the
    # regular modules to be handled by the regular finders via the early
    # returns.
    sys.meta_path.insert(0, Finder())


enable_backends()
