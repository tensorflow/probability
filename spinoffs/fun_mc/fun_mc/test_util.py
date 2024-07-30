# Copyright 2021 The TensorFlow Probability Authors.
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
"""Test utilities."""

import importlib

BACKEND = None  # Rewritten by backends/rewrite.py.


def multi_backend_test(
    globals_dict,
    relative_module_name,
    backends=('jax', 'tensorflow'),
    test_case=None,
):
  """Multi-backend test decorator.

  The end goal of this decorator is that the decorated test case is removed, and
  replaced with a set of new test cases that have been rewritten to use one or
  more backends. E.g., a test case named `Test` will by default be rewritten to
  `Test_jax` and 'Test_tensorflow' which use the JAX and TensorFlow,
  respectively.

  The decorator works by using the dynamic rewrite system to rewrite imports of
  the module the test is defined in, and inserting the approriately renamed test
  cases into the `globals()` dictionary of the original module. A side-effect of
  this is that the global code inside the module is run `1 + len(backends)`
  times, so avoid doing anything expensive there. This does mean that the
  original module needs to be in a runnable state, i.e., when it uses symbols
  from `backend`, those must be actually present in the literal `backend`
  module.

  A subtle point about what this decorator does in the rewritten modules: the
  rewrite system changes the behavior of this decorator to act as a passthrough
  to avoid infinite rewriting loops.

  Args:
    globals_dict: Python dictionary of strings to symbols. Set this to the value
      of `globals()`.
    relative_module_name: Python string. The module name of the module where the
      decorated test resides relative to `fun_mc`. You must not use `__name__`
      for this as that is set to a defective value of `__main__` which is
      sufficiently abnormal that the rewrite system does not work on it.
    backends: Python iterable of strings. Which backends to test with.
    test_case: The actual test case to decorate.

  Returns:
    None, to delete the original test case.
  """
  if test_case is None:
    return lambda test_case: multi_backend_test(  # pylint: disable=g-long-lambda
        globals_dict=globals_dict,
        relative_module_name=relative_module_name,
        test_case=test_case,
    )

  if BACKEND is not None:
    return test_case

  if relative_module_name == '__main__':
    raise ValueError(
        'module_name should be written out manually, not by passing __name__.'
    )

  # This assumes `test_util` is 1 levels deep inside of `fun_mc`. If we
  # move it, we'd change the `-1` to equal the (negative) nesting level.
  root_name_comps = __name__.split('.')[:-1]
  relative_module_name_comps = relative_module_name.split('.')

  # Register the rewrite hooks.
  importlib.import_module('.'.join(root_name_comps + ['backends', 'rewrite']))

  new_test_case_names = []
  for backend in backends:
    new_module_name_comps = (
        root_name_comps
        + ['dynamic', 'backend_{}'.format(backend)]
        + relative_module_name_comps
    )
    # Rewrite the module.
    new_module = importlib.import_module('.'.join(new_module_name_comps))

    # Subclass the test case so that we can rename it (absl uses the class name
    # in its UI).
    base_new_test = getattr(new_module, test_case.__name__)
    new_test = type(
        '{}_{}'.format(test_case.__name__, backend), (base_new_test,), {}
    )
    new_test_case_names.append(new_test.__name__)
    globals_dict[new_test.__name__] = new_test

  # We deliberately return None to delete the original test case from the
  # original module.
