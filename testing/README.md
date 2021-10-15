This directory contains both developer-facing and tool-facing scripts.

Developer-facing:

- `install_test_dependencies.sh` installs dependencies of TFP (but not TFP
  itself) using `pip`, including packages that are only needed for TFP's test
  suite.  This is also indirectly used by tools.

- `run_tfp_test.sh` wraps a `bazel` incantation to run one or a set of TFP tests
  in a virtualenv.  This is also indirectly used by tools.

- `run_tfp_lints.sh` wraps a `pylint` incantation using `pylintrc` (also here).
  This is also indirectly used by tools.

- `define_linting_alias.sh` and `define_testing_alias.sh` can be added to a
  developer's `.bashrc` file to alias `run_tfp_lints.sh and `run_tfp_test.sh`,
  respectively.

- `fresh_tfp_virtualenv.sh` wraps `install_test_dependencies.sh` to automate
  creating and entering a fresh virtualenv every time.

Tool-facing:

- `run_github_tests.sh` is the entry point for the testing Github Action.

- `run_github_lints.sh` is the entry point for the linting Github Action.

Support:

- `pylintrc` is our Pylint configuration (referenced by `run_tfp_lints.sh`).

- `dependency_install_lib.sh` is a bunch of bash functions used by
  `install_test_dependencies.sh`.

- `virtualenv_is_active.py` is a Python script to detect whether we are already
  in a virtualenv or not.
