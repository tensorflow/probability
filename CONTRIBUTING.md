# Contributing

Interested in contributing to TensorFlow Probability? We appreciate all kinds
of help!

## Pull Requests

We gladly welcome [pull requests](
https://help.github.com/articles/about-pull-requests/).

Before making any changes, we recommend opening an issue (if it
doesn't already exist) and discussing your proposed changes. This will
let us give you advice on the proposed changes. If the changes are
minor, then feel free to make them without discussion.

Want to contribute but not sure of what? Here are a few suggestions:

1. Add a new example or tutorial.  Located in [`examples/`](
  https://github.com/tensorflow/probability/tree/main/tensorflow_probability/examples),
  these are a great way to familiarize yourself and others with TFP tools.

2. Solve an [existing issue](https://github.com/tensorflow/probability/issues).
  These range from low-level software bugs to higher-level design problems.
  Check out the label [good first issue](
  https://github.com/tensorflow/probability/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

All submissions, including submissions by project members, require review. After
a pull request is approved, we merge it. Note our merging process differs
from GitHub in that we pull and submit the change into an internal version
control system. This system automatically pushes a git commit to the GitHub
repository (with credit to the original author) and closes the pull request.

## Continuous Integration

We use [GitHub Actions](https://github.com/tensorflow/probability/actions) to do
automated style checking and run unit-tests (discussed in more detail below). A
build will be triggered when you open a pull request, or update the pull request
by adding a commit, rebasing etc.

We test against TensorFlow nightly on Python 3.7. We shard our tests
across several build jobs (identified by the `SHARD` environment variable).
Lints are also done in a separate job.

All pull-requests will need to pass the automated lint and unit-tests before
being merged. As the tests can take a bit of time, see the following sections
on how to run the lint checks and unit-tests locally while you're developing
your change.

## Style

See the [TensorFlow Probability style guide](STYLE_GUIDE.md).  Running `pylint`
detects many (but certainly not all) style issues.  TensorFlow Probability
follows a custom [pylint
configuration](https://github.com/tensorflow/probability/blob/main/testing/pylintrc).

## Unit tests

All TFP code-paths must be unit-tested; see this [unit-test guide](UNITTEST.md)
for recommended test setup.

Unit tests ensure new features (a) work correctly and (b) guard against future
breaking changes (thus lower maintenance costs).

### Setup

#### `bazel`

We use [`bazel`](https://bazel.build/) to manage building, packaging, and
testing TFP. You'll need to install `bazel` before running our tests (we have
recently added some experimental support for running some tests with pytest, but
for a variety of reasons this will probably never work 100%). See instructions
[here](https://docs.bazel.build/versions/3.2.0/install-os-x.html) on installing
`bazel`.

#### `virtualenv`

We strongly recommend running unit tests in an active
[virtualenv](https://virtualenv.pypa.io/en/latest/). Doing so requires some
extra bazel flags, so we created a wrapper script, which we suggest using. An
example invocation (presumed to run from the root of the TFP repo:

```shell
./testing/run_tfp_test.sh //tensorflow_probability/...
```

#### Dependencies

To run the unit tests, you'll need several packages installed (again, we
strongly recommend you work in a virtualenv). We include a script to do this for
you, which also does some sanity checks on the environment:

```shell
./testing/install_test_dependencies.sh
```

See the
[header comments in that script](https://github.com/tensorflow/probability/blob/main/testing/install_test_dependencies.sh)
for more details.

#### Helper scripts

```shell
# Run all TFP tests.
./testing/run_tfp_test.sh //tensorflow_probability/...
```

```shell
# Run one TFP test.
./testing/run_tfp_test.sh //tensorflow_probability/python/distributions:joint_distribution_coroutine_test
```

```shell
# Lint a file (requires pylint installed, e.g. via pip install pylint).
./testing/run_tfp_lints.sh tensorflow_probability/python/distributions/joint_distribution_coroutine.py
```

See comments at the top of the script for more info.

For convenience, also consider sourcing the following script to alias `tfp_test`
and `tfp_lints` to the above script:

```shell
source ./testing/define_testing_alias.sh
source ./testing/define_linting_alias.sh
# Run all TFP tests.
tfp_test //tensorflow_probability/...
# Run one TFP test.
tfp_test //tensorflow_probability/python/distributions:joint_distribution_coroutine_test
# Lint a file.
tfp_lints tensorflow_probability/python/distributions/joint_distribution_coroutine.py
```

We also have a script that bundles creating a new virtualenv and installing TFP
dependencies in it.

```shell
source ./testing/fresh_tfp_virtualenv.sh
```

### Additional considerations

As of early 2020, tensorflow and tf-nightly include GPU support by default,
which means if you have a GPU installed and run tests with the default
`tf-nightly` pip package, tests will try to run using the GPU. To avoid this,
the dependency install script installs tf-nightly-cpu by default. If you *want*
to run tests on the GPU, you can use pass --enable_gpu flag to
`testing/install_test_dependencies.sh`. In this case, you will also need to
include the flag `--jobs=1`, since by default Bazel will run many tests in
parallel, and each one will try to claim all the GPU memory:

```shell
tfp_test --jobs=1 //tensorflow_probability/...
```


## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.
