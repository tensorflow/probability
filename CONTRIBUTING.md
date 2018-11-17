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
  https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples),
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

## Style

See the [TensorFlow Probability style guide](STYLE_GUIDE.md).

## Unit tests

All TFP code-paths must be unit-tested; see this [unit-test guide](UNITTEST.md)
for recommended test setup.

Unit tests ensure new features (a) work correctly and (b) guard against future
breaking changes (thus lower maintenance costs).

To run existing unit-tests on CPU, use the command:


```shell
bazel test --copt=-O3 --copt=-march=native //tensorflow_probability/...
```

from the root of the `tensorflow_probability` repository. To run tests on GPU,
you just need to ensure the GPU-enabled version of TensorFlow is installed.
However, you will also need to include the flag `--jobs=1`, since by default
Bazel will run many tests in parallel, and each one will try to claim all the
GPU memory:

```shell
bazel test --jobs=1 --copt=-O3 --copt=-march=native //tensorflow_probability/...
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
