Project: /probability/_project.yaml
Book: /probability/_book.yaml

# Install

## Stable builds

Install the latest version of TensorFlow Probability:

<pre class="devsite-terminal devsite-click-to-copy prettyprint lang-shell">
pip install --upgrade tensorflow-probability  # depends on tensorflow (CPU-only)
</pre>

TensorFlow Probability depends on a recent stable release of TensorFlow
(pip package `tensorflow`), see the
[TFP release notes](https://github.com/tensorflow/probability/releases) for
details about the latest version of TensorFlow Probability and the version of
TensorFlow it depends on.

There is also a GPU-enabled package:

<pre class="devsite-terminal devsite-click-to-copy prettyprint lang-shell">
pip install --upgrade tensorflow-probability-gpu  # depends on tensorflow-gpu
</pre>

Currently, TensorFlow Probability does not contain any GPU-specific code. The
primary difference between these packages is that `tensorflow-probability-gpu`
depends on a GPU-enabled version of TensorFlow.

To force a Python 3-specific install, replace `pip` with `pip3` in the above
commands. For additional installation help, guidance installing prerequisites,
and (optionally) setting up virtual environments, see the [TensorFlow
installation guide](https://www.tensorflow.org/install).

## Nightly builds

There are also nightly builds of TensorFlow Probability under the pip packages
`tfp-nightly` and `tfp-nightly-gpu`, which depend on `tf-nightly` and
`tf-nightly-gpu`, respectively. These builds include newer features, but may be
less stable than the versioned releases.

## Install from source

You can also install from source. This requires the
[Bazel](https://bazel.build/){:.external} build system.

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt-get install bazel git python-pip</code>
  <code class="devsite-terminal">git clone https://github.com/tensorflow/probability.git</code>
  <code class="devsite-terminal">cd probability</code>
  <code class="devsite-terminal">bazel build --copt=-O3 --copt=-march=native :pip_pkg</code>
  <code class="devsite-terminal">PKGDIR=$(mktemp -d)</code>
  <code class="devsite-terminal">./bazel-bin/pip_pkg $PKGDIR</code>
  <code class="devsite-terminal">pip install --user --upgrade $PKGDIR/*.whl</code>
</pre>
<!-- common_typos_enable -->
