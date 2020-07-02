# Install

## Stable builds

Install the latest version of TensorFlow Probability:

<pre class="devsite-terminal devsite-click-to-copy prettyprint lang-shell">
pip install --upgrade tensorflow-probability
</pre>

TensorFlow Probability depends on a recent stable release of
[TensorFlow](https://www.tensorflow.org/install) (pip package `tensorflow`). See
the [TFP release notes](https://github.com/tensorflow/probability/releases) for
details about dependencies between TensorFlow and TensorFlow Probability.

Note: Since TensorFlow is *not* included as a dependency of the TensorFlow
Probability package (in `setup.py`), you must explicitly install the TensorFlow
package (`tensorflow` or `tensorflow-gpu`). This allows us to maintain one
package instead of separate packages for CPU and GPU-enabled TensorFlow.

To force a Python 3-specific install, replace `pip` with `pip3` in the above
commands. For additional installation help, guidance installing prerequisites,
and (optionally) setting up virtual environments, see the [TensorFlow
installation guide](https://www.tensorflow.org/install).

## Nightly builds

There are also nightly builds of TensorFlow Probability under the pip package
`tfp-nightly`, which depend on one of `tf-nightly` and `tf-nightly-gpu`. Nightly
builds include newer features, but may be less stable than the versioned
releases.

## Install from source

You can also install from source. This requires the
[Bazel](https://bazel.build/){:.external} build system. It is highly recommended
that you install the nightly build of TensorFlow (`tf-nightly`) before trying to
build TensorFlow Probability from source.

<!-- common_typos_disable -->
<pre class="devsite-click-to-copy">
  <code class="devsite-terminal">sudo apt-get install bazel git python-pip</code>
  <code class="devsite-terminal">python -m pip install --upgrade --user tf-nightly</code>
  <code class="devsite-terminal">git clone https://github.com/tensorflow/probability.git</code>
  <code class="devsite-terminal">cd probability</code>
  <code class="devsite-terminal">bazel build --copt=-O3 --copt=-march=native :pip_pkg</code>
  <code class="devsite-terminal">PKGDIR=$(mktemp -d)</code>
  <code class="devsite-terminal">./bazel-bin/pip_pkg $PKGDIR</code>
  <code class="devsite-terminal">python -m pip install --upgrade --user $PKGDIR/*.whl</code>
</pre>
<!-- common_typos_enable -->
