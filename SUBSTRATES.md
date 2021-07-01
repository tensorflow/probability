# Substrates (JAX, NumPy)

_Current as of 9/15/2020_

TFP supports alternative numerical backends to TensorFlow, including both JAX
and NumPy. The intent of this document is to explain some of the details of
supporting disparate backends, and how we have handled them, with the objective
of helping contributors read / write code, and understand how these
alternative-substrate packages are assembled.

## Alternative backends

In
[`tensorflow_probability/python/internal/backend`](https://cs.opensource.google/tensorflow/probability/+/main:tensorflow_probability/python/internal/backend)
we find the implementations of both the NumPy and JAX backends. These imitate
the portion of the TensorFlow API used by TFP, but are implemented in terms of
the corresponding substrate's primitives.

Since JAX provides `jax.numpy`, we are in most cases able to write a single
NumPy implementation under
[`tensorflow_probability/python/internal/backend`](https://cs.opensource.google/tensorflow/probability/+/main:tensorflow_probability/python/internal/backend),
then use a rewrite script (found at
[`tensorflow_probability/python/internal/backend/jax/rewrite.py`](https://cs.opensource.google/tensorflow/probability/+/main:tensorflow_probability/python/internal/backend/jax/rewrite.py))
to generate a JAX variant at `bazel build` time. See the genrules in
[`tensorflow_probability/python/internal/backend/jax/BUILD`](https://cs.opensource.google/tensorflow/probability/+/main:tensorflow_probability/python/internal/backend/jax/BUILD)
for more details of how this occurs. In cases where JAX provides a different API
(e.g. `random`) or a more performant API (e.g. batched matrix decompositions,
vmap, etc.), we will special-case using an `if JAX_MODE:` block.

## Wrinkles

*   __Shapes__

    In TensorFlow, the `shape` attribute of a `Tensor` is a `tf.TensorShape`
    object, whereas in JAX and NumPy, it is a simple `tuple` of `int`s. To
    handle cases for both systems nicely, we use `from
    tensorflow_probability.python.internal import tensorshape_util` and move
    special-casing into this library.

*   __DTypes__

    TensorFlow, and to some extent JAX, use the dtype of inputs to infer the
    dtype of outputs, and generally try to preserve the `Tensor` dtype in a
    binary op between `Tensor` and non-`Tensor`. NumPy, on the other hand,
    aggressively pushes dtypes toward 64-bit when left unspecified. In some
    cases debugging the JAX substrate, we have seen issues with dtypes changing
    from float32 to float64 or vice versa across the iterations of a while loop.
    Finding where this change happens can be tricky. Where possible, we aim to
    fix these in the `internal/backend` package, as opposed to implementation
    files.

*   __Shapes, again (JAX "omnistaging" / `prefer_static`)__

    Every JAX primitive observed within a JIT or JAX control flow context
    becomes an abstract Tracer. This is similar to `@tf.function`. The main
    challenge this introduces for TFP is that TF allows dynamic shapes as
    `Tensor`s whereas JAX (being built atop XLA) needs shapes to be statically
    available (i.e. a `tuple` or `numpy.ndarray`, *not* a JAX `ndarray`).

    If you observe issues with shapes derived from `Tracer`s in JAX, often a
    simple fix is `from tensorflow_probability.python.internal import
    prefer_static as ps` followed by replacing `tf.shape` with `ps.shape`, and
    similar for other ops such as `tf.rank`, `tf.size`, `tf.concat` (when
    dealing with shapes), (the args to) `tf.range`, etc. It's also useful to be
    aware of `ps.convert_to_shape_tensor`, which behaves like
    `tf.convert_to_tensor` for TF, but leaves things as `np.ndarray` for JAX.
    Similarly, in constructors, use the `as_shape_tensor=True` arg to
    `tensor_util.convert_nonref_to_tensor` for shape-related values.

*   __`tf.GradientTape`__

    TF uses a tape to record ops for later gradient evaluation, whereas JAX
    rewrites a function while tracing its execution. Since the function
    transform is more general, we aim to replace usage of `GradientTape` (in
    tests, TFP impl, etc), with `tfp.math.value_and_gradient` or similar. Then,
    we can special-case `JAX_MODE` inside the body of `value_and_gradient`.

*   __`tf.Variable`, `tf.optimizers.Optimizer`__

    TF provides a `Variable` abstraction so that graph functions may modify
    state, including using the TF `Optimizer` subclasses like `Adam`. JAX, in
    contrast, operates only on pure functions. In general, TFP is fairly
    functional (e.g. `tfp.optimizer.lbfgs_minimize`), but in some cases (e.g.
    `tfp.vi.fit_surrogate_posterior`,
    `tfp.optimizer.StochasticGradientLangevinDynamics`) we have felt the
    mismatch too strong to try to port code to JAX. Some approaches to hoisting
    state out of a stateful function can be seen in the TFP spinoff project
    [`oryx`](https://www.tensorflow.org/probability/oryx/notebooks/a_tour_of_oryx).

*   __Custom derivatives__

    JAX supports both forward and reverse mode autodifferentiation, and where
    possible TFP aims to support both in JAX. To do so, in places where we
    define a custom derivative, we use an internal wrapper which provides a
    function decorator that supports both TF and JAX's interfaces for custom
    derivatives, namely:

    ```python
    from tensorflow_probability.python.internal import custom_gradient as tfp_custom_gradient

    def _f(..): pass
    def _f_fwd(..): return _f(..), bwd_auxiliary_data
    def _f_bwd(bwd_auxiliary_data, dy): pass
    def _f_jvp(primals, tangents): return _f(*primals), df(primals, tangents)
    @tfp_custom_gradient.custom_gradient(vjp_fwd=_f_fwd, vjp_bwd=_f_bwd, jvp_fn=_f_jvp)
    def f(..): return _f(..)
    ```

    For more information, the
    [JAX custom derivatives doc](https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html)
    can be useful.

*   __Randomness__

    In TF, we support both "stateful" (i.e. some latent memory tracks the state
    of the sampler) and "stateless" sampling. JAX natively supports only
    stateless, for functional purity reasons. For internal use, we have `from
    tensorflow_probability.python.internal import samplers`, a library that
    provides methods to:

    *   convert stateful seeds to stateless, add salts (`sanitize_seed`)
    *   split stateless seeds to multiple descendant seeds (`split_seed`)
    *   proxy through to a number of stateless samplers (`normal`, `uniform`,
        ...)

    When the rewrite script is dealing with a `..._test.py` file, it will
    rewrite calls to `tf.random.{uniform,normal,...}` to
    `tf.random.stateless_{uniform,normal,...}` to ensure compatibility with the
    JAX backend, which only implements the stateless samplers.

## Rewriting TF code

In a couple cases, we commit into the repository script-munged source from
TensorFlow. These files can be found under
[`tensorflow_probability/python/internal/backend/numpy/gen`](https://cs.opensource.google/tensorflow/probability/+/main:tensorflow_probability/python/internal/backend/numpy/gen).
They currently include:

*   an implementation of `tf.TensorShape`
*   several parts of `tf.linalg`, especially the `tf.linalg.LinearOperator`
    classes

The actual rewriting is accomplished by scripts found under
[`tensorflow_probability/python/internal/backend/meta`](https://cs.opensource.google/tensorflow/probability/+/main:tensorflow_probability/python/internal/backend/meta),
namely `gen_linear_operators.py` and `gen_tensor_shape.py`.

The test
[`tensorflow_probability/python/internal/backend/numpy/rewrite_equivalence_test.py`](https://cs.opensource.google/tensorflow/probability/+/main:tensorflow_probability/python/internal/backend/numpy/rewrite_equivalence_test.py)
verifies that the files in TensorFlow, when rewritten, match the files in the
`gen/` directory. The test uses `BUILD` dependencies on `genrule`s that apply
the rewrite scripts, and compares those genrule inputs to the source of the
files under the `gen/` directory.

Similar to the sources in `internal/backend/numpy`, the sources in
`internal/backend/numpy/gen` are rewritten by `jax/rewrite.py`. Note that the
files under `gen/` do not have the `numpy` import rewritten. This is because we
only want to rewrite TensorFlow usage of TensorFlow-ported code; typically when
TF code is using `numpy`, it is munging shapes, and JAX does not like shapes to
be munged using `jax.numpy` (must use plain `numpy`).

## Rewriting TFP code

With `internal/backend/{numpy,jax}` now ready to provide a `tf2jax` or
`tf2numpy` backend, we can proceed to the core packages of TFP.

The script
[`tensorflow_probability/substrates/meta/rewrite.py`](https://cs.opensource.google/tensorflow/probability/+/main:tensorflow_probability/substrates/meta/rewrite.py)
runs on TFP sources to auto-generate JAX and NumPy python source corresponding
to the given TF source.

The most important job of the rewrite script is to rewrite `import
tensorflow.compat.v2 as tf` to `from
tensorflow_probability.python.internal.backend.jax import v2 as tf`. Second to
that, the script will rewrite dependencies on TFP subpackages to dependencies on
the corresponding substrate-specific TFP subpackages. For example, the line
`from tensorflow_probability.python import math as tfp_math` becomes `from
tensorflow_probability.substrates.jax import math as tfp_math`. Beyond that,
there are a number of peripheral replacements to work around other wrinkles
we've accumulated over time.

In rare cases we will put an explicit `if JAX_MODE:` or `if NUMPY_MODE:` block
into the implementation code of a TFP submodule. This should be very uncommon.
Whenever possible, the intent is for such special-casing to live under
`python/internal`. For example, today we see in `bijectors/softplus.py`:

```python
# TODO(b/155501444): Remove this when tf.nn.softplus is fixed.
if JAX_MODE:
  _stable_grad_softplus = tf.nn.softplus
else:
  @tf.custom_gradient
  def _stable_grad_softplus(x):  # ...
```

_Note that this rewrite currently adds exactly a 10-line header, so line numbers
from stack traces will be +10 from the raw code._

## BUILD rules

[`tensorflow_probability/python/build_defs.bzl`](https://cs.opensource.google/tensorflow/probability/+/main:tensorflow_probability/python/build_defs.bzl)
defines a pair of `bazel` build rules: `multi_substrate_py_library` and
`multi_substrate_py_test`.

These rules automatically invoke
[`tensorflow_probability/substrates/meta/rewrite.py`](https://cs.opensource.google/tensorflow/probability/+/main:tensorflow_probability/substrates/meta/rewrite.py)
to emit JAX/NumPy source variants. The file `bijectors/softplus.py` gets
rewritten into `bijectors/_generated_jax_softplus.py` (you can view the output
under the corresponding `bazel-genfiles` directory).

These build rules are also responsible for rewriting TFP-internal `deps` to the
`some_dep.jax` or `some_dep.numpy` substrate-specific replacement.

The `multi_substrate_py_library` will emit three targets: a TF `py_library` with
the name given by the `name` argument, a JAX `py_library` with name `name +
'.jax'`, and a NumPy `py_library` with name `name + '.numpy'`.

The `multi_substrate_py_test` will emit three targets, each of `name + '.tf'`,
`name + '.jax'`, and `name + '.numpy'`. Rules specified by the
`disabled_substrates` arg will not have BUILD rules emitted at all; `jax_tags`
and `numpy_tags` can be used to specify specific tags that drop CI coverage
while keeping the target buildable and testable. The distinction is useful so
that we can track cases where we think a test should be fixable, but we haven't
yet, as opposed to cases like HMC where we know the test will never pass for
NumPy so we prefer to not even have the test target. All emitted test targets
are aggregated into a `test_suite` with name corresponding to the original
`name` arg.

In cases where we know we will never be able to support a given feature, the
`substrates_omit_deps`, `jax_omit_deps`, and `numpy_omit_deps` args to
`multi_substrate_py_library` can be used to exclude things. Examples include
non-pure code or code using `tf.Variable` (JAX wants pure functions), or HMC (no
gradients in NumPy!). When rewriting an `__init__.py` file, the rewrite script
is set up to comment out imports and `__all__` lines corresponding to the
omitted deps.

In order to test against the same directory hierarchy as we use for wheel
packaging, the `multi_substrate_py_library` does some internal gymnastics with a
custom bazel `rule` which is able to add symlinks into
[`tensorflow_probability/substrates`](https://cs.opensource.google/tensorflow/probability/+/main:tensorflow_probability/python/build_defs.bzl;l=118)
pointing to that point to implementation files generated under
`bazel-genfiles/tensorflow_probability/python` (details in
`_substrate_runfiles_symlinks_impl` of `build_defs.bzl`).

## Wheel packaging

When it comes to building the wheel, we must first use `cp -L` to resolve the
symlinks added as part of the `bazel build`. Otherwise the wheel does not follow
them and fails to include `tfp.substrates`. This `cp -L` command sits in
`pip_pkg.sh` (currently adjacent to this doc).

## Integration testing

A couple of integration tests sit in
[`tensorflow_probability/substrates/meta/jax_integration_test.py`](https://cs.opensource.google/tensorflow/probability/+/main:tensorflow_probability/substrates/meta/jax_integration_test.py)
and
[`tensorflow_probability/substrates/meta/numpy_integration_test.py`](https://cs.opensource.google/tensorflow/probability/+/main:tensorflow_probability/substrates/meta/numpy_integration_test.py).

We run these under CI after building and installing a wheel to verify that the
`tfp.substrates` packages load correctly and do not require a `tensorflow`
install.
