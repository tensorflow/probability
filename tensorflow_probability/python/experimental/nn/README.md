# Standards for the `tfp.experimental.nn` Neural Network Toolbox

The Tensorflow Probability (TFP) Neural Network toolbox, `tfp.experimental.nn`,
provides tools for building Bayesian and/or non-Bayesian neural networks. This
code is highly experimental and currently for discussion purpose only.

Design goals include but are not limited to:

- performance
- debuggability
- extensibility
- simple implementations.

The primary differences from `tf.keras` are:

1. The TFP NN toolbox use `tf.Module` for `tf.Variable` tracking.
2. Users are expected to implement their own train loops.
3. Input shapes are not inferred; there is no concept of "compile" or "build".
4. All `tfp.experimental.nn.Layer` instances should support arbitrary input
   batch shape (e.g., for bootstrap) and batches of parameters (e.g., for Monte
   Carlo approximations under weight uncertainty).

The following describes the implementation commitments of
`tfp.experimental.nn.Layer`s. You are encouraged to subclass
`tfp.experimental.nn.Layer` and invited to disregard any/all of these standards.

## `tfp.experimental.nn.Layer` Requirements

1. `tfp.experimental.nn.Layer`s operate on the rightmost input dimensions.

2. The first two arguments of kernel/bias type `tfp.experimental.nn` layers are
   always `input_size` and `output_size`. (There may be other required
   arguments.)

3. `tfp.experimental.nn.Layer`s always provide a mechanism for
   user-specification of how the underlying `tf.Variable`s are created. For
   non-Bayesian kernel/bias layers this manifests as the `make_kernel_bias_fn`
   argument (default: `tfp.experimental.nn.util.make_kernel_bias`).

4. `tfp.experimental.nn.Layer`s have a fixed `dtype` specified at layer
   creation.


5. `tfp.experimental.nn.Layer`s accept a `name` argument used for debugging
   purposes.

6. `tfp.experimental.nn.Layer`s have a standard `__call__` implementation
   consisting of:

  ```python
  def __call__(self, inputs, **kwargs):
    if callable(inputs):
      return Sequential([inputs, self], **kwargs)
    self._extra_loss = self._extra_result = None
    return self.eval(inputs, **kwargs)
  ```

  Users are expected to override `eval`.

7. The `eval` function should have arguments `self, inputs, is_training=True,
   **kwargs` where `inputs` is the output of the previous layer. The `eval`
   function return only `tf.Tensor`s or objects convertible to `tf.Tensor`s. We
   encourage using `@tf.function` decoration of the `eval` function.

8. `tfp.experimental.nn.Layer`s may return "side results" by either setting the
   `extra_loss` member or `extra_results` member. If these properties exist,
   they will be lifted from graph mode to eager mode upon completion of
   `__call__`.

9. All `tfp.experimental.nn.Layer` instances should support arbitrary input
   batch shape (e.g., for bootstrap) and batches of parameters (e.g., for Monte
   Carlo approximations under weight uncertainty).
