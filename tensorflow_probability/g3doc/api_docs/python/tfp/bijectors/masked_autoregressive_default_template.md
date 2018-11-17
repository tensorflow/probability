<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.bijectors.masked_autoregressive_default_template" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.bijectors.masked_autoregressive_default_template

``` python
tfp.bijectors.masked_autoregressive_default_template(
    hidden_layers,
    shift_only=False,
    activation=tf.nn.relu,
    log_scale_min_clip=-5.0,
    log_scale_max_clip=3.0,
    log_scale_clip_gradient=False,
    name=None,
    *args,
    **kwargs
)
```

Build the Masked Autoregressive Density Estimator (Germain et al., 2015).

This will be wrapped in a make_template to ensure the variables are only
created once. It takes the input and returns the `loc` ("mu" in [Germain et
al. (2015)][1]) and `log_scale` ("alpha" in [Germain et al. (2015)][1]) from
the MADE network.

Warning: This function uses `masked_dense` to create randomly initialized
`tf.Variables`. It is presumed that these will be fit, just as you would any
other neural architecture which uses `tf.layers.dense`.

#### About Hidden Layers

Each element of `hidden_layers` should be greater than the `input_depth`
(i.e., `input_depth = tf.shape(input)[-1]` where `input` is the input to the
neural network). This is necessary to ensure the autoregressivity property.

#### About Clipping

This function also optionally clips the `log_scale` (but possibly not its
gradient). This is useful because if `log_scale` is too small/large it might
underflow/overflow making it impossible for the `MaskedAutoregressiveFlow`
bijector to implement a bijection. Additionally, the `log_scale_clip_gradient`
`bool` indicates whether the gradient should also be clipped. The default does
not clip the gradient; this is useful because it still provides gradient
information (for fitting) yet solves the numerical stability problem. I.e.,
`log_scale_clip_gradient = False` means
`grad[exp(clip(x))] = grad[x] exp(clip(x))` rather than the usual
`grad[clip(x)] exp(clip(x))`.

#### Args:

* <b>`hidden_layers`</b>: Python `list`-like of non-negative integer, scalars
    indicating the number of units in each hidden layer. Default: `[512, 512].
* <b>`shift_only`</b>: Python `bool` indicating if only the `shift` term shall be
    computed. Default: `False`.
* <b>`activation`</b>: Activation function (callable). Explicitly setting to `None`
    implies a linear activation.
* <b>`log_scale_min_clip`</b>: `float`-like scalar `Tensor`, or a `Tensor` with the
    same shape as `log_scale`. The minimum value to clip by. Default: -5.
* <b>`log_scale_max_clip`</b>: `float`-like scalar `Tensor`, or a `Tensor` with the
    same shape as `log_scale`. The maximum value to clip by. Default: 3.
* <b>`log_scale_clip_gradient`</b>: Python `bool` indicating that the gradient of
    `tf.clip_by_value` should be preserved. Default: `False`.
* <b>`name`</b>: A name for ops managed by this function. Default:
    "masked_autoregressive_default_template".
* <b>`*args`</b>: `tf.layers.dense` arguments.
* <b>`**kwargs`</b>: `tf.layers.dense` keyword arguments.


#### Returns:

* <b>`shift`</b>: `Float`-like `Tensor` of shift terms (the "mu" in
    [Germain et al.  (2015)][1]).
* <b>`log_scale`</b>: `Float`-like `Tensor` of log(scale) terms (the "alpha" in
    [Germain et al. (2015)][1]).


#### Raises:

* <b>`NotImplementedError`</b>: if rightmost dimension of `inputs` is unknown prior to
    graph execution.

#### References

[1]: Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. MADE:
     Masked Autoencoder for Distribution Estimation. In _International
     Conference on Machine Learning_, 2015. https://arxiv.org/abs/1502.03509