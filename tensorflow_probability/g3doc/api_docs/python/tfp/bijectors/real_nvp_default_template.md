<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.bijectors.real_nvp_default_template" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.bijectors.real_nvp_default_template

Build a scale-and-shift function using a multi-layer neural network.

``` python
tfp.bijectors.real_nvp_default_template(
    hidden_layers,
    shift_only=False,
    activation=tf.nn.relu,
    name=None,
    *args,
    **kwargs
)
```



Defined in [`python/bijectors/real_nvp.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/bijectors/real_nvp.py).

<!-- Placeholder for "Used in" -->

This will be wrapped in a make_template to ensure the variables are only
created once. It takes the `d`-dimensional input x[0:d] and returns the `D-d`
dimensional outputs `loc` ("mu") and `log_scale` ("alpha").

The default template does not support conditioning and will raise an
exception if `condition_kwargs` are passed to it. To use conditioning in
real nvp bijector, implement a conditioned shift/scale template that
handles the `condition_kwargs`.

#### Arguments:


* <b>`hidden_layers`</b>: Python `list`-like of non-negative integer, scalars
  indicating the number of units in each hidden layer. Default: `[512, 512].
* <b>`shift_only`</b>: Python `bool` indicating if only the `shift` term shall be
  computed (i.e. NICE bijector). Default: `False`.
* <b>`activation`</b>: Activation function (callable). Explicitly setting to `None`
  implies a linear activation.
* <b>`name`</b>: A name for ops managed by this function. Default:
  "real_nvp_default_template".
* <b>`*args`</b>: `tf.layers.dense` arguments.
* <b>`**kwargs`</b>: `tf.layers.dense` keyword arguments.


#### Returns:


* <b>`shift`</b>: `Float`-like `Tensor` of shift terms ("mu" in
  [Papamakarios et al.  (2016)][1]).
* <b>`log_scale`</b>: `Float`-like `Tensor` of log(scale) terms ("alpha" in
  [Papamakarios et al. (2016)][1]).


#### Raises:


* <b>`NotImplementedError`</b>: if rightmost dimension of `inputs` is unknown prior to
  graph execution, or if `condition_kwargs` is not empty.

#### References

[1]: George Papamakarios, Theo Pavlakou, and Iain Murray. Masked
     Autoregressive Flow for Density Estimation. In _Neural Information
     Processing Systems_, 2017. https://arxiv.org/abs/1705.07057