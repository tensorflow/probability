<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.xla.compile_nested_output" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.auto_batching.xla.compile_nested_output

Wraps f with a `tpu.rewrite` or `xla.compile`, propagates output structure.

``` python
tfp.experimental.auto_batching.xla.compile_nested_output(
    f,
    compile_fn=None
)
```



Defined in [`python/internal/auto_batching/xla.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/xla.py).

<!-- Placeholder for "Used in" -->

`xla.compile` insists `f` output a flat list of `Tensor`s or `Op`s, but
tolerates nested input arguments. Here, we capture the output structure in
order to propagate it.

#### Args:


* <b>`f`</b>: Callable to compile, may accept/return nested inputs/outputs.
* <b>`compile_fn`</b>: The function to use to compile, i.e. `xla.compile` or
  `tpu.rewrite`. Accepts two args, `f` and `inputs`.


#### Returns:


* <b>`g`</b>: Callable wrapping `f` which returns XLA-compiled, nested outputs.