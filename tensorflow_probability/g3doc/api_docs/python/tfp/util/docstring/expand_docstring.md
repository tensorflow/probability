<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.util.docstring.expand_docstring" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.util.docstring.expand_docstring

Decorator to programmatically expand the docstring.

``` python
tfp.util.docstring.expand_docstring(**kwargs)
```



Defined in [`python/util/docstring.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/util/docstring.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`**kwargs`</b>: Keyword arguments to set. For each key-value pair `k` and `v`,
  the key is found as `${k}` in the docstring and replaced with `v`.


#### Returns:

Decorated function.