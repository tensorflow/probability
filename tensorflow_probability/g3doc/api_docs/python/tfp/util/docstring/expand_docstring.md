<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.util.docstring.expand_docstring" />
</div>

# tfp.util.docstring.expand_docstring

``` python
tfp.util.docstring.expand_docstring(**kwargs)
```

Decorator to programmatically expand the docstring.

#### Args:

* <b>`**kwargs`</b>: Keyword arguments to set. For each key-value pair `k` and `v`,
    the key is found as `${k}` in the docstring and replaced with `v`.


#### Returns:

Decorated function.