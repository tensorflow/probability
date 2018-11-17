<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.interceptable" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.interceptable

``` python
tfp.edward2.interceptable(func)
```

Decorator that wraps `func` so that its execution is intercepted.

The wrapper passes `func` to the interceptor for the current thread.

If there is no next interceptor, we perform an "immediate" call to `func`.
That is, `func` terminates without forwarding its execution to another
interceptor.

#### Args:

* <b>`func`</b>: Function to wrap.


#### Returns:

The decorated function.