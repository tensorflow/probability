<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.interceptable" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.interceptable


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/edward2/interceptor.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Decorator that wraps `func` so that its execution is intercepted.

### Aliases:

* `tfp.experimental.edward2.interceptable`


``` python
tfp.edward2.interceptable(func)
```



<!-- Placeholder for "Used in" -->

The wrapper passes `func` to the interceptor for the current thread.

If there is no next interceptor, we perform an "immediate" call to `func`.
That is, `func` terminates without forwarding its execution to another
interceptor.

#### Args:


* <b>`func`</b>: Function to wrap.


#### Returns:

The decorated function.
