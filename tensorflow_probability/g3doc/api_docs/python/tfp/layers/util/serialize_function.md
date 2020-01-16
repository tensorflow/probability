<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.layers.util.serialize_function" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.layers.util.serialize_function


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/layers/util.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Serializes function for Keras.

``` python
tfp.layers.util.serialize_function(func)
```



<!-- Placeholder for "Used in" -->

(De)serializing Python functions from/to bytecode is unsafe. Therefore we
return the function's type as an anonymous function ('lambda') or named
function in the Python environment ('function'). In the latter case, this lets
us use the Python scope to obtain the function rather than reload it from
bytecode. (Note that both cases are brittle!)

This serialization mimicks the implementation in `tf.keras.layers.Lambda`.

#### Args:


* <b>`func`</b>: Python function to serialize.


#### Returns:

(serial, function_type): Serialized object, which is a tuple of its
bytecode (if function is anonymous) or name (if function is named), and its
function type.
