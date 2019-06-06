<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.layers.util.deserialize_function" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.layers.util.deserialize_function

Deserializes the Keras-serialized function.

``` python
tfp.layers.util.deserialize_function(
    serial,
    function_type
)
```



Defined in [`python/layers/util.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/layers/util.py).

<!-- Placeholder for "Used in" -->

(De)serializing Python functions from/to bytecode is unsafe. Therefore we
also use the function's type as an anonymous function ('lambda') or named
function in the Python environment ('function'). In the latter case, this lets
us use the Python scope to obtain the function rather than reload it from
bytecode. (Note that both cases are brittle!)

Keras-deserialized functions do not perform lexical scoping. Any modules that
the function requires must be imported within the function itself.

This serialization mimicks the implementation in `tf.keras.layers.Lambda`.

#### Args:


* <b>`serial`</b>: Serialized Keras object: typically a dict, string, or bytecode.
* <b>`function_type`</b>: Python string denoting 'function' or 'lambda'.


#### Returns:


* <b>`function`</b>: Function the serialized Keras object represents.

#### Examples

```python
serial, function_type = serialize_function(lambda x: x)
function = deserialize_function(serial, function_type)
assert function(2.3) == 2.3  # function is identity
```