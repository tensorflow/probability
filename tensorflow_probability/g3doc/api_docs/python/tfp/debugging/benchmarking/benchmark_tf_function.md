<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.debugging.benchmarking.benchmark_tf_function" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.debugging.benchmarking.benchmark_tf_function

Time a TensorFlow function under a variety of strategies and hardware.

``` python
tfp.debugging.benchmarking.benchmark_tf_function(
    user_fn,
    iters=1,
    config=default_benchmark_config(),
    extra_columns=None,
    use_autograph=False,
    print_intermediates=False,
    cpu_device='cpu:0',
    gpu_device='gpu:0'
)
```



Defined in [`python/debugging/benchmarking/benchmark_tf_function.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/debugging/benchmarking/benchmark_tf_function.py).

<!-- Placeholder for "Used in" -->

Runs the zero-argument callable `user_fn` `iters` times under the
strategies (any of Eager, tfe.function + graph, and XLA) and hardware (CPU,
GPU).


# Example:
```python
data_dicts = []
for inner_iters in [10, 100]:
  for size in [100, 1000]:
    def f():
      total = tf.constant(0.0)
      for _ in np.arange(inner_iters):
        m = tf.random_uniform((size, size))
        total += tf.reduce_sum(tf.matmul(m, m))
        return total

    data_dicts += benchmark_tf_function.benchmark_tf_function(
        f,
        iters=5,
        extra_columns={'inner_iters': inner_iters,
                       'size': size})
```

#### Args:


* <b>`user_fn`</b>: A zero-argument, callable function of TensorFlow code.
* <b>`iters`</b>: The number of times to run the function for each runtime and
  hardware combination.
* <b>`config`</b>: A BenchmarkTfFunctionConfig, specifying which strategies and
  hardware to use. Valid strategies are RUNTIME_EAGER, RUNTIME_FUNCTION, and
  RUNTIME_XLA. Valid hardware choices are HARDWARE_CPU, HARDWARE_GPU.
* <b>`extra_columns`</b>: A dictionary of extra information to add to each dictionary
  in data_dicts.
* <b>`use_autograph`</b>: Boolean, controlling whether autograph is used for the
  graph and XLA strategies.
* <b>`print_intermediates`</b>: Boolean. If true, print out each row before adding it
  to the data_dicts.
* <b>`cpu_device`</b>: String, the TensorFlow device to use for CPU.
* <b>`gpu_device`</b>: String, the TensorFlow device to use for GPU.


#### Returns:



* <b>`data_dicts`</b>: A list of dictionaries containing the results of benchmarking
  Time for the first run is stored under the `first_iter_time` key, and time
  for all runs is stored under the `total_time` key.