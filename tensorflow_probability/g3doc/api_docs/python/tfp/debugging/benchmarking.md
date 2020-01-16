<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.debugging.benchmarking" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="HARDWARE_CPU"/>
<meta itemprop="property" content="HARDWARE_GPU"/>
<meta itemprop="property" content="RUNTIME_EAGER"/>
<meta itemprop="property" content="RUNTIME_FUNCTION"/>
<meta itemprop="property" content="RUNTIME_XLA"/>
</div>

# Module: tfp.debugging.benchmarking


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/debugging/benchmarking/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



TensorFlow Probability benchmarking library.

<!-- Placeholder for "Used in" -->


## Classes

[`class BenchmarkTfFunctionConfig`](../../tfp/debugging/benchmarking/BenchmarkTfFunctionConfig.md): BenchmarkTfFunctionConfig(strategies, hardware)

## Functions

[`benchmark_tf_function(...)`](../../tfp/debugging/benchmarking/benchmark_tf_function.md): Time a TensorFlow function under a variety of strategies and hardware.

[`default_benchmark_config(...)`](../../tfp/debugging/benchmarking/default_benchmark_config.md)

## Other Members

* `HARDWARE_CPU = 'cpu'` <a id="HARDWARE_CPU"></a>
* `HARDWARE_GPU = 'gpu'` <a id="HARDWARE_GPU"></a>
* `RUNTIME_EAGER = 'eager'` <a id="RUNTIME_EAGER"></a>
* `RUNTIME_FUNCTION = 'function/graph'` <a id="RUNTIME_FUNCTION"></a>
* `RUNTIME_XLA = 'function/xla'` <a id="RUNTIME_XLA"></a>
