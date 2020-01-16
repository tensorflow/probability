<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.mcmc.CheckpointableStatesAndTrace" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="all_states"/>
<meta itemprop="property" content="trace"/>
<meta itemprop="property" content="final_kernel_results"/>
<meta itemprop="property" content="__new__"/>
</div>

# tfp.mcmc.CheckpointableStatesAndTrace


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/mcmc/sample.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



## Class `CheckpointableStatesAndTrace`

States and auxiliary trace of an MCMC chain.



<!-- Placeholder for "Used in" -->

The first dimension of all the `Tensor`s in the `all_states` and `trace`
attributes is the same and represents the chain length.

#### Attributes:


* <b>`all_states`</b>: A `Tensor` or a nested collection of `Tensor`s representing the
  MCMC chain state.
* <b>`trace`</b>: A `Tensor` or a nested collection of `Tensor`s representing the
  auxiliary values traced alongside the chain.
* <b>`final_kernel_results`</b>: A `Tensor` or a nested collection of `Tensor`s
  representing the final value of the auxiliary state of the
  `TransitionKernel` that generated this chain.

<h2 id="__new__"><code>__new__</code></h2>

``` python
__new__(
    _cls,
    all_states,
    trace,
    final_kernel_results
)
```

Create new instance of CheckpointableStatesAndTrace(all_states, trace, final_kernel_results)




## Properties

<h3 id="all_states"><code>all_states</code></h3>




<h3 id="trace"><code>trace</code></h3>




<h3 id="final_kernel_results"><code>final_kernel_results</code></h3>






