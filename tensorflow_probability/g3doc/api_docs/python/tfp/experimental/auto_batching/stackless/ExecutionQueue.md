<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.stackless.ExecutionQueue" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="dequeue"/>
<meta itemprop="property" content="enqueue"/>
</div>

# tfp.experimental.auto_batching.stackless.ExecutionQueue


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/auto_batching/stackless.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



## Class `ExecutionQueue`

A priority queue of resumption points.



### Aliases:

* Class `tfp.experimental.auto_batching.frontend.st.ExecutionQueue`


<!-- Placeholder for "Used in" -->

Each resumption point is a pair of program counter to resume, and mask of
threads that are waiting there.

This class is a simple wrapper around Python's standard heapq implementation
of priority queues.  There are just two subtleties:

- Dequeue gets all the threads that were waiting at that point, by coalescing
  multiple entries if needed.

- Enqueue drops entries with empty masks, because they need never be resumed.

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/auto_batching/stackless.py">View source</a>

``` python
__init__(backend)
```

Initialize self.  See help(type(self)) for accurate signature.




## Methods

<h3 id="dequeue"><code>dequeue</code></h3>

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/auto_batching/stackless.py">View source</a>

``` python
dequeue()
```




<h3 id="enqueue"><code>enqueue</code></h3>

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/auto_batching/stackless.py">View source</a>

``` python
enqueue(
    program_counter,
    mask
)
```






