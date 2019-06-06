<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.frontend.st.ExecutionQueue" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="dequeue"/>
<meta itemprop="property" content="enqueue"/>
</div>

# tfp.experimental.auto_batching.frontend.st.ExecutionQueue

## Class `ExecutionQueue`

A priority queue of resumption points.





Defined in [`python/internal/auto_batching/stackless.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/stackless.py).

<!-- Placeholder for "Used in" -->

Each resumption point is a pair of program counter to resume, and mask of
threads that are waiting there.

This class is a simple wrapper around Python's standard heapq implementation
of priority queues.  There are just two subtleties:

- Dequeue gets all the threads that were waiting at that point, by coalescing
  multiple entries if needed.

- Enqueue drops entries with empty masks, because they need never be resumed.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(backend)
```

Initialize self.  See help(type(self)) for accurate signature.




## Methods

<h3 id="dequeue"><code>dequeue</code></h3>

``` python
dequeue()
```




<h3 id="enqueue"><code>enqueue</code></h3>

``` python
enqueue(
    program_counter,
    mask
)
```






