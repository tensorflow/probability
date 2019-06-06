<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.dsl" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.experimental.auto_batching.dsl

Python-embedded DSL frontend for authoring autobatchable IR programs.

### Aliases:

* Module `tfp.experimental.auto_batching.dsl`
* Module `tfp.experimental.auto_batching.frontend.dsl`



Defined in [`python/internal/auto_batching/dsl.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/dsl.py).

<!-- Placeholder for "Used in" -->

This domain-specific language frontend serves two purposes:
- Being easier and more pleasant to author by humans than writing IR directly
- Being close enough to the structure of natural Python programs that
  the remaining gap can be bridged by a source-to-source transformation

## Classes

[`class ProgramBuilder`](../../../tfp/experimental/auto_batching/dsl/ProgramBuilder.md): An auto-batching DSL context.

