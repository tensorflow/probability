<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.dsl" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.experimental.auto_batching.dsl


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/auto_batching/dsl.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Python-embedded DSL frontend for authoring autobatchable IR programs.

### Aliases:

* Module `tfp.experimental.auto_batching.frontend.dsl`


<!-- Placeholder for "Used in" -->

This domain-specific language frontend serves two purposes:
- Being easier and more pleasant to author by humans than writing IR directly
- Being close enough to the structure of natural Python programs that
  the remaining gap can be bridged by a source-to-source transformation

## Classes

[`class ProgramBuilder`](../../../tfp/experimental/auto_batching/dsl/ProgramBuilder.md): An auto-batching DSL context.

