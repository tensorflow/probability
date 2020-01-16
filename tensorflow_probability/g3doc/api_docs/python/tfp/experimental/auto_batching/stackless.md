<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.stackless" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.experimental.auto_batching.stackless


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/auto_batching/stackless.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



A stackless auto-batching VM.

### Aliases:

* Module `tfp.experimental.auto_batching.frontend.st`


<!-- Placeholder for "Used in" -->

Borrows the stack, and conditional execution, from the host Python; manages only
divergence.

## Modules

[`inst`](../../../tfp/experimental/auto_batching/instructions.md) module: Instruction language for auto-batching virtual machine.

## Classes

[`class ExecutionQueue`](../../../tfp/experimental/auto_batching/stackless/ExecutionQueue.md): A priority queue of resumption points.

## Functions

[`execute(...)`](../../../tfp/experimental/auto_batching/stackless/execute.md): Executes a given program in stackless auto-batching mode.

[`is_running(...)`](../../../tfp/experimental/auto_batching/stackless/is_running.md): Returns whether the stackless machine is running a computation.

