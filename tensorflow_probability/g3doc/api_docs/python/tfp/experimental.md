<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.experimental


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



TensorFlow Probability API-unstable package.

<!-- Placeholder for "Used in" -->

This package contains potentially useful code which is under active development
with the intention eventually migrate to TFP proper. All code in
<a href="../tfp/experimental.md"><code>tfp.experimental</code></a> should be of production quality, i.e., idiomatically
consistent, well tested, and extensively documented. <a href="../tfp/experimental.md"><code>tfp.experimental</code></a> code
relaxes the TFP non-experimental contract in two regards:
1. <a href="../tfp/experimental.md"><code>tfp.experimental</code></a> has no API stability guarantee. The public footprint of
   <a href="../tfp/experimental.md"><code>tfp.experimental</code></a> code may change without notice or warning.
2. Code outside <a href="../tfp/experimental.md"><code>tfp.experimental</code></a> cannot depend on code within
   <a href="../tfp/experimental.md"><code>tfp.experimental</code></a>.

You are welcome to try any of this out (and tell us how well it works for you!).

## Modules

[`auto_batching`](../tfp/experimental/auto_batching.md) module: TensorFlow Probability auto-batching package.

[`edward2`](../tfp/experimental/edward2.md) module: Edward2 probabilistic programming language.

[`mcmc`](../tfp/experimental/mcmc.md) module: TensorFlow Probability experimental NUTS package.

[`substrates`](../tfp/experimental/substrates.md) module: TensorFlow Probability alternative substrates.

[`vi`](../tfp/experimental/vi.md) module: Experimental methods and objectives for variational inference.

