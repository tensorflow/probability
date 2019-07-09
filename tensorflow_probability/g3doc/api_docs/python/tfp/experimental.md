<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.experimental

TensorFlow Probability API-unstable package.



Defined in [`python/experimental/__init__.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/experimental/__init__.py).

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

[`mcmc`](../tfp/experimental/mcmc.md) module: TensorFlow Probability Google-internal NUTS package.

