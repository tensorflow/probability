<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.mcmc.default_exchange_proposed_fn" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.mcmc.default_exchange_proposed_fn

Default exchange proposal function, for replica exchange MC.

``` python
tfp.mcmc.default_exchange_proposed_fn(prob_exchange)
```



Defined in [`python/mcmc/replica_exchange_mc.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/mcmc/replica_exchange_mc.py).

<!-- Placeholder for "Used in" -->

With probability `prob_exchange` propose combinations of replica for exchange.
When exchanging, create combinations of adjacent replicas in
[Replica Exchange Monte Carlo](
https://en.wikipedia.org/wiki/Parallel_tempering)

```
exchange_fn = default_exchange_proposed_fn(prob_exchange=0.5)
exchange_proposed = exchange_fn(num_replica=3)

exchange_proposed.eval()
==> [[0, 1]]  # 1 exchange, 0 <--> 1

exchange_proposed.eval()
==> []  # 0 exchanges
```

#### Args:


* <b>`prob_exchange`</b>: Scalar `Tensor` giving probability that any exchanges will
  be generated.


#### Returns:


* <b>`default_exchange_proposed_fn_`</b>: Python callable which take a number of
  replicas (a Python integer), and return combinations of replicas for
  exchange as an [n, 2] integer `Tensor`, `0 <= n <= num_replica // 2`,
  with *unique* values in the set `{0, ..., num_replica}`.