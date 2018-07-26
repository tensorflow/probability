Project: /probability/_project.yaml
Book: /probability/_book.yaml
page_type: reference
<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.mcmc.default_exchange_proposed_fn" />
</div>

# tfp.mcmc.default_exchange_proposed_fn

``` python
tfp.mcmc.default_exchange_proposed_fn(probs)
```

Default function for `exchange_proposed_fn` of `kernel`.

Depending on the probability of `probs`, decide whether to propose
combinations of replica for exchange.
When exchanging, create combinations of adjacent replicas from 0 or 1 index.

#### Args:

* <b>`probs`</b>: A float-like Tensor which represents the probability of proposing
    combinations of replicas for exchange.


#### Returns:

* <b>`default_exchange_proposed_fn_`</b>: Python callable which take a number of
    replicas, and return combinations of replicas for exchange and a number of
    combinations.