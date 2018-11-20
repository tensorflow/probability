#### Copyright 2018 The TensorFlow Authors.

```none
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

# The `joint_log_prob` callable: connecting model building and inference in TFP.

The goal of probabilistic inference is to find a probability distribution
consistent with one or more observed values. TensorFlow Probability (TFP) offers
different tools for probabilistic inference, which are all based on a
user-defined Python `callable`: a `joint_log_prob` function. This callable
returns a single `Tensor` representing the log of the joint probability of a
given set of concretized values of some model's random variables. The
concretized values must themselves also be `Tensor`s and are specified by the
callable's function signature.

All `joint_log_prob` functions have the following structure:

1. The function takes a set of **inputs** representing the values of the random
   variables in the model. These inputs must be convertible to `Tensor`s.
   Typically the function would support computing batches of inputs, i.e., a
   vector of inputs. (For more information on batches see [TensorFlow
   Distributions](https://arxiv.org/abs/1711.10604), section 3.3.)

2. Internally the `joint_log_prob` function uses probability distributions to
   **measure** the plausibility of the given set of values. By convention, we
   recommend that the distribution which measures the plausibility of the value
   `foo` be named `rv_foo`. Distributions may or may not be composed from values
   associated with other random variables; we use the following names to
   distinguish these two cases:

   a. A **prior distribution** is an unconditional measure of the likelihood of
      an input value. It is not parameterized by other input values.
      Mathematically, a prior probability is written as `p(X=x)`.

   b. A **conditional distribution** measures the likelihood of an input value
      but is parameterized by other input values. Mathematically, a conditional
      probability is written as `p(Y=y|X=x)`.

3. The function returns the **total log probability** of the inputs. The total
   log probability is the sum of the log probabilities as measured by the
   model's constituent prior and conditional distributions. As a convention, we
   measure model values using log probabilities rather than "straight"
   probabilities. Since probabilities have a very large dynamic range, computing
   log probabilities is often more numerically stable.

Consider the following example.

```python
def joint_log_prob(heads, coin_bias):
    """Joint log probability of coin flips under a non-informative prior.

    Args:
      heads: `Tensor` of coin flips; a `0` or `1`.
      coin_bias: `Tensor` of possible coin biases.

    Returns:
      joint_log_prob: The joint log probability measure of given coin flips and
        coin bias.
    """
    rv_coin_bias = tfp.distributions.Uniform(low=0., high=1.)  # Prior
    rv_heads = tfp.distributions.Bernoulli(probs=coin_bias)    # Conditional
    return (rv_coin_bias.log_prob(coin_bias) +
            tf.reduce_sum(rv_heads.log_prob(heads), axis=-1))
```

In this example, the input values are observed outcomes from a coin flipping
experiment and the coin bias (a number in `[0, 1]` representing the chance of
heads). The `joint_log_prob` takes the provided values of `heads` and
`coin_bias` and computes the overall log probability of these values.

1. The prior distribution, `rv_coin_bias`, measures the plausibility of the
   provided `coin_bias`.

2. The conditional distribution, `rv_heads`, measures the plausibility of
 `heads`, assuming `coin_bias` is the probability of heads.

The resulting sum of the log of these probabilities is the joint log
probability.

The `joint_log_prob` is particularly useful in concert with the
[`tfp.mcmc`](https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc)
and [`tfp.vi`](https://www.tensorflow.org/probability/api_docs/python/tfp/vi)
modules. Markov chain Monte Carlo (MCMC) algorithms are useful for sampling from
implicitly normalized probability measures, i.e., they need only an unnormalized
log-probability function to draw samples from the corresponding normalized log-
probability.

Although we generically refer to this callable as the `joint_log_prob`, it is
not presumed normalized. In other words, the sum of probabilities over all
possible combinations of its arguments does not necessarily equal `1`.  This
relaxation is useful, for example, when using MCMC to generate samples
from the posterior distribution, i.e., `p(x|y) = p(y|x)p(x) / p(y)` where `p(y)
= integral{ p(y|x)p(x) : x in X }`. If `p(y)` is intractable (and it often is),
the [`tfp.mcmc`](
https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc) module can
generate samples from the normalized posterior log-probability using only the
unnormalized posterior log-probability. For example, returning to our coin
flipping example, the unnormalized (`heads`) posterior log-probability is:

```python
unnormalized_posterior_log_prob = lambda coin_bias: joint_log_prob(heads, coin_bias)
```

Notice that this closure "pins" the `joint_log_prob` at some value of `heads`
and is only a function of `coin_bias`. If we had normalized this "pinned"
function, we'd obtain the normalized posterior log-probability. To normalize
this pinned function we could calculate the sum of all possible joint
probabilities over all possible combinations of `heads`.
