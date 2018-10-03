A key idea in using TFP for Bayesian modelling is to use Python closures to
represent unnormalized posterior probabilities. In general, if you can define
the joint `log_prob` of your data under a generative model, `tfp.mcmc` can
sample from one partition of your model's variables, given the remaining.

Let's talk about the following `joint_log_prob` function for estimating the bias
of a coin after observing `num_heads` in `total_count` flips:

```python
def joint_log_prob(total_count, num_heads, heads_prob):
  rv_heads_prob = tfd.Uniform(low=0., high=1.)
  rv_num_heads = tfd.Binomial(total_count=total_count, probs=heads_prob)
  return (rv_heads_prob.log_prob(heads_prob) +
          rv_num_heads.log_prob(num_heads))
```

Typically a user-supplied `joint_log_prob` function has three parts:

1.  A function definition which takes as input, concretized values for model
    variables. These inputs can be regarded as **test values**, i.e., what``
    is the likelihood if the model variables had these values? Two types of test
    values are common: the observations fixed by the real world and the
    sampled values from the MCMC alorightm. The *observations*, `total_count`
    and `num_heads` in this example, are supplied by a closure before calling
    MCMC. In this example, the only sampled test value is our current guess at
    the probability of the coin landing on heads, `heads_prob`. During MCMC,
    this function will get sampled test values to evaluate. If there are
    multiple stochastic variables, this function will receive one test value for
    each variable. By making both the observed and sampled test variables
    arguments, this function can be used with MCMC or in other contexts without
    modification.

2.  Probability distributions which will measure the likelihood of each test
    variable. As a convention, the likelihood of the test value named `foo` is
    measured by a distribution named `rv_foo` (where `rv` is short for random
    variable). Some distributions may be constructed from test values others may
    not. I.e.:

 a.  "Prior" distributions are never constructed from test values. They are
     "fundamental" sources of stochasticity in our model. In this example, the
     prior, `rv_heads_prob`, is what values we think `heads_prob` may take: a
     uniform distribution between 0 and 1. The prior tells us how likely we
     thought different test values might be before we started. The prior helps
     MCMC search for values more efficiently.  Generally, you need one prior for
     each stochastic variable that we are optimizing over.

 b.  "Conditional" distributions are always constructed from test values. The
     manner by which they measure a test value depends on other test values.
     Commonly, we condition a distribution on the sampled test value to
     determine what the log likelihood of the fixed observation value is given
     that sampled test value. In this example, we think observation is a
     Binomial distribution, `rv_num_heads`, based on a value of `heads_prob` and
     `total_count`. The conditional distribution should only rely on test values
     passed in as arguments. It should not depend on the prior.

3.  Finally, we return the **joint log probability** of the test
    values (also called `joint_log_prob` for short). The `joint_log_prob` is
    the sum of the log-probability evaluations of every test value under its
    corresponding distribution.  In this example, the joint_log_prob breaks down
    into parts: the log probability from the prior is
    `rv_heads_prob.log_prob(heads_prob)`, and the log probability from the model
    is `rv_num_heads.log_prob(num_heads)`. The observatiosn, `num_heads`, is what
    ties our model to the observed reality. MCMC will work to find test values
    that are high likelihood under the prior *and* have a high likelihood of
    potentially generating the data we observed.


