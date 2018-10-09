The goal of probabilistic inference is to find model parameters that may explain
data you have observed. TFP performs probabilistic inference by evaluating the
model parameters using a `joint_log_prob` function. The idea behind the
`joint_log_prob` is simple: it is a function that takes in all of the values
that specify your model and data, and it returns the likelihood that the
parameterized model generated the data. TFP is built around defining this joint
log probability and then identifying the distribution of model parameters that
may have generated the data.

All `joint_log_prob` functions have a common structure:

1. The function takes a set of **inputs** to evaluate. Each input is either an
observed value or a random variate. (A random variate is a possible sample from
one of the model's distributions.) The purpose of the function is to evaluate
how likely the inputs are with respect to the model defined in the function’s
code.

1. The **model** used to evaluate the inputs is defined with a set of
distributions. These distributions measure the likelihood of the input values.
(By convention, the distribution that measures the likelihood of the variable
`foo` will be named `rv_foo` to note that it is a random variable.) We use two
types of distributions in `joint_log_prob` functions:

 a. **Prior distributions** measure the likelihood of input values individually.
 Prior distributions never depend on input values, and are used to measure the
 likelihood of a single value. Each unknown variable—one that has not been
 observed directly—needs a corresponding prior. Beliefs about which values could
 be reasonable determine the prior distribution. Choosing a prior can be tricky,
 so we will cover it in depth in Chapter 6.

 b. **Conditional distributions** measure the likelihood of an input value given
 other input values. The typical case is measuring the likelihood of observed
 data given the current guess of a parameter in the model. The `joint_log_prob`
 computes conditional distributions from input values. The conditional
 distributions can then return the likelihood of another input value, often the
 observed data.

1. Finally, we calculate and return the **joint log probability** of the inputs.
The joint log probability is the sum of the log probabilities from all of the
prior and conditional distributions. (We take the sum of log probabilities
instead of multiplying the probabilities directly for reasons of numerical
stability: floating point numbers in computers cannot represent the very small
values necessary to calculate the joint log probability unless they are in the
log domain.) The sum is actually an unnormalized density; although the total
joint probability over all possible inputs might not sum to one, it is
proportional to the true probability density. This proportional distribution is
sufficient to estimate the distribution of likely inputs.

Let's look at an example:

```python
def joint_log_prob(total_count, num_heads, heads_prob):
  rv_heads_prob = tfd.Uniform(low=0., high=1.)
  rv_num_heads = tfd.Binomial(total_count=total_count, probs=heads_prob)
  return (rv_heads_prob.log_prob(heads_prob) +
          rv_num_heads.log_prob(num_heads))
```

We can map these terms onto the code above. In this example, the input values
are the observed values in `num_heads` and `total_count` and the unknown value
for `heads_prob`. The `joint_log_prob` takes the current guess for `heads_prob`
and answers, what is the likelihood of the data, `num_heads` and `total_count`,
if `heads_prob` is the probability of a coin landing on heads. The answer
depends on two distributions. The prior distribution, `rv_heads_prob`, indicates
how likely the current value of `heads_prob` is by itself. The conditional
distribution, `rv_num_heads`, indicates the likelihood of `num_heads` if
`heads_prob` were the  probability for the Binomial distribution. The sum of
these log probabilities is the joint log probability.

The `joint_log_prob` is particularly useful in conjunction with the `tfp.mcmc`
package. MCMC algorithms proceed by making educated guesses about the unknown
test values (we’ll talk about how it makes those guesses in Chapter 3) and
computing what the likelihood of this set of arguments is. By repeating this
many times, MCMC builds a distribution of likely parameters. Constructing this
distribution is the goal of probabilistic inference.
