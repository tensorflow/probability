# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Trains a grammar variational auto-encoder on synthetic data.

The grammar variational auto-encoder (VAE) [1] posits a generative model over
productions from a context-free grammar, and it posits an amortized variational
approximation for efficient posterior inference. We train the grammar VAE
on synthetic data using the grammar from [1] (Figure 1). Note for real data
analyses, one should implement a parser to convert examples into lists of
production rules.

This example showcases eager execution in order to train a model where data
points have a variable number of time steps (that is, without padding). However,
note that handling a variable number of time steps requires a batch size of 1.
In this example, we assume data points arrive in a stream, one at a time. Such a
setting has an unbounded maximum length which prevents padding.

Summaries are written under the flag `model_dir`. Point TensorBoard to that
directory in order to monitor progress.

Example output:

```none
Random examples from synthetic data distribution:
222N1N21c
1c2N2C2C12C1N
C11C12c
2C
NCC

Step:   0 Loss: -13.724 (0.494 sec)
Step: 500 Loss: -0.004 (145.741 sec)
Step: 1000 Loss: -0.000 (292.205 sec)
Step: 1500 Loss: -0.000 (438.819 sec)
```

#### References

[1]: Matt J. Kusner, Brooks Paige, and Jose Miguel Hernandez-Lobato. Grammar
     Variational Autoencoder. In _International Conference on Machine Learning_,
     2017. https://arxiv.org/abs/1703.01925
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

# Dependency imports
from absl import flags
import six
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from tensorflow_probability import edward2 as ed

flags.DEFINE_float("learning_rate",
                   default=1e-4,
                   help="Initial learning rate.")
flags.DEFINE_integer("max_steps",
                     default=5000,
                     help="Number of training steps to run.")
flags.DEFINE_integer("latent_size",
                     default=128,
                     help="Number of dimensions in the latent code.")
flags.DEFINE_integer("num_units",
                     default=256,
                     help="Number of units in the generative model's LSTM.")
flags.DEFINE_string("model_dir",
                    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
                                         "grammar_vae/"),
                    help="Directory to put the model's fit.")

FLAGS = flags.FLAGS


class SmilesGrammar(object):
  """Context-free grammar for SMILES strings.

  A context-free grammar is a 4-tuple consisting of the following elements:

  + `nonterminal_symbols`: finite set of strings.
  + `alphabet`: finite set of strings (terminal symbols). It is disjoint from
      `nonterminal_symbols`.
  + `production_rules`: list of 2-tuples. The first and second elements of
      each tuple respectively denote the left-hand-side and right-hand-side of a
      production rule. All right-hand-sides are written as lists, since the
      number of right-hand-side symbols may be greater than 1.
  + `start_symbol`: string, a distinct nonterminal symbol.
  """

  @property
  def nonterminal_symbols(self):
    return {"smiles", "chain", "branched atom", "atom", "ringbond",
            "aromatic organic", "aliphatic organic", "digit"}

  @property
  def alphabet(self):
    return {"c", "C", "N", "1", "2"}

  @property
  def production_rules(self):
    return [
        ("smiles", ["chain"]),
        ("chain", ["chain", "branched atom"]),
        ("chain", ["branched atom"]),
        ("branched atom", ["atom", "ringbond"]),
        ("branched atom", ["atom"]),
        ("atom", ["aromatic organic"]),
        ("atom", ["aliphatic organic"]),
        ("ringbond", ["digit"]),
        ("aromatic organic", ["c"]),
        ("aliphatic organic", ["C"]),
        ("aliphatic organic", ["N"]),
        ("digit", ["1"]),
        ("digit", ["2"]),
    ]

  @property
  def start_symbol(self):
    return "smiles"

  def convert_to_string(self, productions):
    """Converts a sequence of productions into a string of terminal symbols.

    Args:
      productions: Tensor of shape [1, num_productions, num_production_rules].
        Slices along the `num_productions` dimension represent one-hot vectors.

    Returns:
      str that concatenates all terminal symbols from `productions`.

    Raises:
      ValueError: If the first production rule does not begin with
        `self.start_symbol`.
    """
    symbols = []
    for production in tf.unstack(productions, axis=1):
      lhs, rhs = self.production_rules[
          tf.argmax(input=tf.squeeze(production), axis=-1)]
      if not symbols:  # first iteration
        if lhs != self.start_symbol:
          raise ValueError("`productions` must begin with `self.start_symbol`.")
        symbols = rhs
      else:
        # Greedily unroll the nonterminal symbols based on the first occurrence
        # in a linear sequence.
        index = symbols.index(lhs)
        symbols = symbols[:index] + rhs + symbols[index + 1:]
    string = "".join(symbols)
    return string

  def mask(self, symbol, on_value, off_value):
    """Produces a masking tensor for (in)valid production rules.

    Args:
      symbol: str, a symbol in the grammar.
      on_value: Value to use for a valid production rule.
      off_value: Value to use for an invalid production rule.

    Returns:
      Tensor of shape [1, num_production_rules]. An element is `on_value`
      if its corresponding production rule has `symbol` on its left-hand-side;
      the element is `off_value` otherwise.
    """
    mask_values = [on_value if lhs == symbol else off_value
                   for lhs, _ in self.production_rules]
    mask_values = tf.reshape(mask_values, [1, len(self.production_rules)])
    return mask_values


class ProbabilisticGrammar(tf.keras.Model):
  """Deep generative model over productions that follow a grammar."""

  def __init__(self, grammar, latent_size, num_units):
    """Constructs a probabilistic grammar.

    Args:
      grammar: An object representing a grammar. It has members
        `nonterminal_symbols`, `alphabet`, `production_rules`, and
        `start_symbol`, and a method `mask` determining (in)valid
        production rules given a symbol.
      latent_size: Number of dimensions in the latent code.
      num_units: Number of units in the LSTM cell.
    """
    super(ProbabilisticGrammar, self).__init__()
    self.grammar = grammar
    self.latent_size = latent_size
    self.lstm = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units)
    self.output_layer = tf.keras.layers.Dense(len(grammar.production_rules))

  def __call__(self, *args, **kwargs):
    inputs = 0.  # fixes a dummy variable so Model can be called without inputs
    return super(ProbabilisticGrammar, self).__call__(inputs, *args, **kwargs)

  def call(self, inputs):
    """Runs the model forward to generate a sequence of productions.

    Args:
      inputs: Unused.

    Returns:
      productions: Tensor of shape [1, num_productions, num_production_rules].
        Slices along the `num_productions` dimension represent one-hot vectors.
    """
    del inputs  # unused
    latent_code = ed.MultivariateNormalDiag(loc=tf.zeros(self.latent_size),
                                            sample_shape=1,
                                            name="latent_code")
    state = self.lstm.zero_state(1, dtype=tf.float32)
    t = 0
    productions = []
    stack = [self.grammar.start_symbol]
    while stack:
      symbol = stack.pop()
      net, state = self.lstm(latent_code, state)
      logits = (self.output_layer(net) +
                self.grammar.mask(symbol, on_value=0., off_value=-1e9))
      production = ed.OneHotCategorical(logits=logits,
                                        name="production_" + str(t))
      _, rhs = self.grammar.production_rules[tf.argmax(
          input=tf.squeeze(production), axis=-1)]
      for symbol in rhs:
        if symbol in self.grammar.nonterminal_symbols:
          stack.append(symbol)
      productions.append(production)
      t += 1
    return tf.stack(productions, axis=1)


class ProbabilisticGrammarVariational(tf.keras.Model):
  """Amortized variational posterior for a probabilistic grammar."""

  def __init__(self, latent_size):
    """Constructs a variational posterior for a probabilistic grammar.

    Args:
      latent_size: Number of dimensions in the latent code.
    """
    super(ProbabilisticGrammarVariational, self).__init__()
    self.latent_size = latent_size
    self.encoder_net = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, 3, padding="SAME"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(tf.nn.elu),
        tf.keras.layers.Conv1D(128, 3, padding="SAME"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(tf.nn.elu),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(latent_size * 2, activation=None),
    ])

  def call(self, inputs):
    """Runs the model forward to return a stochastic encoding.

    Args:
      inputs: Tensor of shape [1, num_productions, num_production_rules]. It is
        a sequence of productions of length `num_productions`. Each production
        is a one-hot vector of length `num_production_rules`: it determines
        which production rule the production corresponds to.

    Returns:
      latent_code_posterior: A random variable capturing a sample from the
        variational distribution, of shape [1, self.latent_size].
    """
    net = self.encoder_net(tf.cast(inputs, tf.float32))
    return ed.MultivariateNormalDiag(
        loc=net[..., :self.latent_size],
        scale_diag=tf.nn.softplus(net[..., self.latent_size:]),
        name="latent_code_posterior")


def main(argv):
  del argv  # unused
  if tf.io.gfile.exists(FLAGS.model_dir):
    tf.compat.v1.logging.warning(
        "Warning: deleting old log directory at {}".format(FLAGS.model_dir))
    tf.io.gfile.rmtree(FLAGS.model_dir)
  tf.io.gfile.makedirs(FLAGS.model_dir)
  tf.compat.v1.enable_eager_execution()

  grammar = SmilesGrammar()
  synthetic_data_distribution = ProbabilisticGrammar(
      grammar=grammar, latent_size=FLAGS.latent_size, num_units=FLAGS.num_units)

  print("Random examples from synthetic data distribution:")
  for _ in range(5):
    productions = synthetic_data_distribution()
    string = grammar.convert_to_string(productions)
    print(string)

  probabilistic_grammar = ProbabilisticGrammar(
      grammar=grammar, latent_size=FLAGS.latent_size, num_units=FLAGS.num_units)
  probabilistic_grammar_variational = ProbabilisticGrammarVariational(
      latent_size=FLAGS.latent_size)

  checkpoint = tf.train.Checkpoint(
      synthetic_data_distribution=synthetic_data_distribution,
      probabilistic_grammar=probabilistic_grammar,
      probabilistic_grammar_variational=probabilistic_grammar_variational)
  global_step = tf.compat.v1.train.get_or_create_global_step()
  optimizer = tf.compat.v1.train.AdamOptimizer(FLAGS.learning_rate)
  writer = tf.compat.v2.summary.create_file_writer(FLAGS.model_dir)
  writer.set_as_default()

  start_time = time.time()
  for step in range(FLAGS.max_steps):
    productions = synthetic_data_distribution()
    with tf.GradientTape() as tape:
      # Sample from amortized variational distribution and record its trace.
      with ed.tape() as variational_tape:
        _ = probabilistic_grammar_variational(productions)

      # Set model trace to take on the data's values and the sample from the
      # variational distribution.
      values = {"latent_code": variational_tape["latent_code_posterior"]}
      values.update({"production_" + str(t): production for t, production
                     in enumerate(tf.unstack(productions, axis=1))})
      with ed.tape() as model_tape:
        with ed.interception(ed.make_value_setter(**values)):
          _ = probabilistic_grammar()

      # Compute the ELBO given the variational sample, averaged over the batch
      # size and the number of time steps (number of productions). Although the
      # ELBO per data point sums over time steps, we average in order to have a
      # value that remains on the same scale across batches.
      log_likelihood = 0.
      for name, rv in six.iteritems(model_tape):
        if name.startswith("production"):
          log_likelihood += rv.distribution.log_prob(rv.value)

      kl = tfp.distributions.kl_divergence(
          variational_tape["latent_code_posterior"].distribution,
          model_tape["latent_code"].distribution)

      timesteps = tf.cast(productions.shape[1], dtype=tf.float32)
      elbo = tf.reduce_mean(input_tensor=log_likelihood - kl) / timesteps
      loss = -elbo
      with tf.compat.v2.summary.record_if(
          lambda: tf.math.equal(0, global_step % 500)):
        tf.compat.v2.summary.scalar(
            "log_likelihood",
            tf.reduce_mean(input_tensor=log_likelihood) / timesteps,
            step=global_step)
        tf.compat.v2.summary.scalar(
            "kl", tf.reduce_mean(input_tensor=kl) / timesteps, step=global_step)
        tf.compat.v2.summary.scalar("elbo", elbo, step=global_step)

    variables = (probabilistic_grammar.variables
                 + probabilistic_grammar_variational.variables)
    grads = tape.gradient(loss, variables)
    grads_and_vars = list(zip(grads, variables))
    optimizer.apply_gradients(grads_and_vars, global_step)

    if step % 500 == 0:
      duration = time.time() - start_time
      print("Step: {:>3d} Loss: {:.3f} ({:.3f} sec)".format(
          step, loss, duration))
      checkpoint.save(file_prefix=FLAGS.model_dir)

if __name__ == "__main__":
  tf.compat.v1.app.run()
