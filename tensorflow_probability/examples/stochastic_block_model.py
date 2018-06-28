import time
import functools as ft
from matplotlib import pylab as plt
import numpy as np
from observations import karate

from sklearn.metrics.cluster import adjusted_rand_score
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd


# Data & parameters
# -----------------------------------------------------------------------------
x_data, z_true = karate('~/data')
V = x_data.shape[0]
K = 2
starter_learning_rate = 1e-2
max_steps = 1000000
epsilon= 1e-5
tf.set_random_seed(42)

def joint_log_prob(data, priors, qgamma, qpi, qz):
    """ Compute the joint log-likelihood of the model.

    Parameters
    ----------
    data: np.array, shape (V, V)
        The adjacency matrix of the graph.
    priors: dict of tf.distribution
        The prior distributions of the model.
    qgamma: tensorflow variable
        Values that the gamma variable takes.
    qpi: tensorflow variable
        Values taken by the pi variable.
    qz: tensorflow variable
        Values taken by the z variable.
    """
    pi = priors['pi']
    gamma = priors['gamma']
    z = tfd.Multinomial(total_count=tf.ones(data.shape[0]),
                        probs=qgamma)
    x = tfd.Bernoulli(probs=tf.matmul(qz, tf.matmul(qpi, tf.transpose(qz))))

    log_prob_parts = [
            gamma.log_prob(qgamma),
            pi.log_prob(qpi),
            z.log_prob(qz),
            x.log_prob(data)
            ]

    log_prob = 0.
    for prob in log_prob_parts:
        log_prob += tf.reduce_sum(prob)

    return -log_prob


# Model
# -----------------------------------------------------------------------------
gamma = tfd.Dirichlet(concentration=tf.ones([K]))
pi = tfd.Beta(concentration0=tf.ones([K,K]),
              concentration1=tf.ones([K,K]))

priors = {'gamma': gamma,
          'pi': pi}

qgamma = tf.nn.softmax(tf.get_variable('qgamma/params', [K])) # must sum to one
qpi = tf.nn.sigmoid(tf.get_variable('qpi/params', [K, K])) # must be between 0 and 1
qz = tf.nn.softmax(tf.get_variable('qz/param', [V, K]))

log_prob = ft.partial(joint_log_prob, data=x_data, priors=priors)
loss = log_prob(qgamma=qgamma, qpi=qpi, qz=qz)


# Inference
# -----------------------------------------------------------------------------
global_step = tf.Variable(0, trainable=False, name='global_step')
learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                          global_step,
                                          100, 0.9, staircase=True)
optimizer = tf.train.AdagradOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

loss_vals = []
with tf.Session() as session:
    start = time.time()
    session.run(tf.global_variables_initializer())


    for step in range(max_steps):
        _, loss_value = session.run([train_op, loss])
        duration = time.time() - start
        if step % 1000 == 0:
            print("Step: {:>3d} Loss: {:.3f} ({:.3f} sec)".format(step, 
                                                                  loss_value,
                                                                  duration))
        if step > 0:
            if abs(loss_vals[-1]-loss_value) <  epsilon:
                break
        loss_vals.append(loss_value)

    # Criticism
    z_pred = session.run(qz).argmax(axis=1)
    print("Adjusted Rand Index=", adjusted_rand_score(z_pred, z_true))

    print(z_pred)
    print(z_true)


fig = plt.figure()
plt.plot(range(len(loss_vals)), loss_vals)
plt.show()
