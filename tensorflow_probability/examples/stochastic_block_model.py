import time
import functools as ft
from matplotlib import pylab as plt
import numpy as np
from observations import karate

from sklearn.metrics.cluster import adjusted_rand_score
import tensorflow as tf
from tensorflow_probability import edward2 as ed


# Data & parameters
# -----------------------------------------------------------------------------
x_data, z_true = karate('~/data')
V = x_data.shape[0]
K = 2

learning_rate = 1e-2
max_steps = 1000000
epsilon= 1e-5
tf.set_random_seed(42)


# Model
# -----------------------------------------------------------------------------
def stochastic_block_model(V, K):
    """ Stochastic block model.
    """
    gamma = ed.Dirichlet(concentration=tf.ones([K]), name='gamma')
    pi = ed.Beta(concentration0=tf.ones([K,K]),
                 concentration1=tf.ones([K,K]),
                 name='pi')
    z = ed.Multinomial(total_count=tf.ones([V]),
                       probs=gamma,
                       name='z')
    x = ed.Bernoulli(probs=tf.matmul(z, tf.matmul(pi, tf.transpose(z))),
                     name='x')
    return x

qgamma = tf.nn.softmax(tf.get_variable('qgamma/params', [K])) # must sum to one
qpi = tf.nn.sigmoid(tf.get_variable('qpi/params', [K, K])) # must be between 0 and 1
qz = tf.nn.softmax(tf.get_variable('qz/param', [V, K]))

log_joint = ed.make_log_joint_fn(stochastic_block_model)
loss = -log_joint(V, K, x=x_data, gamma=qgamma, pi=qpi, z=qz)


# Inference
# -----------------------------------------------------------------------------
optimizer = tf.train.AdagradOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

loss_vals = []
with tf.Session() as session:
    start = time.time()
    tf.global_variables_initializer().run()
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
