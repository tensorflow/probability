import tensorflow as tf
import tensorflow_probability as tfp

def quadratic_loss_and_grad(x):
    """f(x) = (x-2)^2 + (y-3)^2, minimum at [2, 3]"""
    diff = x - tf.constant([2.0, 3.0])
    loss = tf.reduce_sum(tf.square(diff))
    grad = 2.0 * diff
    return loss, grad

print("Testing line_search_kwargs feature...")

# Test 1: Default behavior
start = tf.constant([0.0, 0.0])
results_default = tfp.optimizer.bfgs_minimize(
    quadratic_loss_and_grad, start, tolerance=1e-10)
print("Default result:", results_default.position.numpy())

# Test 2: With custom line search parameters
custom_kwargs = {
    'initial_step_size': 0.5,
    'threshold_use_approximate_wolfe_condition': 1e-4
}
results_custom = tfp.optimizer.bfgs_minimize(
    quadratic_loss_and_grad, start, tolerance=1e-10,
    line_search_kwargs=custom_kwargs)
print("Custom result:", results_custom.position.numpy())

# Both should be close to [2.0, 3.0]
expected = tf.constant([2.0, 3.0])
assert tf.reduce_all(tf.abs(results_default.position - expected) < 1e-6)
assert tf.reduce_all(tf.abs(results_custom.position - expected) < 1e-6)
print("✅ All tests passed!")