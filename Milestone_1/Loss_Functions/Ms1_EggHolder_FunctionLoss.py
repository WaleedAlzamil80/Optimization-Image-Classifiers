import numpy as np
import tensorflow as tf

def EggHolder(X):
    """
    Computes the value of the EggHolder function at a given point x using TensorFlow.

    Parameters:
        X (tf.Tensor): A TensorFlow tensor representing the decision variable.

    Returns:
        tf.Tensor: The value of the EggHolder function at X.
    """
    X = tf.cast(X, dtype=tf.float32)
    x1, x2 = X[0], X[1]

    term1 = tf.multiply(-1 * (x2 + 47), tf.sin(tf.sqrt(tf.abs(x2 + tf.multiply(x1, 0.5) + 47))))
    term2 = tf.multiply(x1, tf.sin(tf.sqrt(tf.abs(x1 - (x2 + 47)))))

    return (term1 - term2)
