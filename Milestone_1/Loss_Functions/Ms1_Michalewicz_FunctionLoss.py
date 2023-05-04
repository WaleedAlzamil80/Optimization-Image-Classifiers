import numpy as np
import tensorflow as tf


def Michalewicz(X, m=10):
    """
    Computes the value of the Michalewicz function at a given point x using TensorFlow.

    Parameters:
        X (tf.Tensor): A TensorFlow tensor representing the decision variables.
        m (int): A constant value used in the Michalewicz function. Default is 10.

    Returns:
        tf.Tensor: The value of the Michalewicz function at X.
    """

    if tf.rank(X) == 3:
        i = np.arange(1, X.shape[0] + 1).reshape(X.shape[0], 1, 1) + np.zeros(
            (1, X.shape[1], X.shape[2]), dtype=np.float32
        )
    else:
        i = np.array(range(X.shape[0]), dtype=np.float32).reshape(X.shape[0], 1) + 1

    i = tf.Variable(i, dtype=tf.float32)
    X = tf.cast(X, dtype=tf.float32)

    M = tf.sin(tf.math.multiply(tf.math.multiply(X, X), i) / np.pi) ** (2 * m)
    S = tf.math.multiply(tf.sin(X), M)

    return -1 * tf.reduce_sum(S, 0)
