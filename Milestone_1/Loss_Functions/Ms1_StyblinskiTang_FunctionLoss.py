import numpy as np
import tensorflow as tf

def StyblinskiTang(X):
    """
    Computes the value of the StyblinskiTang function at a given point x using TensorFlow.

    Parameters:
        X (tf.Tensor): A TensorFlow tensor representing the decision variable.

    Returns:
        tf.Tensor: The value of the StyblinskiTang function at X.
    """

    f =  tf.subtract(tf.pow(X, 4), tf.multiply(16, tf.pow(X, 2)))
    s = tf.multiply(5, X)

    sum = tf.multiply(tf.add(f, s), 0.5)

    return tf.reduce_sum(sum)
