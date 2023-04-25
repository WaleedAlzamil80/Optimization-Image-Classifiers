import numpy as np
import tensorflow as tf

def MATYAS(X):
    """
    Computes the value of the MATYAS function at a given point x using TensorFlow.
    
    Parameters:
        X (tf.Tensor): A TensorFlow tensor representing the decision variable.
    
    Returns:
        tf.Tensor: The value of the MATYAS function at X.
    """

    X = tf.cast(X, dtype=tf.float32)  # Cast X to float32
    X1, X2 = X[0], X[1]
    x1sq = tf.math.multiply(X1, X1)
    x2sq = tf.math.multiply(X2, X2)
    sum1 = tf.math.multiply(tf.math.add(x1sq, x2sq), 0.26)

    sum2 = tf.math.multiply(tf.math.multiply(X1,X2), -0.48)

    result = tf.math.add(sum1, sum2)

    return result