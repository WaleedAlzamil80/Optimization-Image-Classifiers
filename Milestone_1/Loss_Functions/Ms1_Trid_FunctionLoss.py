import numpy as np
import tensorflow as tf

def Trid(X):
    """
    Computes the value of the Trid function at a given point x using TensorFlow.

    Parameters:
        X (tf.Tensor): A TensorFlow tensor representing the decision variables.

    Returns:
        tf.Tensor: The value of the Michalewicz function at X.
    """
    sum0 = 0
    for i in range(X.shape[0]):
        if i == (X.shape[0] - 1):
            break
        sum0 += X[i+1]*X[i]
    return (tf.reduce_sum((X - 1)**2, 0) - sum0)


def TridOptmValue(X):
    """
    Computes the value of the Trid function using the diven formula to test the optimizers.

    Parameters:
        X (tf.Tensor): A TensorFlow tensor representing the decision variables.

    Returns:
        float: The value of the optimum X that the optimzer must reach.
    """
    d= X.shape[0]
    return (-d * (d + 4) * (d - 1) / 6)
