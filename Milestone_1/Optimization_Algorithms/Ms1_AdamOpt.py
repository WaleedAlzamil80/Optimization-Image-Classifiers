import numpy as np
import tensorflow as tf

def Adam(loss_func, X_init, V_init, S_init, X_vals, loss_vals, eta=0.01, beta_1=0.9, beta_2=0.99, eps=1e-8, bias_correction=False, t=1):
    """
    Adam optimization algorithm for updating the values of a given variable.

    Args:
        - loss_func: Callable, the loss function that computes the loss given the current variable values.
        - X_init: tf.Variable, the initial variable values to be updated.
        - V_init: tf.Variable, the initial values for the first moment estimates (momentum).
        - S_init: tf.Variable, the initial values for the second moment estimates (RMSprop).
        - eta: float, the learning rate used to control the size of the update steps (default is 0.01).
        - beta_1: float, the amount used to control how the momentum (exponentially weighted moving averages)
                  depending on the last values (default is 0.9)
        - beta_2: float, the amount used to control how the learning rate depend on the last values (default is 0.99)
        - eps: float, Ensure S in not zero by adding this number to it (default is 1e-8)
        - bias_correction: boolean, to limit the first iteration from being biased (default is True)
        - t: int, iteration number for bias correction
        - X_vals: list(), empty list used to save X at each iteration.
        - loss_vals: list(), empty list used to save the loss values at each iteration.

    Returns:
        None
    """

    with tf.GradientTape(persistent=True) as tape:
        # Compute the current loss
        current_loss = loss_func(X_init)

    # Compute the gradient of the loss with respect to the variables
    dx = tape.gradient(current_loss, X_init)

    # Compute the first and second moment estimates (momentum and RMSprop)
    V = (beta_1 * V_init) + (1 - beta_1) * dx      # momentum
    S = (beta_2 * S_init) + (1 - beta_2) * (dx**2) # RMSprop

    # bias correction
    if (bias_correction):
        V_corrected = V / (1 - (beta_1**(t)))
        S_corrected = S / (1 - (beta_2**(t)))
    else:
        V_corrected = V
        S_corrected = S

    # update the variables using Adam update rule
    X_init.assign_sub(eta * V_corrected / (tf.sqrt(S_corrected) + eps))

    # update the first and second moment estimates (momentum and RMSprop)
    V_init.assign(V)
    S_init.assign(S)

    # calc the loss value and append it to the list
    loss_vals.append(tf.squeeze (loss_func(X_init)))

    #append X to the list
    X_vals.append(X_init.numpy())