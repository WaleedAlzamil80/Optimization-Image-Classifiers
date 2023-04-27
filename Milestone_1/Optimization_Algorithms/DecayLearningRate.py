import numpy as np
import tensorflow as tf

def DecayLearningRate(eta, t = 1, decay_rate = 0.01, decay_type = None)
    """
    Different techniques for the decay learning rate.

    Args:
      - eta: float, the learning rate used to control the size of the update steps.
      - t: int, the iteration number.
      - decay_type: string, the type of learning rate decay to apply (default is None)
                       possible values: 'time-based', 'step-based', 'exponential', 'performance'
      - decay_rate: float, the decay rate for exponential decay

    Returns:
      - eta: float, the new learning rate
      """

    if decay_type is not None:
      if decay_type == 'time-based':
        eta = eta / (1 + decay_rate * t)
      elif decay_type == 'step-based':
        eta = eta * decay_rate ** tf.floor(t/1000)
      elif decay_type == 'exponential':
        eta = eta *  tf.math.pow(decay_rate, t)
      elif decay_type == 'performance':
        if len(loss_val) > 1 and loss_val[-1] > loss_val[-2]:
          eta = eta / decay_rate

    return eta
