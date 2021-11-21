"""
    TODO build a schedule that works even if it get stuck early on
"""
import tensorflow as tf


def scheduler1(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)