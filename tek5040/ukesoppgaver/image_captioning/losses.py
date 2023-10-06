import tensorflow as tf
from tensorflow.keras import activations, layers, losses, optimizers

loss_object = losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    # num_samples is the number of samples trained on
    num_samples = tf.reduce_sum(mask)

    return tf.reduce_sum(loss_), num_samples

