''''## Model

The model architecture is inspired by the [Show, Attend and Tell](https://arxiv.org/pdf/1502.03044.pdf) paper.

* In this example, you extract the features from the lower convolutional layer of NASNetMobile giving us a vector of shape (7, 7, 1056).
* You squash that to a shape of (49, 1056).
* This vector is then passed through the CNN Encoder (which consists of a single Fully connected layer).
* The RNN (here LSTM) attends over the image to predict the next word.
'''

import tensorflow as tf
from tensorflow.keras import activations, layers, losses, optimizers
import numpy as np

###################################
# Uniform Attention
###################################
class UniformAttention(layers.Layer):

  def call(self, feature_vectors, state_output):
      """Note: We do not use state_output."""

      batch_size = tf.shape(feature_vectors)[0]
      num_feature_vectors = tf.shape(feature_vectors)[1]
      attention_weights = tf.ones((batch_size, num_feature_vectors)) / np.float32(num_feature_vectors)

      # ==> [batch_size, feature_units]
      context_vector = tf.reduce_sum(tf.expand_dims(attention_weights, axis=-1)* feature_vectors, axis=1)

      return context_vector, attention_weights
      
###################################
# DotProduct Attention
###################################
class DotProductAttention(layers.Layer):    
    #implement this
    pass
    
###################################
# Bahdanau Attention
###################################
class BahdanauAttention(layers.Layer):
    #implemet this
    pass

###################################
# Encoder
###################################
class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = activations.relu(x)
        return x

###################################
# Decoder
###################################
class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size, attention_type):
    super(RNN_Decoder, self).__init__()
    self.units = units

    self.embedding = layers.Embedding(vocab_size, embedding_dim)
    self.lstm = layers.LSTMCell(self.units,
                                recurrent_initializer='glorot_uniform')
    self.fc1 = layers.Dense(self.units)
    self.fc2 = layers.Dense(vocab_size)

    self.attention_type = attention_type.lower()
    if self.attention_type == "uniform":
      self.attention = UniformAttention()
    elif self.attention_type == "dotproduct":
        raise NotImplementedError("TODO")
    elif self.attention_type == "bahdanau":
        raise NotImplementedError("TODO")
    else:
      raise ValueError(
        "attention_type '%s' not recognized. Expected oneof %s" %
        (self.attention_type, ["uniform", "dotproduct", "bahdanau"])
      )

    self.get_initial_state = self.lstm.get_initial_state

  def call(self, inputs):

    y, features, state_output, hidden = inputs

    # defining attention as a separate model
    context_vector, attention_weights = self.attention(features, state_output)

    # y shape after passing through embedding == (batch_size, 1, embedding_dim)
    y = self.embedding(y)

    # x shape after concatenation == (batch_size, features_dim + embedding_dim)
    x = tf.concat([context_vector, tf.squeeze(y, axis=1)], axis=-1)

    # passing the concatenated vector to the LSTM cell
    state_output, state = self.lstm(x, hidden)

    # shape == (batch_size, units)
    x = self.fc1(state_output)

    # output shape == (batch_size, vocab_size)
    x = self.fc2(x)

    return x, state_output, state, attention_weights

