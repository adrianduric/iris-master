import tensorflow as tf
from tensorflow.keras import activations, layers, losses, optimizers

# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle
from tqdm import tqdm

from dataset import load_image


def plot_attention(image, result, attention_plot, feature_height, feature_width):
    temp_image = np.array(Image.open(image))

    fig = Figure(figsize=(10, 10))
    canvas = FigureCanvas(fig)

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (feature_height, feature_width))
        num_cols = int(np.ceil(np.sqrt(len_result)))
        num_rows = (len_result + num_cols - 1) // num_cols
        ax = fig.add_subplot(num_rows, num_cols, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    fig.tight_layout()
    canvas.draw()
    x = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    # width, height = fig.get_size_inches() * fig.get_aligni()
    width, height = fig.get_size_inches() * fig.get_dpi()
    width = int(np.round(width))
    height = int(np.round(height))
    x = np.reshape(x, [height, width, 3])

    return x

def evaluate(image, max_length, attention_features_shape, encoder, decoder,tokenizer,image_features_extract_model):
    max_vis = min(max_length, 9)
    attention_plot = np.zeros((max_vis, attention_features_shape))

    hidden = decoder.get_initial_state(batch_size=1, dtype="float32")
    state_out = hidden[0] # why not this returned...

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_vis):
        predictions, state_out, hidden, attention_weights = decoder([dec_input, features, state_out, hidden])

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot