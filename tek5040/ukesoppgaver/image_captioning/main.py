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

from ed_model import CNN_Encoder, RNN_Decoder
from utils import plot_attention,evaluate
from dataset import download_data,fea_extract_model,get_unique_images,create_train_validate_data,create_dataset
#from dataset import calc_max_length
from train_test import train_step, val_step, val_loss

######################################
#Definition of Hyper-parameters
######################################
embedding_dim = 256
units = 512

# Shape of the vector extracted from NASNetMobile is (49, 1056)
# These two variables represent that vector shape
feature_channels = 1056
feature_height = feature_width = 7
attention_features_shape = feature_height * feature_width
spatial_positions = feature_height * feature_width

attention_type = "uniform"
#attention_type = "dotproduct"
#attention_type = "bahdanau"
BATCH_SIZE_ = 16

######################################
# Download and save data
######################################
trval_cap,img_nm_vec =download_data()

######################################
# Define the feature extraction model
# Function is in dataset.py
######################################
fe_model=fea_extract_model()

######################################
# 
######################################
get_unique_images(fe_model,img_nm_vec)

######################################
# Create train and validation datasets
# Uses tf.data.Dataset API
# Functions are in dataset.py
######################################
im_trn, img_name_val, cap_trn, cap_val,vocab_size, trn_size,tokenizer,max_length=create_train_validate_data(trval_cap,img_nm_vec)

dataset_train = create_dataset(im_trn, cap_trn, shuffle=True,BATCH_SIZE=BATCH_SIZE_)
dataset_val = create_dataset(img_name_val, cap_val, shuffle=False,BATCH_SIZE=BATCH_SIZE_)

num_steps =trn_size // BATCH_SIZE_

######################################
# Definition of Encoder and Decoder models
# Functions are in ed_model.py
######################################
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size,
                      attention_type=attention_type)

######################################
# Define the optimizer
######################################
# Note that the learning rate has not been optimized. You may also want to
# implement a decreasing learning rate schedule for optimal performance.
optimizer = optimizers.Adam(learning_rate=0.001)





######################################
# Handle checkpoints
######################################
train_dir = "train_dir/%s" % attention_type
checkpoint_path = train_dir + "/checkpoints"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
if ckpt_manager.latest_checkpoint:
    print("Restored weights from {}".format(ckpt_manager.latest_checkpoint))
    ckpt.restore(ckpt_manager.latest_checkpoint)
else:
    print("Initializing random weights.")

start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])


######################################
## Training
#
#* You extract the features stored in the respective `.npy` files and then pass those features through the encoder.
#* The encoder output, hidden state(initialized to 0) and the decoder input (which is the start token) is passed to the decoder.
#* The decoder returns the predictions and the decoder hidden state.
#* The decoder hidden state is then passed back into the model and the predictions are used to calculate the loss.
#* Use teacher forcing to decide the next input to the decoder.
#* Teacher forcing is the technique where the target word is passed as the next input to the decoder.
#* The final step is to calculate the gradients and apply it to the optimizer and backpropagate.
######################################
train_writer = tf.summary.create_file_writer(train_dir + "/train", flush_millis=3000)
val_writer = tf.summary.create_file_writer(train_dir + "/val", flush_millis=3000)

EPOCHS = 10
summary_interval = 10
step = num_steps * start_epoch
num_summary_images = 5
checkpoint_every_n_epochs = 1
for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0
    total_samples = 0

    # Create summaries (To explore in Tensorboard)
    # evaluate() is defined in utils.py    
    summary_images = img_name_val[:num_summary_images]
    for idx, image in enumerate(summary_images):
        result, attention_plot = evaluate(image,max_length, attention_features_shape, encoder, decoder,tokenizer,fe_model)
        x = plot_attention(image, result, attention_plot,feature_height,feature_width)
        x = tf.expand_dims(tf.convert_to_tensor(x), 0)
        with val_writer.as_default():
            tf.summary.image("image_%d" % idx, x, step=step)

    #Training loop (train_step() is defined in train_test.py)
    for (batch, (img_tensor, target)) in enumerate(dataset_train):
        loss, samples = train_step(img_tensor, target,encoder,decoder,tokenizer,optimizer)
        total_loss += loss
        total_samples += samples
        step += 1
        if batch % 100 == 0:
            # NOTE: this loss will have high variance
            print ('Epoch {} Batch {} Loss {:.4f}'.format(
              epoch + 1, batch, loss.numpy() / samples.numpy()))
            with train_writer.as_default():
              tf.summary.scalar("loss", total_loss/total_samples, step=step)

    # storing the epoch end loss value to plot later
    average_loss_epoch = total_loss / total_samples

    # do validattion (val_loss() is defined in train_test.py)
    val_l = val_loss(dataset_val,encoder,decoder,tokenizer,optimizer)
    with val_writer.as_default():
      tf.summary.scalar("loss", val_l, step=step)
    
    #checkpointing 
    if (epoch+1) % checkpoint_every_n_epochs == 0:
      print("Checkpointing model after %d epochs of training." % (epoch+1))
      ckpt_manager.save(epoch+1)

    print('Epoch {} Loss {:.6f}'.format(epoch + 1, average_loss_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


######################################
## Try it on your own images
# System outputs the caption as well as a visualization of the attention on imsges
# Keep in mind, it was trained on a relatively small amount of data, and your images may be different from the training data (so be prepared for weird results!)
######################################
image_url = 'https://tensorflow.org/images/surf.jpg'
image_extension = image_url[-4:]
image_path = tf.keras.utils.get_file('image'+image_extension,
                                     origin=image_url)

result, attention_plot = evaluate(image_path, max_length, attention_features_shape, encoder, decoder,tokenizer,fe_model)
print ('Prediction Caption:', ' '.join(result))
x = plot_attention(image_path, result, attention_plot,feature_height,feature_width)
plt.imshow(x)
plt.show()


######################################
#
## Save models
#You can also restore model from checkpoints, but then you have to first build your model with the code from this script and use a checkpointmanager to load the weights. 
#An often more convenient method is to use the `model.save` method, to save both the model and the weights. 
#We need to call `model._set_inputs` when we haven't used the `model.predict` or `model.fit` functions.
#
######################################
encoder._set_inputs(tf.keras.Input([spatial_positions, feature_channels]))
encoder.save(os.path.join(train_dir, "encoder.hd5"))
#decoder._set_inputs(
#  [
#    tf.keras.Input([vocab_size]), # predicted word
#    tf.keras.Input([spatial_positions, embedding_dim]), # embedded spatial features
#    tf.keras.Input([units]), # output LSTM
#    [tf.keras.Input([units]), tf.keras.Input([units])] # hidden LSTM state
#  ]
#)
#decoder.save(os.path.join(train_dir, "decoder.hd5"))
# Currently having some issues with setting inputs like this. Thus we save
# weights only for now, which makes it a bit more complicated to load the
# model.
decoder.save_weights(os.path.join(train_dir, "decoder.hd5"))
# In e.g. a different script you may now load the models in this way
# encoder = tf.keras.models.load_model("/path/to/encoder.hd5")
# decoder = RNNDecoder(same params used for saved model)
# decoder.load_weights("/path/to/decoder.hd5")

