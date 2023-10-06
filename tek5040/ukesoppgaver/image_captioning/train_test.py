import tensorflow as tf
import time
from tensorflow.keras import activations, layers, losses, optimizers
from ed_model import CNN_Encoder, RNN_Decoder
from losses import loss_function

#######################################################################################################
## Training
#
#* You extract the features stored in the respective `.npy` files and then pass those features through the encoder.
#* The encoder output, hidden state(initialized to 0) and the decoder input (which is the start token) is passed to the decoder.
#* The decoder returns the predictions and the decoder hidden state.
#* The decoder hidden state is then passed back into the model and the predictions are used to calculate the loss.
#* Use teacher forcing to decide the next input to the decoder.
#* Teacher forcing is the technique where the target word is passed as the next input to the decoder.
#* The final step is to calculate the gradients and apply it to the optimizer and backpropagate.
#####################################################################################################



#########################################
# Train step
#########################################
def train_step(img_tensor, target, encoder,decoder,tokenizer,optimizer):
    loss = 0
    num_samples = 0

    batch_size = target.shape[0]
    hidden = decoder.get_initial_state(batch_size=batch_size, dtype="float32")
    state_out = hidden[0]

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * batch_size, 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            predictions, state_out, hidden, _ = decoder([dec_input, features, state_out, hidden])

            loss_t, num_samples_t = loss_function(target[:, i], predictions)
            loss += loss_t
            num_samples += num_samples_t

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

        # value loss of each samle in batch equally
        average_loss = loss / num_samples

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(average_loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, num_samples

#########################################
# Validation step
#########################################
def val_step(img_tensor, target, encoder,decoder,tokenizer,optimizer):
    """Similar to train loop, except that we don't calculate gradients and update
    variables.
    """
    loss = 0
    num_samples = 0

    batch_size = target.shape[0]
    hidden = decoder.get_initial_state(batch_size=batch_size, dtype="float32")
    state_out = hidden[0]

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * batch_size, 1)

    features = encoder(img_tensor)

    for i in range(1, target.shape[1]):
        # passing the features through the decoder
        predictions, state_out, hidden, _ = decoder([dec_input, features, state_out, hidden])

        loss_t, num_samples_t = loss_function(target[:, i], predictions)
        loss += loss_t
        num_samples += num_samples_t

        # using teacher forcing
        dec_input = tf.expand_dims(target[:, i], 1)

    return loss, num_samples
    
#########################################
# Return validation loss after executing val_step()
#########################################
def val_loss(dataset_val,encoder,decoder,tokenizer,optimizer):
    """Calculate validation loss for entire validation set."""

    start = time.time()
    total_loss = 0
    total_samples = 0

    for (batch, (img_tensor, target)) in enumerate(dataset_val):
        loss, samples = val_step(img_tensor, target, encoder,decoder,tokenizer,optimizer)
        total_loss += loss
        total_samples += samples

    # storing the epoch end loss value to plot later
    average_loss_epoch = total_loss / total_samples
    print('Validation time: {} sec\n'.format(time.time() - start))

    return average_loss_epoch