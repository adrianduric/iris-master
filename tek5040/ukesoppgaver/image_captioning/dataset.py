
from __future__ import absolute_import, division, print_function, unicode_literals

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

'''## Download and prepare the MS-COCO dataset
You will use the [MS-COCO dataset](http://cocodataset.org/#home) to train our model. The dataset contains over 82,000 images, each of which has at least 5 different caption annotations. Due to computational considerations we will only consider the 71,973 captions that have 8 or fewer words, and the corresponding 48,659 corresponding images. Indeed we limit this further to the 20,000 images with the most captions, a total of 43,314 captions. The code below downloads and extracts the dataset automatically.
    
    **Caution: large download ahead**. The 20,000 images, is about ~3GB file.
    '''

def download_data():

    annotation_zip = tf.keras.utils.get_file(
        'captions.zip',
        cache_subdir=os.path.abspath('.'),
        #origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
        origin="https://www.uio.no/studier/emner/matnat/its/TEK5040/h19/data/captions_trainval2014_8_20000.zip?vrtxPreviewUnpublished",
        extract=True
    )
    annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'

    name_of_zip = 'train2014.zip'
    if not os.path.exists(os.path.abspath('.') + '/' + name_of_zip):
        image_zip = tf.keras.utils.get_file(
            name_of_zip,
            cache_subdir=os.path.abspath('.'),
            #origin='http://images.cocodataset.org/zips/train2014.zip',
            #origin="https://www.uio.no/studier/emner/matnat/its/TEK5040/h19/data/train2014_8_20000.zip?vrtxPreviewUnpublished",
            origin="https://www.uio.no/studier/emner/matnat/its/TEK5040/h21/data/train2014_8_20000.zip",
            extract=True)
        PATH = os.path.dirname(image_zip)+'/train2014/'
    else:
        PATH = os.path.abspath('.')+'/train2014/'

    '''## Optional: limit the size of the training set
    In the beginning you probably want to use only a small subset of the captions. In this way you don't have to wait too long to create the cached features. When you are happy with your code you could scale up to a larger subset, or even all the captions.
    
    *NOTE*: When changing NUM_EXAMPLES you may have to delete/rename old checkpoints as they may not be compatible due to change in vocabulary size, and thus our output layer.
    '''

    # Read the json file
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Store captions and image names in vectors
    all_captions = []
    all_img_name_vector = []

    # We add special words <start> and <end> to indicate 'start' and 'end' of sentence
    for annot in annotations['annotations']:
        caption = '<start> ' + annot['caption'] + ' <end>'
        image_id = annot['image_id']
        full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

        all_img_name_vector.append(full_coco_image_path)
        all_captions.append(caption)

    # Shuffle captions and image_names together
    # Set a random state
    captions, img_name_vector = shuffle(all_captions,
                                        all_img_name_vector,
                                        random_state=1)

    # limit to the first NUM_EXAMPLES from the shuffled set
    NUM_EXAMPLES = 500
    trainval_captions = captions[:NUM_EXAMPLES]
    img_name_vector = img_name_vector[:NUM_EXAMPLES]

    print("Number of total captions: %d" % len(all_captions))
    print("Number of captions used for training and validation: %d" % len(trainval_captions))

    return trainval_captions, img_name_vector

'''## Preprocess the images using NASNetMobile
Next, you will use [NASNetMobile](https://keras.io/applications/#nasnet) (which is pretrained on Imagenet) to classify each image. You will extract features from the last convolutional layer.

First, you will convert the images into NASNetMobile expected format by:
* Resizing the image to 224px by 224px
* Preprocess the images by normalize the image so that it contains pixels in the range of -1 to 1, which matches the format of the images used to train NASNet. 
'''
image_height = 224
image_width = 224

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (image_height, image_width))
    img = tf.keras.applications.nasnet.preprocess_input(img)
    return img, image_path

'''## Initialize NASNetMobile and load the pretrained Imagenet weights

Now you'll create a tf.keras model where the output layer is the last convolutional layer in the NASNetMobile architecture. The shape of the output of this layer is ```7x7x1056```. You use the last convolutional layer because you are using attention in this example. You don't perform this initialization during training because it could become a bottleneck.

* You forward each image through the network and store the resulting vector in a dictionary (image_name --> feature_vector).
* After all the images are passed through the network, you pickle the dictionary and save it to disk.
'''

def fea_extract_model():
    image_model = tf.keras.applications.NASNetMobile(include_top=False,
                                                     weights='imagenet',
                                                     input_shape=(224,224,3))
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    return image_features_extract_model

'''## Caching the features extracted from NASNetMobile

You will pre-process each image with NASNetMobile and cache the output to disk. Caching the output in RAM would be faster but also memory intensive, requiring 7 \* 7 \* 1056 floats per image.

Performance could be improved with a more sophisticated caching strategy (for example, by sharding the images to reduce random access disk I/O), but that would require more code.

The caching will take a few minutes to run in Colab with a GPU, but may take several hours on a laptop without GPU.

'''

# whether to force recompute of features even if exist in cache, useful if e.g.
# changing model to compute features from
FORCE_FEATURE_COMPUTE = False

def get_unique_images(image_features_extract_model,img_name_vector):
# Get unique images
    encode_train = sorted(set(img_name_vector))
    if not FORCE_FEATURE_COMPUTE:
        encode_train = [p for p in encode_train if not os.path.exists(p+'.npy')]

    if len(encode_train) > 0:
        # Feel free to change batch_size according to your system configuration
        image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
        image_dataset = image_dataset.map(load_image, num_parallel_calls=1).batch(1)

        print("Caching features for %d images." % len(encode_train))
        for img, path in tqdm(image_dataset):
            batch_features = image_features_extract_model(img)
            # collapse height and width dimension
            # (batch_size, 7, 7, 1056) --> (batch_size, 49, 1056)
            batch_features = tf.reshape(batch_features,
                                        (batch_features.shape[0], -1, batch_features.shape[3]))

            for bf, p in zip(batch_features, path):
                path_of_feature = p.numpy().decode("utf-8")
                np.save(path_of_feature, bf.numpy())


'''## Preprocess and tokenize the captions

* First, you'll tokenize the captions (for example, by splitting on spaces). This gives us a  vocabulary of all of the unique words in the data (for example, "surfing", "football", and so on).
* Next, you'll limit the vocabulary size to the top 5,000 words (to save memory). You'll replace all other words with the token "UNK" (unknown).
* You then create word-to-index and index-to-word mappings.
* Finally, you pad all sequences to be the same length as the longest one.
'''

# Find the maximum length of any caption in our dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)


def create_train_validate_data(trainval_captions, img_name_vector):
    # Choose the top 5000 words from the vocabulary
    top_k = 5000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

    tokenizer.fit_on_texts(trainval_captions)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    # Create the tokenized vectors, e.g. hypothetically
    # [<start> A man walking his dog <end>] --> [4, 4201, 13, 403, 35, 5, 321]
    trainval_seqs = tokenizer.texts_to_sequences(trainval_captions)

    # Pad each vector to the max_length of the captions
    # If you do not provide a max_length value, pad_sequences calculates it automatically
    # e.g. if maxlen was 10 we would get
    # [4, 4201, 13, 403, 35, 5, 321] --> [4, 4201, 13, 403, 35, 5, 321, 0, 0, 0]
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(trainval_seqs, padding='post')

    # Calculates the max_length, which is used to store the attention weights
    max_length = calc_max_length(trainval_seqs)
    print("Max length of captions in trainval: %d" % max_length)

    '''## Split the data into training and testing'''

    # Create training and validation sets using an 80-20 split
    img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                        cap_vector,
                                                                        test_size=0.2,
                                                                        random_state=0)

    print("Number of training captions: %d" % len(cap_train))
    print("Number of validation captions: %d" % len(cap_val))

    vocab_size = len(tokenizer.word_index) + 1
    train_size=len(img_name_train)

    return img_name_train, img_name_val, cap_train, cap_val, vocab_size, train_size,tokenizer,max_length

'''## Create a tf.data dataset for training

Our images and captions are ready! Next, let's create a tf.data dataset to use for training our model.
'''

# Feel free to change these parameters according to your system's configuration

#BATCH_SIZE = 16
#BUFFER_SIZE = 100
#embedding_dim = 256
#units = 512
'''/////////////////
vocab_size = len(tokenizer.word_index) + 1
num_steps = len(img_name_train) // BATCH_SIZE
///////////'''
# Shape of the vector extracted from NASNetMobile is (49, 1056)
# These two variables represent that vector shape
#feature_channels = 1056
#feature_height = feature_width = 7
#attention_features_shape = feature_height * feature_width
#spatial_positions = feature_height * feature_width

# Load the numpy files
def map_func(img_name, cap):
  img_tensor = np.load(img_name.decode('utf-8')+'.npy')
  return img_tensor, cap

def create_dataset(img_name_in, cap_in, shuffle=False, BATCH_SIZE = 16, BUFFER_SIZE = 100):
    dataset = tf.data.Dataset.from_tensor_slices((img_name_in, cap_in))

    # Use map to load the numpy files in parallel
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
        map_func, [item1, item2], [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle and batch
    if shuffle:
        dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

'''
trval_cap,img_nm_vec =download_data()
fe_model=fea_extract_model()
get_unique_images(fe_model,img_nm_vec)
im_trn, im_val, cap_trn, cap_val,voc_size, trn_size,t=create_train_validate_data(trval_cap,img_nm_vec)

dataset_train = create_dataset(im_trn, cap_trn, shuffle=True)
dataset_val = create_dataset(im_val, cap_val, shuffle=False)
'''