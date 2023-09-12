import tensorflow as tf
from keras import layers, models



def simple_model(input_shape):

    height, width, channels = input_shape
    image = layers.Input(input_shape)
    x = layers.Conv2D(32, 5, strides=(2, 2), padding='same', activation='relu')(image)
    x = layers.Conv2D(64, 5, strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2D(1, 1, padding='same', activation=None)(x)
    # resize back into same size as regularization mask
    x = tf.image.resize(x, [height, width])
    x = tf.keras.activations.sigmoid(x)

    model = models.Model(inputs=image, outputs=x)

    return model


def conv2d_3x3(filters):
    conv = layers.Conv2D(
        filters, kernel_size=(3, 3), activation='relu', padding='same'
    )
    return conv

def conv2d_transpose_2x2(filters):
    conv_transpose = layers.Conv2DTranspose(
        filters, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same'
    )
    return conv_transpose

def max_pool():
    return layers.MaxPooling2D((2, 2), strides=2, padding='same')


def unet(input_shape):

    image = layers.Input(shape=input_shape)

    c1 = conv2d_3x3(8)(image)
    c1 = conv2d_3x3(8)(c1)
    p1 = max_pool()(c1)

    c2 = conv2d_3x3(16)(p1)
    c2 = conv2d_3x3(16)(c2)
    p2 = max_pool()(c2)

    c3 = conv2d_3x3(32)(p2)
    c3 = conv2d_3x3(32)(c3)
    p3 = max_pool()(c3)

    c4 = conv2d_3x3(64)(p3)
    c4 = conv2d_3x3(64)(c4)
    p4 = max_pool()(c4)

    c5 = conv2d_3x3(128)(p4)
    c5 = conv2d_3x3(128)(c5)
    u5 = conv2d_transpose_2x2(128)(c5)

    c6 = conv2d_3x3(64)(u5)
    c6 = conv2d_3x3(64)(c6)
    u6 = conv2d_transpose_2x2(64)(c6)

    c7 = conv2d_3x3(32)(u6)
    c7 = conv2d_3x3(32)(c7)
    u7 = conv2d_transpose_2x2(32)(c7)

    c8 = conv2d_3x3(16)(u7)
    c8 = conv2d_3x3(16)(c8)
    u8 = conv2d_transpose_2x2(16)(c8)

    c9 = conv2d_3x3(8)(u8)
    c9 = conv2d_3x3(8)(c9) 
    
    probs = layers.Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(c9)
    
    model = models.Model(inputs=image, outputs=probs)

    return model
