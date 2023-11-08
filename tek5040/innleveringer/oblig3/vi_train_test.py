from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks, optimizers
from common import gen_data
from common import neg_log_likelihood
from common import train_size
from common import noise
from densevariational import DenseVariational
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import tqdm
import matplotlib.pyplot as plt

tf_ver=tf.__version__
tfv=tf_ver.split('.')
print(int(tfv[1]))
if (int(tfv[0]) <2):
    print('Need tensorflow version 2')
    exit(0)

batch_size = train_size
num_batches = train_size / batch_size

var_weight = 1.0 / num_batches
prior_params = {
    'prior_sigma_1': 1.5,
    'prior_sigma_2': 0.1,
    'prior_pi': 0.5
}

x_in = Input(shape=(1,))
x = DenseVariational(20, var_weight, prior_params, activation='relu')(x_in)
x = DenseVariational(20, var_weight, prior_params, activation='relu')(x)
x = DenseVariational(1, var_weight, prior_params)(x)

model = Model(x_in, x)

if (int(tfv[1]) <11):
    model.compile(loss=neg_log_likelihood, optimizer=optimizers.Adam(lr=0.08), metrics=['mse'])
else:
    model.compile(loss=neg_log_likelihood, optimizer=optimizers.legacy.Adam(lr=0.08), metrics=['mse'])

X,y,_= gen_data(train_size,noise, False)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")


print('start fitting the model....')
model.fit(X, y, batch_size=batch_size, epochs=1500, verbose=1, callbacks=[tensorboard_callback]);



############################################################
# Testing
############################################################

X_test = np.linspace(-1.5, 1.5, 1000).reshape(-1, 1)
y_pred_list = []


for i in tqdm.tqdm(range(500)):
    y_pred = model(X_test, training=False) #model.predict(X_test)
    y_pred_list.append(y_pred.numpy())

y_preds = np.concatenate(y_pred_list, axis=1)

y_mean = np.mean(y_preds, axis=1)
y_sigma = np.std(y_preds, axis=1)


plt.plot(X_test, y_mean, 'r-', label='Predictive mean');
plt.scatter(X, y, marker='+', label='Training data')
plt.fill_between(X_test.ravel(),
                 y_mean + 2 * y_sigma,
                 y_mean - 2 * y_sigma,
                 alpha=0.3, label='Epistemic uncertainty',
                 color='yellow')
plt.title('Prediction')
plt.legend();
plt.show()
