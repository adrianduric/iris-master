from tensorflow.keras.layers import Dropout, Dense,Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from common import gen_data
from common import train_size
from common import noise
from common import neg_log_likelihood
import tensorflow as tf


def normal_dist(mu_and_sigma):
  return tfd.Normal(loc=mu_and_sigma[:,0:1], scale=tf.math.exp(mu_and_sigma[:,1:2]))


x_in = Input(shape=(1,))
x = Dense(20, activation='relu')(x_in)
x = Dropout(0.1)(x, training=True)
x = Dense(20, activation='relu')(x)
x = Dropout(0.1)(x, training=True)
mu_and_sigma = Dense(1)(x)



mc_model = Model(inputs=x_in, outputs=mu_and_sigma)

mc_model.compile(Adam(learning_rate=0.08), loss=neg_log_likelihood)

X,y,_= gen_data(train_size,noise, False)

mc_model.fit(X, y, epochs=5000, verbose=1,batch_size=512)

#######################################################
# testing
##################################################

X_test = np.linspace(-1.5, 1.5, 1000).reshape(-1, 1)
y_pred_list = []

for i in tqdm.tqdm(range(500)):
   y_pred= mc_model(X_test, training=True)
   #print(y_pred.shape)
   y_pred_list.append(y_pred)

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






