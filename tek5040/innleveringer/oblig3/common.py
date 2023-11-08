import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import tensorflow as tf


def f(x, sigma):
    epsilon = np.random.randn(*x.shape) * sigma
    return 10 * np.sin(2 * np.pi * (x)) + epsilon



def gen_data(train_size=32,noise = 1.0, show=False):
    X = np.linspace(-0.5, 0.5, train_size).reshape(-1, 1)
    y = f(X, sigma=noise)
    y_true = f(X, sigma=0.0)

    if (show):
        plt.scatter(X, y, marker='+', label='Training data')
        plt.plot(X, y_true, label='Truth')
        plt.title('Noisy training data and ground truth')
        plt.legend();
        plt.show()
    return X,y, y_true


train_size=32
noise =1.0

def neg_log_likelihood(y_obs, y_pred, sigma=noise):
    dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
    return tf.math.reduce_sum(-dist.log_prob(y_obs))


if __name__ == "__main__":
    gen_data(32,1.0, True)