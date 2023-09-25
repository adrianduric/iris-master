import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from min_err_rate import min_err_rate
from least_squares import least_squares
from nearest_neighbor import nearest_neighbor


def train_test_split(data):
    """Split data in train and test.

    The train dataset contains the odd numbered data and the test dataset
    contains the even numbered data.

    Keyword arguments:
    data -- input dataset

    Return value:
    train -- train data containing odd numbered data
    test -- test data containing even numbered data
    """
    # create boolean index array with True for every even index and False for
    # every odd index
    index = np.linspace(1, len(data), len(data))
    boolean = index % 2 == 0

    # split data in train and test
    train = data[np.invert(boolean)]
    test = data[boolean]

    return train, test

def readfile(filename):
    """Convert file content to numpy array using pandas.

    Keyword arguments:
    filename -- name of file to read

    Return value:
    data -- file content as numpy array
    """
    data = pd.read_csv(filename, header=None, sep='\s+', engine='python')
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    data = data.to_numpy()
    return data

def best_combination(err_rate, idx_array):
    """Finds the feature combination with the smalles error rate.

    Keyword arguments:
    err_rate -- array containing error rate for each feature combination
    idx_array -- list containing all feature combinations

    Return value:
    err_rate[best_idx] -- the smallest error rate
    best_comb -- the feature combination corresponding to the lowest error rate
    """
    best_idx = np.argmin(err_rate)
    best_comb = idx_array[best_idx]
    return err_rate[best_idx], best_comb

# convert txt file to numpy array
data = readfile('../Datafiles/ds-2.txt')

# Plot the feature space for datasett 2
fig = plt.figure()
ax = fig.gca(projection='3d')
class1 = data[data[:, 0] == 1]
class2 = data[data[:, 0] == 2]
surf = ax.scatter(class1[:, 1], class1[:, 2], class1[:, 3], label='Klasse 1')
surf = ax.scatter(class2[:, 1], class2[:, 2], class2[:, 3], label='Klasse 2')
ax.set_xlabel('Egenskap 1', size=12)
ax.set_ylabel('Egenskap 2', size=12)
ax.set_zlabel('Egenskap 3', size=12)
ax.set_title('Egenskapsrommet for datasett 2', size=16)
plt.legend(loc='best')
plt.show()

train, test = train_test_split(data)

if train.shape[1] == 5:
    err_rate_d1, err_rate_d2, err_rate_d3, err_rate_d4, \
        idx_d1, idx_d2, idx_d3, idx_d4 = nearest_neighbor(test, train)

    error_rate = [err_rate_d1, err_rate_d2, err_rate_d3, err_rate_d4]
    index = [idx_d1, idx_d2, idx_d3, idx_d4]
else:
    err_rate_d1, err_rate_d2, err_rate_d3, \
        idx_d1, idx_d2, idx_d3 = nearest_neighbor(test, train)

    error_rate = [err_rate_d1, err_rate_d2, err_rate_d3]
    index = [idx_d1, idx_d2, idx_d3]

for err_rate, idx in zip(error_rate, index):
    NN_err_rate, best_comb = best_combination(err_rate, idx)
    minimum_err_rate = min_err_rate(test, train, best_comb)
    ls_err_rate = least_squares(test, train, best_comb)

    print('-------------------------------------------')
    print(f'Best combination: \t {best_comb}')
    print(f'Nearest neighbor: \t {NN_err_rate:.3}')
    print(f'Minimum error rate: \t {minimum_err_rate:.3}')
    print(f'Least squares: \t \t {ls_err_rate:.3}')
    print('-------------------------------------------\n')

print(err_rate_d1)
print(err_rate_d2)
print(err_rate_d3)
print(err_rate_d4)