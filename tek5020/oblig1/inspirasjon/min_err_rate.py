import numpy as np

def g(x, W, w, w0):
    """ Calcualates the quadratic discriminant function.

    Keyword arguments:
    x -- 1xd feature vector
    W -- dxd matrix
    w -- dx1 vector
    w0 -- 1x1 scalar

    Return value:
    the result of the quadratic discriminant function
    """
    return x @ W @ x.T + w.T @ x.T + w0

def min_err_rate(test_data, train_data, feature_idx):
    """
    Calculate the error rate for the best feature combination using the minimum
    error rate.

    Keyword arguments:
    test_date -- input test data
    train_data -- input train data
    feature_idx -- index for the best best feature combination

    Return value:
    err_rate -- classification error rate for the given feature combination

    Other:
    Every variable name ended with 1 or 2 refers to class 1 or 2 respectively.
    """
    test_objects = test_data[:, 1:]
    test_objects = test_objects[:, feature_idx]
    train_objects = train_data[:, 1:]
    train_objects = train_objects[:, feature_idx]

    idx1 = train_data[:, 0] == 1
    idx2 = ~idx1
    train1 = train_objects[idx1]
    train2 = train_objects[idx2]

    # number of objects
    n = train_data.shape[0]
    n1 = np.sum(idx1)
    n2 = n - n1

    # a priori probability for each class
    P_omega1 = n1/n
    P_omega2 = n2/n

    # maximum likelihood estimate of the expectation value for each class
    mu1 = 1/n1 * np.sum(train1, axis=0)
    mu2 = 1/n2 * np.sum(train2, axis=0)

    # maximum likelihood estimate of the covariance matrix for each class
    cov1 = 1/n2 * (train1 - mu1).T @ (train1 - mu1)
    cov2 = 1/n2 * (train2 - mu2).T @ (train2 - mu2)

    W1 = -1/2 * np.linalg.pinv(cov1)
    W2 = -1/2 * np.linalg.pinv(cov2)

    w1 = np.linalg.pinv(cov1) @ mu1.T
    w2 = np.linalg.pinv(cov2) @ mu2.T

    w10 = -1/2 * mu1 @ np.linalg.pinv(cov1) @ mu1.T \
          - 1/2 * np.log(np.linalg.det(cov1)) + np.log(P_omega1)
    w20 = -1/2 * mu2 @ np.linalg.pinv(cov2) @ mu2.T \
          - 1/2 * np.log(np.linalg.det(cov2)) + np.log(P_omega2)

    err_rate = 0
    for i in range(test_objects.shape[0]):
        g1 = g(test_objects[i], W1, w1, w10)
        g2 = g(test_objects[i], W2, w2, w20)
        g_ = g1 - g2
        if g_ > 0:
            class_error = test_data[i, 0] != 1
        else:
            class_error = test_data[i, 0] != 2
        err_rate += class_error
    err_rate /= test_objects.shape[0]

    return err_rate