from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(X):
    return 1/(1+np.exp(-X))


def load_data():
    mndata = MNIST("data/")
    X, y = mndata.load_training()
    y = np.array(y, dtype="uint8")
    X = np.array([np.array(x) for x in X], dtype="uint8")
    N, D = X.shape
    Xtest, ytest = mndata.load_testing()
    ytest = np.array(ytest, dtype="uint8")
    Xtest = np.array([np.array(x) for x in Xtest], dtype="uint8")
    Ntest, Dtest= Xtest.shape

    return X, y, N, D, Xtest, ytest, Ntest, Dtest


def binary_log_loss(y, y_hat):
    return -y*np.log(y_hat) - (1 - y)*np.log(1 - y_hat)


def multi_log_loss(y, y_hat):
    N, K = y_hat.shape

    p1 = np.log(y_hat[range(N), y])
    print(p1)
    p2 = (1-y)*np.log(1-y_hat[range(N), y])
    print(p2)
    return -(np.sum(p1 + p2))/N

def reg_loss(lamb, W1, W2):
    return (lamb/2)*(np.sum(W1*W1) + np.sum(W2*W2))



