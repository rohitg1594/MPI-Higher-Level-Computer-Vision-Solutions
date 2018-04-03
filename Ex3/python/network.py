import numpy as np
from layers import *
from layers_utils import *

class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, output_size, std=1e-3, reg=0):
        self.params = {}
        self.reg = reg
        self.D = input_size
        self.M = hidden_size
        self.C = output_size
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)


    def loss(self, X, y=None):
        #print(self.params)
        N, D = X.shape
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        reg = self.reg

        # Forward pass
        hout, hcache = affine_relu_forward(X, W1, b1)
        scores, fcache = affine_forward(hout, W2, b2)

        if y is None:
            return scores

        # Loss
        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2)
        loss = data_loss + reg_loss

        # Backward pass
        dhout, dW2, db2 = affine_backward(dscores, fcache)
        dW2 += self.reg*W2
        dX, dW1, db1 = affine_relu_backward(dhout, hcache)
        dW1 += self.reg*W1

        grads = {}
        grads['W1'] = dW1
        grads['W2'] = dW2
        grads['b1'] = db1
        grads['b2'] = db2

        return loss, grads


    def train(self, optim_func, X, y, lr=0.01, epochs=50, print_every=10):
        for i in range(epochs):
            batch_mask = np.random.choice(X.shape[0], 100)
            X_batch = X[batch_mask]
            y_batch = y[batch_mask]
            curr_loss, grads = self.loss(X_batch, y_batch)
            self.params = optim_func(self.params, grads, lr=0.01)
            # print(self.params)
            if i % print_every == 0:
                print('Epoch : {} , Current loss : {}'.format(i, curr_loss))

        return self.params

