import sys

from network import *
from utils import sigmoid, binary_log_loss, multi_log_loss, load_data, reg_loss
from visualize import showdigits
from layers import *
from layers_utils import *
import optim


if __name__ == '__main__':
    test_func = sys.argv[1]
    if test_func == 'bl':
        y_hat = np.linspace(0,1,1000)
        loss_0 = binary_log_loss(0, y_hat)
        loss_1 = binary_log_loss(1, y_hat)
        plt.subplot(1,2,1)
        plt.plot(y_hat, loss_0)
        plt.subplot(1,2,2)
        plt.plot(y_hat, loss_1)
        plt.show()

    if test_func == 'load':
        X, y, N, D, Xtest, ytest, Ntest = load_data()
        print(y)

    if test_func == 'ml':
        X, y, N, D, Xtest, ytest, Ntest = load_data()
        params = initialize_params()
        y_hat, cache = forward(X, params)
        print(multi_log_loss(y, y_hat))
        print(reg_loss(1, params[0], params[2]))

    if test_func == 'showdigits':
        X, y, N, d, X_test, y_test, N_test = load_data()
        print(X.shape)
        print(len(y))
        print(N)
        print(d)
        showdigits(X, y)
        plt.show()

    if test_func == 'train':
        #input_size = sys.argv[2]
        #hidden_size = sys.argv[3]
        #output_size = sys.argv[4]

        X, y, N, D, Xtest, ytest, Ntest, Dtest = load_data()
        net = TwoLayerNet(784, 25, 10, 1e-3, 1)
        params = net.params
        print('X :{}, y : {}, N : {}, D : {}, Ntest : {}, Dtest : {}'.format(X.shape, len(y), N, D, Ntest, Dtest))
        print('W1 :{}, W2 : {}, b1 : {}, b2 : {}'.format(params['W1'].shape, params['W2'].shape, params['b1'].shape, params['b2'].shape))
        final_params = net.train(optim.vanilla_gd, X, y, lr =0.01, epochs=5000)

        W1, W2, b1, b2 = final_params['W1'], final_params['W2'], final_params['b1'], final_params['b2']
        # print('W1 :{}, W2 : {}, b1 : {}, b2 : {}'.format(W1.shape, W2.shape, b1.shape, b2.shape))        
        hidden_train, _ = affine_relu_forward(X, W1, b1)

        scores_train, _ = affine_forward(hidden_train, W2, b2)
        predictions_train = np.argmax(scores_train, axis=1)
        print('Train accuracy : {}'.format(np.sum(predictions_train==ytest) / N))

        hidden_test, _ = affine_relu_forward(Xtest, W1, b1)
        scores_test, _ = affine_forward(hidden_test, W2, b2)
        predictions_test = np.argmax(scores_test, axis=1)
        print('Test accuracy : {}'.format(np.sum(predictions_test==ytest) / Ntest))

    
