import numpy as np

def affine_forward(X, W, b):
    out = X@W + b
    cache = (X, W, b)

    return out, cache


def affine_backward(dout, cache):
    X, W, b = cache

    dW = np.reshape(X, (X.shape[0], W.shape[0])).T@dout
    dX = np.dot(dout,W.T).reshape(X.shape)
    db = np.sum(dout,axis = 0, keepdims=True)

    return dX, dW, db


def relu_forward(X):
    cache = X
    out = np.maximum(0, X)

    return out, cache


def relu_backward(dout, cache):
    dX, X = None, cache
    dX = dout
    dX[X <= 0] = 0

    return dX


def softmax_loss(x, y):
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
######################################################################
# N, D = scores.shape                                                #
#     scores -= np.max(scores, axis=1, keepdims = True)              #
#     exp_scores = np.exp(scores)                                    #
#     probs = exp_scores/np.sum(exp_scores,axis = 1,keepdims = True) #
#     correct_logprobs = -np.log(probs[range(N),y])                  #
#     loss = np.sum(correct_logprobs) / N                            #
#                                                                    #
#     dscores = probs.copy()                                         #
#     dscores[np.arange(N), y] -= 1                                  #
#     dscores /= N                                                   #
#                                                                    #
#     return loss, dscores                                           #
######################################################################



