from layers import *


def affine_relu_forward(X, W, b):
    a, fc_cache = affine_forward(X, W, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)

    return out, cache

def affine_relu_backward(dout, cache):
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dX, dW, db = affine_backward(da, fc_cache)

    return dX, dW, db

