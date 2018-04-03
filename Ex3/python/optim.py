import numpy as np

def sgd(w, dw, config=None):
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config
###############################################################################
# W1, W2, b1, b2 = params['W1'], params['W2'], params['b1'], params['b2']     #
#     dW1, dW2, db1, db2 = grads['W1'], grads['W2'], grads['b1'], grads['b2'] #
#                                                                             #
#     params['W1'] = W1 - lr*dW1                                              #
#     params['W2'] = W2 - lr*dW2                                              #
#     params['b1'] = b1 - lr*db1                                              #
#     params['b2'] = b2 - lr*db2                                              #
#                                                                             #
#     return params                                                           #
###############################################################################


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the momentum update formula. Store the updated value in #
    # the next_w variable. You should also use and update the velocity v.     #
    ###########################################################################
    m = config['momentum']
    alpha = config['learning_rate']
    v = m*v - alpha*dw
    next_w = w + v
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    config['velocity'] = v

    return next_w, config


def adam(x, dx, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 1)

    next_x = None
    ###########################################################################
    # TODO: Implement the Adam update formula, storing the next value of x in #
    # the next_x variable. Don't forget to update the m, v, and t variables   #
    # stored in config.                                                       #
    ###########################################################################
    learning_rate = config['learning_rate']
    beta1 = config['beta1']
    beta2 = config['beta2']
    epsilon = config['epsilon']
    m = config['m']
    v = config['v']
    t = config['t']
    
    t += 1
    m = beta1*m + (1-beta1)*dx
    mt = m/(1-beta1**t)
    v = beta2*v + (1-beta2)*(dx**2)
    vt = v/(1-beta2**t)
    next_x = x - learning_rate * mt / (np.sqrt(vt) + epsilon)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    config['m'] = m
    config['v'] = v
    config['t'] = t
    
    return next_x, config
