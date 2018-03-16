function [J grad] = nnGradient(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Setup some useful variables
N = size(X, 2);

W1 = nn_params.W1;
b1 = nn_params.b1;
W2 = nn_params.W2;
b2 = nn_params.b2;

% Feedforward is run here.

[J, activations] = nnLossRegFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda);

a1 = activations.a1;
z2 = activations.z2;
a2 = activations.a2;
z3 = activations.z3;
a3 = activations.a3;

% You need to return the following variables correctly 

W1_grad = zeros(size(W1)); 
b1_grad = zeros(size(b1));
W2_grad = zeros(size(W2));  
b2_grad = zeros(size(b2));

% ====================== YOUR CODE HERE ======================
% Instructions: Write the backpropagation algorithm for our network.
% Input: N. Number of training data
%        num_labels. Number of classes for our task (10 in our case)
%        y. Ground truth labels
%        a1. First layer activation. Identical to X
%        z2. Second layer activation before sigmoid activation.
%        a2. Second layer activation after sigmoid activation.
%        z3. Third layer activation before sigmoid activation.
%        a3. Third layer activation after sigmoid activation. Corresponds
%            to the output of the network.
%        W1, b1, W2, b2. Network parameters.
%        lambda. Regularisation hyperparameter
% Output: W1_grad. updated momentum vector
%         b1_grad. updated parameter
%         W2_grad. updated momentum vector
%         b2_grad. updated parameter

















% -------------------------------------------------------------
% =========================================================================

% Return gradients
grad.W1 = W1_grad;
grad.b1 = b1_grad;
grad.W2 = W2_grad;
grad.b2 = b2_grad;

end
