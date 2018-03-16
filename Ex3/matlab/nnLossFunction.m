function [J, activations] = nnLossFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y)
%NNLOSSFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   J = NNLOSSFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y) computes the (unregularised) loss of the neural network.

% Setup some useful variables
N = size(X, 2);

% Compute feedForward probs
[probs, activations] = feedForward(nn_params, ...
                     input_layer_size, ...
                     hidden_layer_size, ...
                     num_labels, ...
                     X);

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Implement the log loss here.
% Input: N. Number of training data
%        num_labels. Number of classes for our task (10 in our case)
%        y. Ground truth labels
%        probs. Output of the network
% Output: J. Unregularised loss









% -------------------------------------------------------------
% =========================================================================


end
