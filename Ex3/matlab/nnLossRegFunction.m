function [J, activations] = nnLossRegFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNLOSSREGFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J, activations] = NNLOSSREGFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

W1 = nn_params.W1;
W2 = nn_params.W2;

% You need to return the following variables correctly 
J = 0;

% Return loss value without regularisation
[J, activations] = nnLossFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y);

% ====================== YOUR CODE HERE ======================
% Instructions: Implement the L2 regularisation term
%        W1, W2. Network parameters.
%        lambda. Regularisation hyperparameter
% Output: R. Regularisation term




% -------------------------------------------------------------
% =========================================================================

% Combine regularisation term
J = J + R;


end
