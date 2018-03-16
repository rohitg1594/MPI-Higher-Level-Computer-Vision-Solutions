function [probs, activations] = feedForward(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X)
%FEEDFORWARD Implements the neural network cost function for a two layer
%neural network which performs classification
%   [scores, activations] = FEEDFORWARD(nn_params, hidden_layer_size, num_labels, ...
%   X) computes the score of the neural network. 
% 

% neural net parameters
W1 = nn_params.W1;
b1 = nn_params.b1;
W2 = nn_params.W2;
b2 = nn_params.b2;

% Setup some useful variables
N = size(X, 2);

% First layer activation is identical to the input
a1 = X;

% You need to return the following variables correctly 
probs = zeros(10, N);

% ====================== YOUR CODE HERE ======================
% Instructions: Write the feedForward function of the network.
% Input: N. number of data
%        X. matrix of data. X(:,i) indicates the i^th data of dimension 400.
%        W1, b1, W2, b2. Network parameters.
% Output: z2. Second layer activation before sigmoid activation.
%         a2. Second layer activation after sigmoid activation.
%         z3. Third layer activation before sigmoid activation.
%         a3. Third layer activation after sigmoid activation. Corresponds
%             to the output of the network.




% -------------------------------------------------------------
% =========================================================================

% return activation values
activations.a1 = a1;
activations.z2 = z2;
activations.a2 = a2;
activations.z3 = z3;
activations.a3 = a3;

% Last layer activation is identical to the model output
probs = a3;
activations.probs = probs;


end
