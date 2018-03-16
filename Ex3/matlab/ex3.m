%% Training neural network from scratch

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions 
%  in this exericse:
%
%     feedForward.m
%     nnGradient.m
%     nnLossFunction.m
%     nnLossRegFunction.m
%     computeNumericalGradient.m
%     gradientStep.m
%     stochasticGradientDescent.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part A: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

ld = load('ex3data1.mat');
X = ld.X;
y = ld.y;
N = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

% In our exercise, we put data vectors in columns. e.g. 2nd data vector is
% X(:, 2).
X = X';
y = y';


%% ================ Part B: Loading Pameters ================
% In this part of the exercise, we load some pre-initialized 
% neural network parameters.

fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2
ld = load('ex3weights.mat');
Theta1 = ld.Theta1;
Theta2 = ld.Theta2;

% Putting network parameters into W1, b1, W2, b2.
nn_params.W1 = Theta1(:,2:end);
nn_params.b1 = Theta1(:,1);
nn_params.W2 = Theta2(:,2:end);
nn_params.b2 = Theta2(:,1);


%% ================ Part C: Feedforward ================
%  To the neural network, you should first start by implementing the
%  feedforward part of the neural network only. You
%  should complete the code in feedForward.m to return scores for each data for each label. After
%  implementing the feedforward, you can verify you get the same prediction
%  accuracy as us.

fprintf('\nFeedforward Using Neural Network ...\n')

probs = feedForward(nn_params, ...
                     input_layer_size, ...
                     hidden_layer_size, ...
                     num_labels, ...
                     X);

fprintf('\nComputing prediction accuracy ...\n');
[dummy, pred] = max(probs, [], 1);

fprintf(['\nTraining Set Accuracy: %2.2f'...
         '\n(this value should be 97.52)\n'], mean(double(pred == y)) * 100);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ================ Part D: Compute Loss (Unregularised) ================
%  To the neural network, you should first start by implementing the
%  feedforward part of the neural network that returns the loss only. You
%  should complete the code in nnLossFunction.m to return the loss. After
%  implementing the feedforward to compute the loss, you can verify that
%  your implementation is correct by verifying that you get the same loss
%  as us for the fixed debugging parameters.

fprintf('\nFeedforward with Loss Layer (Unregularised) ...\n')

J = nnLossFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y);

fprintf(['Loss at parameters (loaded from ex3weights): %f '...
         '\n(this value should be about 0.287629)\n'], J);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% =============== Part E: Implement Regularization ===============
%  Once your loss function implementation is correct, you should now
%  implement the regularisation term in nnLossRegFunction.m.
%

fprintf('\nChecking Loss Function (w/ Regularization) ... \n')

% Weight regularization parameter (we set this to 1 here).
lambda = .0002;

J = nnLossRegFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Loss at parameters (loaded from ex3weights): %f '...
         '\n(this value should be about 0.383770)\n'], J);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =============== Part F: Implement Backpropagation ===============
%  Once your cost matches up with ours, you should proceed to implement the
%  backpropagation algorithm for the neural network. You should add to the
%  code you've written in nnGradient.m to return the partial
%  derivatives of the parameters.
%
fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
checkNNGradients;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ================ Part G: Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. randInitializeWeights.m
%  initializes the weights of the neural network.

fprintf('\nInitializing Neural Network Parameters ...\n')

Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params.W1 = Theta1(:,2:end);
initial_nn_params.b1 = Theta1(:,1);
initial_nn_params.W2 = Theta2(:,2:end);
initial_nn_params.b2 = Theta2(:,1);


%% =================== Part H: Training NN with Gradient Descent ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will use gradientDescent.m. It
%  implements the gradient descent algorithm with gradient step algorithm
%  that you shall implement in gradientStep.m.
%
fprintf('\nTraining Neural Network with Gradient Descent... \n')

%  Default hyperparameters
options.MaxIter = 100;
options.LearningRate = 0.1;
options.Momentum = 0.9;
lambda = .0002;

% Create "short hand" for the cost function to be minimized
lossFunction = @(p) nnGradient(p, ...
                               input_layer_size, ...
                               hidden_layer_size, ...
                               num_labels, X, y, lambda);

% Now, lossFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, loss_vec] = gradientDescent(lossFunction, initial_nn_params, options);

% Now, test the trained parameters over the training set to measure how the
% learning has succeeded in fitting to the data in terms of accuracy.

probs = feedForward(nn_params, ...
                     input_layer_size, ...
                     hidden_layer_size, ...
                     num_labels, ...
                     X);

[dummy, pred] = max(probs, [], 1);

fprintf('\nTraining Set Accuracy: %2.2f', mean(double(pred == y)) * 100);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% =================== Part I: Training NN with Stochastic Gradient Descent ===================
%  We train the neural network with SGD. You shall implement the random
%  sampling code in stochasticGradientDescent.m.
%
fprintf('\nTraining Neural Network with Stochastic Gradient Descent... \n')

%  Default hyperparameters
options.MaxIter = 10000;
options.LearningRate = 0.1;
options.Momentum = 0.9;
options.BatchSize = 10;

lambda = .0002;

% Create "short hand" for the cost function to be minimized, now with
% control over the training batch.
lossFunction = @(p, X_batch, y_batch) nnGradient(p, ...
                                                 input_layer_size, ...
                                                 hidden_layer_size, ...
                                                 num_labels, X_batch, y_batch, lambda);

% Now, lossFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, loss_vec] = stochasticGradientDescent(lossFunction, X, y, initial_nn_params, options);

% Now, test the trained parameters over the training set to measure how the
% learning has succeeded in fitting to the data in terms of accuracy.

probs = feedForward(nn_params, ...
                     input_layer_size, ...
                     hidden_layer_size, ...
                     num_labels, ...
                     X);
[dummy, pred] = max(probs, [], 1);

fprintf('\nTraining Set Accuracy: %2.2f', mean(double(pred == y)) * 100);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

