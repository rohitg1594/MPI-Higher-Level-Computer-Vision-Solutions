function checkNNGradients(lambda)
%CHECKNNGRADIENTS Creates a small neural network to check the
%backpropagation gradients
%   CHECKNNGRADIENTS(lambda) Creates a small neural network to check the
%   backpropagation gradients, it will output the analytical gradients
%   produced by your backprop code and the numerical gradients (computed
%   using computeNumericalGradient). These two gradient computations should
%   result in very similar values.
%

if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end

% We test the backpropagation for a smaller network.
input_layer_size = 3;
hidden_layer_size = 5;
num_labels = 3;
N = 5;

% We generate some 'random' test data
Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);
% Reusing debugInitializeWeights to generate X
X  = debugInitializeWeights(N, input_layer_size - 1);
y  = 1 + mod(1:N, num_labels)';

X = X';
y = y';

% Unroll parameters
nn_params.W1 = Theta1(:,2:end);
nn_params.b1 = Theta1(:,1);
nn_params.W2 = Theta2(:,2:end);
nn_params.b2 = Theta2(:,1);

% Short hand for cost function
lossFunc = @(p) nnGradient(p, input_layer_size, hidden_layer_size, ...
                               num_labels, X, y, lambda);

[loss, grad] = lossFunc(nn_params);
numgrad = computeNumericalGradient(lossFunc, nn_params);

% Evaluate the norm of the difference between two solutions.  
% If you have a correct implementation, and assuming you used EPSILON = 0.0001 
% in computeNumericalGradient.m, then diff below should be less than 1e-9
diff = computeGradientDifference(numgrad,grad);

fprintf(['If your backpropagation implementation is correct, then \n' ...
         'the relative difference will be small (less than 1e-10). \n' ...
         '\nRelative Difference: %g\n'], diff);

end

function difference = computeGradientDifference(params1, params2)

  names = fieldnames(params1);
  
  sqrsumdiff = 0;
  sqrsumsum = 0;
  n = 0;

  for l = 1:numel(names)
    name = names{l};
    d = (params1.(name) - params2.(name));
    s = (params1.(name) + params2.(name));
    sqrsumdiff = sqrsumdiff + sum(d(:).^2);
    sqrsumsum = sqrsumsum + sum(s(:).^2);
    n = n + size(d,1)*size(d,2);
  end
  
  diffnorm = sqrsumdiff / n;
  sumnorm = sqrsumsum / n;
  
  difference = diffnorm / sumnorm;

end
