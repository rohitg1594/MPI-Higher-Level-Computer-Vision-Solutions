function [nn_params, loss_vec] = stochasticGradientDescent(lossFunction, X, y, initial_nn_params, options)

nn_params = initial_nn_params;
layerParams = fieldnames(nn_params);

% Initialise momentum vectors
for l = 1:numel(layerParams)
  name = layerParams{l};
  momentumVec.(name) = zeros(size(nn_params.(name)));
end

% Gradient descent iterations
for it = 1:options.MaxIter
  N = size(X, 2);
  B = options.BatchSize;
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Write a random training subset selector.
  % Input: X. entire training set. 400xN matrix
  %        y. training labels. vector of dimension N
  %        N. size of the entire training set
  %        B. desired batch size
  % Output: X_batch. a random subset of X. 400xB matrix. Make sure
  %                  different random subsets are picked at each iteration.
  %         y_batch. the random subset of y corresponding to X_batch.
  %                  vector of dimension B
  
  
  
  
  
  % -------------------------------------------------------------
  % =========================================================================
  
  % Compute the loss and gradient of the neural network at current
  % parameter values and selected training batch.
  
  [c, grad] = lossFunction(nn_params, X_batch, y_batch);
  loss_vec(it) = c;
  fprintf('Iter %3d. Loss: %2.6f\n', it-1, c);
  
  % Update the parameters for each layer.
  
  for l = 1:numel(layerParams)
    % Prepare variables
    
    name = layerParams{l};
    v = momentumVec.(name);
    p = nn_params.(name);
    d = grad.(name);
    
    % Gradient descent step with momentum.
    
    [p, v] = gradientStep(d,p,v,options.Momentum,options.LearningRate);
    
    % Update variables
    
    nn_params.(name) = p;
    momentumVec.(name) = v;
  end
  
end

end