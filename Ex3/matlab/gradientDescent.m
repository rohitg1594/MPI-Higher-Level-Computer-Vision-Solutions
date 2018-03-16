function [nn_params, loss_vec] = gradientDescent(lossFunction, initial_nn_params, options)

nn_params = initial_nn_params;
layerParams = fieldnames(nn_params);

% Initialise momentum vectors
for l = 1:numel(layerParams)
  name = layerParams{l};
  momentumVec.(name) = zeros(size(nn_params.(name)));
end

% Gradient descent iterations
for it = 1:options.MaxIter
  
  % Compute the loss and gradient of the neural network at current
  % parameter values.
  
  [c, grad] = lossFunction(nn_params);
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