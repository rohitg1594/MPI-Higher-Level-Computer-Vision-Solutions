function numgrad = computeNumericalGradient(J, nn_params)
%COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
%and gives us a numerical estimate of the gradient.
%   numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
%   gradient of the function J around theta. Calling y = J(theta) should
%   return the function value at theta.

e = 1e-4;

names = fieldnames(nn_params);

for l = 1:numel(names)
  name = names{l};
  numgrad.(name) = zeros(size(nn_params.(name)));
  shape = size(nn_params.(name));
  
  for i = 1:shape(1)
    for j = 1:shape(2)
      % Set perturbation vector
      perturbed_pos = nn_params;
      perturbed_neg = nn_params;
      
      % Set small positive and negative perturbation at position (i,j)
      perturbed_pos.(name)(i,j) = perturbed_pos.(name)(i,j) + e;
      perturbed_neg.(name)(i,j) = perturbed_neg.(name)(i,j) - e;
      
      % ====================== YOUR CODE HERE ======================
      % Instructions: Implement the numerical gradient
      % Input: perturbed_pos (neural net parameter positively perturbed at (i,j)
      %        perturbed_neg (neural net parameter negatively perturbed at (i,j)
      % Output: g, numerical gradient of J corresponding to the parameter at (i,j)

      
      
      
      % -------------------------------------------------------------
      % =========================================================================
      
      % Numerical gradient
      numgrad.(name)(i,j) = g;
    end
  end
end

end
