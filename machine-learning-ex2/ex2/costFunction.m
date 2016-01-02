function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

    H = sigmoid(X * theta)
    J = sum((-y.*log(H)) - ((1-y).*log(1-H))) / m
    
    %calculate the cost
    %for i = 1:size(X,1)
    %    J = J + ((-y(i)*log(H(i))) - ((1-y(i))*log(1-H(i))))
    %end
    %J = J / m
    
    grad = X.' * (H-y) / m
    %calcuate the gradients
    %temp = 0
    %for k = 1:size(theta,1) 
    %    for i = 1:size(X,1)
    %        temp = temp + (H(i) - y(i)) * X(i,k)
    %    end
    %    temp = temp / m
    %    grad(k) = temp
    %    temp = 0
    %nd





% =============================================================

end
