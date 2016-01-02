function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

    H = sigmoid(theta.' * X.')

    %calculate the cost
    for i = 1:size(X,1)
        J = J + ((-y(i)*log(H(i))) - ((1-y(i))*log(1-H(i))))
    end
    J = J/m + lambda/(2*m) * sum(theta(2:end).^2)
    
    %calcuate the gradients
    temp = 0
    for k = 1:size(theta,1) 
        for i = 1:size(X,1)
            temp = temp + (H(i) - y(i)) * X(i,k)
        end
        temp = temp / m
        if (k > 1)
           temp = temp + (lambda * theta(k) / m) 
        end
        grad(k) = temp
        temp = 0
    end


% =============================================================

end
