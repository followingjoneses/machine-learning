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

n = length(theta);

sigma1 = 0;
sigma2 = 0;
sigma3 = 0;

for i = 1 : m
	h = sigmoid(X(i, :)*theta);
	sigma1 = sigma1 - (y(i)*log(h) + (1-y(i))*log(1-h));
end

for j = 2 : n
	sigma2 = sigma2 + theta(j).^2;
end

J = ((1/m)*sigma1) + ((lambda/(2.*m))*sigma2);

grad = (1/m)*X'*(sigmoid(X*theta)-y) + (lambda/m)*theta;

for k = 1 : m
	sigma3 = sigma3 + (sigmoid(X(k, :)*theta) - y(k));
end

grad(1) = (1/m).*sigma3;

% =============================================================

end
