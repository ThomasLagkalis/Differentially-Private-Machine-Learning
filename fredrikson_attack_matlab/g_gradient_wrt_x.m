function grad_X = g_gradient_wrt_x(theta, X)
% Calculates the derivative of g_m(x; theta) wrt x (see the notation
% in page 52 A. Lindholm, N. Wahlström et al. book of ML)
% Computes the derivative for all m = 1,...,M
%
% theta : (d x M) matrix of class weights (each column = theta_m)
% x     : (d x 1) input vector
% grad_X: (d x M) matrix, where column m = \nabla_x g_m(x; theta)
[d, M] = size(theta);
z = theta' * X; % Logits (M x 1)
g = exp(z) ./ sum(exp(z)); % apply softmax to z (M x 1)
g_theta = theta * g; % (d x 1)

grad_X = zeros(d, M);

for m = 1:M
    grad_X(:,m) = g(m) * (theta(:,m) - g_theta);
end
end

