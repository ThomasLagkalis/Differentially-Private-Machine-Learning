function grad = lr_grad_wrt_x(theta, X, label, lambda)
% Gradient wrt input x for multiclass LR 
[d, n] = size(X);
M = size(theta, 2);

grad = zeros(d, n);
for ii=1:n
    z = theta' * X(:,ii);
    g = exp(z) / sum(exp(z));          % softmax
    weighted_theta = theta * g;        % weighted average
    grad(:,ii) = weighted_theta - theta(:, label) + lambda*X(:, ii);
end
end