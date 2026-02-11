function grad = fun_grad_M_class_LR(theta, X, y, lambda)
%
% Computes the gradient of the M-class Logistic Regression (see Notes)
% The coefficients form an n x M matrix, where n is the dimensipn of the data


n = length(y);
M = size(theta, 2);

tmp_grad = zeros(size(theta));
for ii=1:n
    tmp = 0;
    for ll=1:M 
        tmp = tmp + exp( theta(:,ll)' * X(:,ii) ); 
    end
    for jj=1:M
        if ( y(ii) == jj ) 
            tmp_grad(:, jj) = tmp_grad(:, jj) - X(:,ii) + 1/tmp * exp( theta(:,jj)' * X(:,ii) ) * X(:,ii);
        else
            tmp_grad(:, jj) = tmp_grad(:, jj) + 1/tmp * exp( theta(:,jj)' * X(:,ii) ) * X(:,ii);
        end
    end
    
end
grad = 1/n * tmp_grad + lambda * theta;
