function res = fun_M_class_LR(theta, X, y, lambda)
%
% Computes the cost function of the M-class Logistic Regression
% See notes

n = length(y);
M = size(theta, 2);
 
tmp = 0;
for ii=1:n
    tmp1 = - theta(:, y(ii))' * X(:,ii);
    tmp2 = 0;
    for jj=1:M
        tmp2 = tmp2 + exp( theta(:,jj)' * X(:,ii) );
    end
    tmp = tmp + tmp1 + log( tmp2 ) ;
end
res = tmp / n + 0.5 * lambda * theta(:)' * theta(:);
