function [theta_grad, iters] = adaptive_alg(theta_init, X, y, lambda, theta_cvx)
%
% M file that implements the gradient with backtracking for the
% minimization of the M-class Logistic Regression cost function

alpha = 0.3;
beta = 0.5;
thresh = 10^(-5);

iter = 1;
theta_grad = theta_init;
while (1)
    
    % Compute gradient
    grad = fun_grad_M_class_LR(theta_grad, X, y, lambda);
    
    % Backtracking line search
    tau = 2/(lambda*(iter+50));
    %tau = 0.01;
    %while ( fun_M_class_LR(theta_grad - tau * grad, X, y, lambda) > fun_M_class_LR(theta_grad, X, y, lambda) - alpha * tau * norm(grad,'fro')^2 ) 
    %    tau = beta * tau; %, pause(1)
    %end
    new_theta_grad = theta_grad - tau * grad;
    
    %if (mod(iter, 100) == 0), plot([vec(theta_cvx) vec(new_theta_grad)]), pause(0.01), end 
    
    if ( norm(new_theta_grad - theta_grad) < thresh), break, end
    %if ( iter > 100), break, end
    theta_grad = new_theta_grad;
    iter = iter + 1;
    iters = iter
    
end