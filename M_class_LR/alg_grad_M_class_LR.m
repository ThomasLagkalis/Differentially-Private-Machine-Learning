function theta_grad = alg_grad_M_class_LR(theta_init, X, y, lambda, theta_opt)
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
    tau = 1;
    while ( fun_M_class_LR(theta_grad - tau * grad, X, y, lambda) > fun_M_class_LR(theta_grad, X, y, lambda) - alpha * tau * norm(grad,'fro')^2 ) 
        tau = beta * tau; %, pause(1)
    end
    new_theta_grad = theta_grad - tau * grad;
    
    if (mod(iter, 100) == 0), plot([vec(theta_opt) vec(new_theta_grad)]), pause(.001), end 
    
    if ( norm(new_theta_grad - theta_grad) < thresh ), break, end
    theta_grad = new_theta_grad;
    iter = iter + 1;
    
end
