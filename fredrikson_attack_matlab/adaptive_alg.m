function [theta_grad, iters] = adaptive_alg(theta_init, X, y, lambda)
%
% M file that implements the gradient with backtracking for the
% minimization of the M-class Logistic Regression cost function

alpha = 0.3;
beta = 0.5;
thresh = 10^(-7);

iter = 1;
theta_grad = theta_init;
while (1)
    
    % Compute gradient
    grad = fun_grad_M_class_LR(theta_grad, X, y, lambda);
    
    % Backtracking line search
    tau = 2/(lambda*(iter+50));
    %tau = 0.1;
    %while ( fun_M_class_LR(theta_grad - tau * grad, X, y, lambda) > fun_M_class_LR(theta_grad, X, y, lambda) - alpha * tau * norm(grad,'fro')^2 ) 
    %    tau = beta * tau; %, pause(1)
    %end
    new_theta_grad = theta_grad - tau * grad;
    %new_theta_grad = clip(new_theta_grad, 0, 1);
    
    %if (mod(iter, 100) == 0), plot([vec(new_theta_grad)]), pause(0.01), end 
    term_cond = norm(new_theta_grad - theta_grad)/(10^(-10)+norm(new_theta_grad));
    if ( term_cond < thresh ), break, end

    %if ( iter > 100), break, end
    theta_grad = new_theta_grad;
    iter = iter + 1;
    iters = iter;

    %img = theta_grad(:, 1);
    %min_pixel = min(img);
    %max_pixel = max(img);
    %img = reshape(img, 112, 92);
    %figure(1);
    %imshow(img, [min_pixel, max_pixel])
    %cost = fun_M_class_LR(theta_grad, X, y, lambda);
    
end