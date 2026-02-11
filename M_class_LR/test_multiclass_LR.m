%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% M file that tests M-class Logistic Regression                 %
%          See Lindholm et al, page 53                          %
%                                                               %
% A. P. Liavas, Oct. 11, 2024                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear, clc, clf

n = 100;  % number of data pairs (x_i, y_i)
N = 2;    % dimension of each x_i
M = 4;    % number of classes
noise_std = 2; % determinew how noisy our classes are


[X, y] = generate_M_class_data(N, n, M, noise_std);
X = [ones(1,100); X];
if (N == 2), figure(1), plot_M_class_data(X(2:end, :), y, M); pause(.001), end

f_LR = @(theta, X, y) fun_M_class_LR(theta, X, y);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solve multi-class Logistic Regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nSolution of M-class Logistic Regression classification via CVX...')
    
lambda = .1;
cvx_begin quiet
    variable theta_cvx(N+1, M);
        minimize ( fun_M_class_LR(theta_cvx, X, y, lambda) )
cvx_end
fprintf('\nSolution of M-class Logistic Regression classification via gradient...')


%theta_init = randn(N,M);
theta_init = zeros(N+1,M);
%theta_grad = alg_grad_M_class_LR(theta_init, X, y, lambda, theta_cvx);
[theta_grad, iters] = adaptive_alg(theta_init, X, y, lambda, theta_cvx);
theta = vec(theta_grad(2:end, :));
grad = fun_grad_M_class_LR(theta_grad, X, y, lambda)
X = X(2:end, :);
err = abs(theta_grad - theta_cvx)
theta_grad = theta_grad(2:end,:)
save('test_data.mat', 'X', 'y', 'theta', 'theta_grad');

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If N = 2, plot SVM-based decision regions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (N == 2)
   [x1, x2] = constract_grid_x(X);
   hold on
   plot_M_class_data(X, y, M)
   plot_decision_regions(theta_grad, x1, x2); pause(.01)
end
hold off