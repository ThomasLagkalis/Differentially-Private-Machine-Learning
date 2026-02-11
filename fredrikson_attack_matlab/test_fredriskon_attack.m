clear, clc, clf

n = 100;  % number of data pairs (x_i, y_i)
N = 112*92;    % dimension of each x_i
%N = 2;
M = 40;    % number of classes
%M=5;
noise_std = 1; % determinew how noisy our classes are

% Load images
%imgs = zeros(112*92, 40*10);
%y = zeros(40*10, 1);
%for c=1:40
%    for i=1:10
%        path = sprintf('orl/s%d/%d.pgm', [c, i]);
%        x = imread(path);
%        imgs(:, (c-1)*10 + i) = x(:);
%        y((c-1)*10+i) = c;
%    end
%end

%imgs=imgs/255;

%[X, y] = generate_M_class_data(N, n, M, noise_std);
train_loader = load('../dp_LR/train_loader.mat');
X = train_loader.train_data';
y = train_loader.train_labels' + 1;
N = 3;
n = 400;
M = 10;
% X== [ones(1,100); X];
%if (N == 2), figure(1), plot_M_class_data(X, y, M); pause(.001), end
%X = imgs;
f_LR = @(theta, X, y) fun_M_class_LR(theta, X, y);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solve multi-class Logistic Regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%fprintf('\nSolution of M-class Logistic Regression classification via CVX...')

lambda = 0.001;
%cvx_begin quiet
%    variable theta_cvx(N+1, M);
%        minimize ( fun_M_class_LR(theta_cvx, X, y, lambda) )
%cvx_end
fprintf('\nSolution of M-class Logistic Regression classification via gradient...')


%theta_init = randn(N,M);
theta_init = zeros(N,M);
%theta_grad = alg_grad_M_class_LR(theta_init, X, y, lambda, theta_cvx);
[theta_grad, iters] = adaptive_alg(theta_init, X, y, lambda);
%w = load('../2d_multiclass_classification/weights.mat');
%theta_grad = w.w;
%theta_grad = theta_grad';
%rec_img = w.x;
%rec_vec = reshape(rec_img, 1, 112*92);
return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ATTACK
% Note: Here the cost = f_label
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
thres = 10^-10;
label = 2;
tv_weight = 0.0;
a = 100000;
beta = 50000;
gamma = 0.99;
step = 0.1;
%x = zeros(N,1);
%x = theta_grad(:, label);
x = 0.05*randn(N, 1);
%x = rand(N,1);
best_prob = 0;
iters_no_change = 0;
i=0;

figure;

while true
    plot(x(1:100));hold on; plot(rec_vec(1:100)); legend('x','python'); pause(0.00001); hold off;
    x_prev = x;
    i = i+1
    %grad_x = g_gradient_wrt_x(theta_grad, x);
    %grad_x_label = grad_x(:, label);
    x_2d = reshape(x, [112, 92]);
    gradtv = grad_tv(x_2d);
    gradtv = reshape(x_2d, [92*112, 1]);
    grad_x_label = lr_grad_wrt_x(theta_grad, x, label, lambda) + tv_weight*gradtv;
    x = x - step * grad_x_label;

    p = LR_softmax(theta_grad, x);
    p(label)
    cond_val = abs(norm(x) - norm(x_prev))/norm(x)
    if (cond_val< thres), break, end

end

min_pixel = min(x);
max_pixel = max(x);
img = reshape(x, 92, 112);
figure(1);
imshow(img', [min_pixel, max_pixel], Border="tight");
%saveas(gcf, 'reconstruction.png');
figure(2);
plot(x); hold on;
plot(rec_vec);
legend('Matlab', 'Python')

% Draw the vectors
if (N==2)
    plot_M_class_data(X, y, M); hold on;
    starts = zeros(2,2);
    ends = [x'; theta_grad(:, label)'];

    quiver(starts(:,1), starts(:,2), ends(:,1), ends(:,2))
    axis equal
end
