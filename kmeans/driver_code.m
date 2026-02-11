clc; clear all;
% DEMO_DP_KMEANS Demonstration of ε-DP k-means on synthetic data
% with comparison to non-private k-means


n = 300; d = 2; k = 3;
num_expirements = 100;
errors_chat = zeros(num_expirements, k);
errors_book_default = zeros(num_expirements, k);
errors_elemwise = zeros(num_expirements, k);

for exp_i=1:num_expirements
    % Generate synthetic 2D data (3 clusters)
    %rng(42);

    % --- Fixed centers ---
    % X1 = 0.5*randn(n/3, d) + [2, 2];
    % X2 = 0.5*randn(n/3, d) + [-2, -2];
    % X3 = 0.5*randn(n/3, d) + [5, -3];

    % --- Random centers ---
    K = 5;
    center1 = K*randn(1, 2);
    center2 = K*randn(1, 2);
    center3 = K*randn(1, 2);
    X1 = 0.5*randn(n/3, d) + center1;
    X2 = 0.5*randn(n/3, d) + center2;
    X3 = 0.5*randn(n/3, d) + center3;

    X = [X1; X2; X3];
    total_mean = mean(X);
    X = X - total_mean; % Bring data points around (0,0)
    
    % Parameters
    T = 10;                  % number of iterations
    epsilon = 0.5;             % total privacy budget
    %C = 4.0;                % clipping norm
    C = 10.0;
    BOX = [-5, -5; 8, 5];    % bounding box [L; U]
    %BOX = [-10, -10; 16, 10];
    
    L = BOX(1,:); U = BOX(2,:);
    init_centroids = rand(k, d) .* (U - L) + L;
    
    % Run DP k-means
    centroids_dp_elemwise = dp_kmeans_elemwise(X, k, T, epsilon, C, BOX, init_centroids);
    centroids_dp_book_default = dp_kmeans_book_default(X, k, T, epsilon, C, BOX, init_centroids);
    centroids_dp_chat = chat_dp_kmeans(X, k, T, epsilon, C, BOX, init_centroids);
    
    % Run non-private k-means (MATLAB built-in)
    [~, centroids_np] = kmeans(X, k, 'MaxIter', T, 'Replicates', 5);
    
    % Post process the data points
    X = X + total_mean;
    centroids_np = centroids_np + total_mean;
    centroids_dp_elemwise = centroids_dp_elemwise + total_mean;
    centroids_dp_book_default = centroids_dp_book_default + total_mean;
    centroids_dp_chat = centroids_dp_chat + total_mean;
    
    for kk = 1:k
        errors_elemwise(exp_i, kk) = norm(centroids_dp_elemwise(kk,:) - centroids_np(kk,:));
        errors_book_default(exp_i, kk) = norm(centroids_dp_book_default(kk,:) - centroids_np(kk,:));
        errors_chat(exp_i, kk) = norm(centroids_dp_chat(kk,:) - centroids_np(kk,:));
    end
end 

% Plot results
figure;
scatter(X(:,1), X(:,2), 20, 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
scatter(centroids_np(:,1), centroids_np(:,2), 200, 'go', 'LineWidth', 3);
scatter(centroids_dp_elemwise(:,1), centroids_dp_elemwise(:,2), 200, 'x', 'LineWidth', 4);
scatter(centroids_dp_book_default(:,1), centroids_dp_book_default(:,2), 200, 'x', 'LineWidth', 4);
scatter(centroids_dp_chat(:,1), centroids_dp_chat(:,2), 200, 'x', 'LineWidth', 4);
title(sprintf('ε-DP k-means vs Non-private (ε=%.2f, T=%d)', epsilon, T));
legend('Data', 'Non-private centroids', 'DP centroids (elem-wise)', 'DP centroids (book default)', 'DP centroids (chat)');
axis equal;
grid on;

xmax1 = max([errors_chat(:,1); errors_book_default(:,1); errors_book_default(:,1)]);
xmax2 = max([errors_chat(:,2); errors_book_default(:,2); errors_book_default(:,2)]);
xmax3 = max([errors_chat(:,3); errors_book_default(:,3); errors_book_default(:,3)]);

figure;
subplot(3,3,1);
histogram(errors_chat(:, 1), 20); title('Chat GPT, controid 1'); xlim([0 xmax1]);
subplot(3,3,2);
histogram(errors_chat(:, 2), 20); title('Chat GPT, controid 2'); xlim([0 xmax2]);
subplot(3,3,3);
histogram(errors_chat(:, 3), 20); title('Chat GPT, controid 3'); xlim([0 xmax3]);

subplot(3,3,4);
histogram(errors_book_default(:,1), 20); title('Book Default, centroid 1');xlim([0 xmax1]);
subplot(3,3,5);
histogram(errors_book_default(:,2), 20); title('Book Default, centroid 2');xlim([0 xmax2]);
subplot(3,3,6);
histogram(errors_book_default(:,3), 20); title('Book Default, centroid 3');xlim([0 xmax3]);

subplot(3,3,7);
histogram(errors_elemwise(:, 1), 20); title('Element-wise, centroid 1');xlim([0 xmax1]);
subplot(3,3,8);
histogram(errors_elemwise(:, 2), 20); title('Element-wise, centroid 2');xlim([0 xmax2]);
subplot(3,3,9);
histogram(errors_elemwise(:, 3), 20); title('Element-wise, centroid 3');xlim([0 xmax3]);
