function centroids = dp_kmeans_elemwise(X, k, T, epsilon, C, BOX, init_centroids)
%DP_KMEANS ε-Differentially Private Lloyd's k-means
%   centroids = dp_kmeans(X, k, T, epsilon, C, BOX)
%
%   Inputs:
%       X        - n x d data matrix
%       k        - number of clusters
%       T        - number of iterations
%       epsilon  - total privacy budget
%       C        - clipping norm (L2)
%       BOX      - [L; U] bounding box for initialization (2 x d)
%
%   Output:
%       centroids - k x d matrix of final DP centroids

[n, d] = size(X);


% --- privacy budget allocation ---
eps_per_iter = epsilon / T;
eps_count = eps_per_iter / 2;
eps_sum = eps_per_iter / 2;

% --- data-independent initialization ---
L = BOX(1,:); U = BOX(2,:);
centroids = init_centroids;


% --- main loop ---
for t = 1:T
    % 1. Clip data points
    % TODO: plot x_clipped and x.
    X_clipped = X;
    norms = sqrt(sum(X.^2, 2));
    scale = min(1, C ./ norms);
    X_clipped = X .* scale;

    r = zeros(1,d);
    for i=1:d
        r(i) = max(abs(X_clipped(:,i)));
    end
    e = epsilon/(sum(r)+1); % For book implementation using coordinate-wise sensetivity.
    
    % 2. Assign to nearest centroid
    D = pdist2(X_clipped, centroids);
    [~, assignments] = min(D, [], 2);
    
    % 3. Aggregate exact sums and counts
    S = zeros(k, d);
    N = zeros(k, 1);
    for j = 1:k
        members = (assignments == j);
        N(j) = sum(members);
        if N(j) > 0
            S(j,:) = sum(X_clipped(members,:), 1);
        end
    end
    
    % 4. Noise scales
    %b_count = 1 / eps_count;
    b_count = 1/e;
    %b_sum = (sqrt(sum(r)) * C) / eps_sum;  
    %b_sum = (d * C) / eps_sum;
    b_sum = 1/e;

    % 5. Add Laplace noise
    N_tilde = N + laplace_noise([k,1], 0, b_count);
    S_tilde = S + laplace_noise([k,d], 0, b_sum);
    
    % 6. Update centroids
    for j = 1:k
        if N_tilde(j) <= 0
            centroids(j,:) = rand(1,d) .* (U - L) + L;
        else
            centroids(j,:) = S_tilde(j,:) ./ N_tilde(j);
        end
        % project back into BOX
        centroids(j,:) = min(max(centroids(j,:), L), U);
    end
end
% figure; hold on;
% 
% scatter(X(:,1),        X(:,2),        20, 'b', 'filled'); 
% scatter(X_clipped(:,1), X_clipped(:,2), 20, 'r', 'filled'); 
% 
% legend('X\_clipped','X');
% xlabel('X_1');
% ylabel('X_2');
% title('Scatter Plot with Different Colors');
% grid on;


end

% Helper: Laplace noise generator
function Z = laplace_noise(sz, mu, b)
    U = rand(sz) - 0.5;
    Z = mu - b * sign(U) .* log(1 - 2*abs(U));
end
