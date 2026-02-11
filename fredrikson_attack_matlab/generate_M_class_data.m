function [X, y] = generate_M_class_data(dim, n, M, noise_std)
% [X, y] = generate_M_class_data(dim, M)

if (M > 6), fprintf('\nNumber of classes larger than six. I return...'), return, end

% Create M random centers
center = zeros(dim, M);
angles = 2*pi*(0:(M-1)) / M;
R = 10;
for iM=1:M
    %center(:,iM) = 10 * randn(dim,1);
    theta = angles(iM);
    center(1, iM) = R * cos(theta);
    center(2, iM) = R * sin(theta);
end

% Generate n pairs of M-class data
for in=1:n
    tmp = rand;
    for im = 0:M-1
        if ( (im * 1/M <= tmp)  &&  (tmp < (im+1)*1/M))
            X(:,in) = center(:,im+1) + noise_std * randn(dim, 1);
            y(in,1) = im+1;
        end 
    end
end