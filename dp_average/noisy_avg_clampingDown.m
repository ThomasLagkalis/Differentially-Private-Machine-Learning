function [true_avg, avg] = noisy_avg_clampingDown(epsilon,a_min, a_max, D)
%NOISY_AVG_RESAMPLE Summary of this function goes here
%   Detailed explanation goes here
S = sum(D);
C = length(D);

if C ~= 0
    A = (S + (2*(a_max-a_min)/epsilon)*randl)/C;
    if A < a_min
        A = a_min;
    elseif A > a_max
        A = a_max;
    end
    avg = A;
    true_avg = S/C;
else
    r = rand;
    avg = a_min*(r < 0.5*exp(-epsilon/2)) + a_max*(r<exp(-epsilon/2) && r>0.5*exp(-epsilon/2)) + random('unif', a_min, a_max)*(r > exp(-epsilon/2));
end
end

