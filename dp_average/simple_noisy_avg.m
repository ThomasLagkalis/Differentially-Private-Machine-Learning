function [true_avg, avg] = simple_noisy_avg(epsilon,a_min, a_max, D)
%SIMPLE_NOISY_AVG Summary of this function goes here
%   Detailed explanation goes here
S = sum(D) + (2*(a_max-a_min)/epsilon)*randl;
C = length(D) + (2/epsilon)*randl;

avg = ((a_min+a_max)/2)*(C<=1) + (S/C)*(C>1);
true_avg = sum(D)/length(D);
end

