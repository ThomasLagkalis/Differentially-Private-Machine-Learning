function res = attack_cost(theta, X, y, lambda, label)
%ATTACK_COST Summary of this function goes here
%   Detailed explanation goes here
f = fun_M_class_LR(theta, X, y, lambda);
res = 1 - f
end

