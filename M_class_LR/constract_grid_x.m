function [x1, x2] = constract_grid_x(X) 
% [x1, x2] = constract_grid_x(X);

x1_min = min(X(1,:));
x1_max = max(X(1,:));
x2_min = min(X(2,:));
x2_max = max(X(2,:));

step = (x1_max - x1_min)/50;

x1 = x1_min:step:x1_max;
x2 = x2_min:step:x2_max;


