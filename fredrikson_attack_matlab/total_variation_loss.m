function tv = total_variation_loss(X)
% TOTAL_VARIATION_LOSS Computes isotropic total variation loss of a 2D matrix
%
%   tv = total_variation_loss(X)
%
%   X : input 2D matrix (e.g. image)
%   tv: scalar

dx = diff(X,1,2);   % horizontal differences (cols)
dy = diff(X,1,1);   % vertical differences (rows)

tv = sum(abs(dx(:))) + sum(abs(dy(:)));
end
