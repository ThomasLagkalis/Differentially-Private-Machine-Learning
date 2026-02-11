function grad = grad_tv(X)
% grad = grad_tv(X)
% Computes the gradient of the total variation function wrt input X (e.g.
% image
% X : input matrix H x W
% grad : gradient H x W
[H, W] = size(X);
grad = zeros(H, W);


% TODO: Handle edge cases.
grad = zeros(H,W);
for r=1:H
    for c=1:W
        if r>1
            grad(r, c) = grad(r, c) + sign(X(r,c) - X(r-1,c));
        end
        if c>1
            grad(r, c) = grad(r, c) + sign(X(r,c) - X(r,c-1));
        end
        if r<H
            grad(r, c) = grad(r, c) - sign(X(r+1,c) - X(r,c));
        end
        if c<W
            grad(r, c) = grad(r, c) - sign(X(r,c+1) - X(r,c));
        end
    end
end

end

