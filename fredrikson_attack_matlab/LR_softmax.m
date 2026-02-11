function res = LR_softmax(theta, X_i)
% Returns a probability vectror, which is the result of
% a forward pass on a Multiclass LR model with input vector X_i and t
% model weights theta.

logits = theta' * X_i;

res = exp(logits)/sum(exp(logits));

end

