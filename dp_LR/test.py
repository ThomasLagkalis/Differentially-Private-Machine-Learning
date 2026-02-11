import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as sio
import matplotlib.pyplot as plt

def simple_gradient(x):
    x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    y = x**2

    y.backward()
    return x.grad

def logistic_regression_cost(X, y, W, b=None, reg_lambda=0.0):
    N, d = X.shape
    M = W.shape[0]

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    W = torch.tensor(W, dtype=torch.float32, requires_grad=True)
    if b is None:
        b = torch.zeros(M, dtype=torch.float32, requires_grad=True)
    else:
        b = torch.tensor(b, dtype=torch.float32, requires_grad=True)

    # logits: (N, M)
    logits = X @ W.T + b

    ce_loss = F.cross_entropy(logits, y, reduction="mean")

    l2_reg = 0.5 * reg_lambda * torch.sum(W ** 2)

    loss = ce_loss + l2_reg

    loss.backward()

    print("Gradient W:\n", W.grad.numpy())
    print("Gradient b:\n", b.grad.numpy())

    return loss.item()


def logistic_regression_cost_grad(X, y, W, b=None, reg_lambda=0.0, reg_bias=False):
    """
    Multiclass logistic regression cost and gradient with L2 regularization.

    Args:
        X : ndarray (N, d) input data (rows = samples)
        y : ndarray (N,) class labels in {0,...,M-1}
        W : ndarray (M, d) weight matrix
        b : ndarray (M,) bias vector (optional)
        reg_lambda : float, L2 regularization coefficient
        reg_bias : bool, whether to regularize bias too

    Returns:
        cost : float
        grad_W : ndarray (M, d)
        grad_b : ndarray (M,)
    """
    N, d = X.shape
    M = W.shape[0]
    if b is None:
        b = np.zeros(M)

    # logits: (N, M)
    logits = X @ W.T + b  # broadcasting adds bias

    # softmax probabilities (N, M)
    logits -= np.max(logits, axis=1, keepdims=True)  # stability
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Cross-entropy loss (mean over samples)
    ce_loss = -np.mean(np.log(probs[np.arange(N), y]))

    # L2 penalty
    l2_reg = 0.5 * reg_lambda * np.sum(W ** 2)
    if reg_bias:
        l2_reg += 0.5 * reg_lambda * np.sum(b ** 2)

    cost = ce_loss + l2_reg

    # Gradients
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(N), y] = 1

    dlogits = (probs - one_hot) / N   # (N, M)

    grad_W = dlogits.T @ X + reg_lambda * W   # (M, d)
    grad_b = np.sum(dlogits, axis=0)
    if reg_bias:
        grad_b += reg_lambda * b

    return cost, grad_W, grad_b


def fun_grad_M_class_LR(theta, X, y, lambda_):
    """
    Computes the gradient of M-class Logistic Regression.
    
    Parameters:
    - theta: (d x M) coefficient matrix
    - X: (d x n) data matrix (columns are data points)
    - y: (n,) labels in {1, ..., M}
    - lambda_: regularization parameter
    
    Returns:
    - grad: (d x M) gradient matrix
    """
    n = X.shape[1]  # number of samples
    M = theta.shape[1]
    
    tmp_grad = np.zeros_like(theta)
    
    for ii in range(n):
        tmp = 0
        for ll in range(M):
            tmp += np.exp(np.dot(theta[:, ll], X[:, ii]))
        
        for jj in range(M):
            if y[ii] == jj + 1:  # MATLAB indices start at 1
                tmp_grad[:, jj] -= X[:, ii]
            tmp_grad[:, jj] += (np.exp(np.dot(theta[:, jj], X[:, ii])) / tmp) * X[:, ii]
    
    grad = tmp_grad / n + lambda_ * theta
    return grad


def main():
    x = torch.tensor(3.0, requires_grad=True)
    optimizer = torch.optim.SGD([x], lr=0.1)
    f = x**2
    f.backward()
    
    print("Pre addition gradient: ", x.grad.item())

    x.grad.data.add_(1.0)
    print("Post addition gradient: ", x.grad.item())

    optimizer.step()

    print('x_new after GD step: ', x.item())

    return 

if __name__ == '__main__':
    main()
