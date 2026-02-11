import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as sio

import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as sio



def logistic_regression_cost(X, y, W, b=None, reg_lambda=0.0):
    N, d = X.shape
    M = W.shape[0]

    X = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    y = torch.tensor(y, dtype=torch.long)

    W = torch.tensor(W, dtype=torch.float32)
    if b is None:
        b = torch.zeros(M, dtype=torch.float32)
    else:
        b = torch.tensor(b, dtype=torch.float32)

    # logits: (N, M)
    logits = X @ W.T + b

    ce_loss = F.cross_entropy(logits, y, reduction="mean")

    l2_reg = 0.5 * reg_lambda * torch.sum(W ** 2)

    loss = ce_loss + l2_reg

    loss.backward()

    print("Gradient X:\n", X.grad.numpy())

    return loss.item() 



data_mat = sio.loadmat('../fredrikson_attack_matlab/test_data.mat')
data = np.array(data_mat['X'].T)
labels = np.array(data_mat['y'])
theta = np.array(data_mat['theta'])
labels = labels.reshape(10,)-1

data = torch.tensor(data, requires_grad=True)
cost = logistic_regression_cost(data, labels, theta.T)
