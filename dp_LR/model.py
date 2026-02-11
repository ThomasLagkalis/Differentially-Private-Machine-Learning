import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from utils import sample_euclid_exp



# Softmax Regression Class
class SoftmaxRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SoftmaxRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        logits = self.linear(x)

    
        return logits

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=3000):
        super(MLPClassifier, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = torch.sigmoid(self.hidden(x))
        logits = self.output(h)
        #probs = F.softmax(logits, dim=1)
        return logits


def train_model(model, train_loader, test_loader,
                verbose=True, epochs=20, lr=0.001, 
                weight_decay=0.0, reg_lambda=0.0,
                thres = 1e-5, objective_perturbation = False,
                epsilon=None, c=None
                ):

    '''
    Inputs epsilon and c are for objective perturbation only (when objective_perturbation = True)
    '''
    if objective_perturbation != None and epsilon != None and c != None:
        print(f"Starting training for epsilon_p = {epsilon}")
        W = list(model.parameters())[0]
        n = len(train_loader) * train_loader.batch_size
        e_p2 = epsilon - np.log(1 + (2*c)/(n*weight_decay) + (c**2)/((n**2) * (weight_decay**2)))
        if e_p2 > 0:
            delta = 0
        else:
            delta = c/(n*(np.exp(epsilon/4)-1)) - weight_decay
            e_p2 = epsilon/2
        beta = e_p2/2
        # Calculate empirical and theoretical variance for validation.
        var = 2*(1/beta)**2
        scale = 1/beta
        #b_np = np.random.laplace(0, scale, size=W.shape)
        b_np = sample_euclid_exp(W.shape, beta)
        b = torch.from_numpy(b_np).float()
        scipy.io.savemat('b.mat', {'b': b_np, 'delta': delta})
        mean_b = b.mean()
        squared_diffs = (b - mean_b) ** 2
        N = b.numel()
        empirical_var = squared_diffs.sum() / (N - 1)
        
    else:
        delta = 1

    criterion = nn.CrossEntropyLoss(reduction="mean")
    if objective_perturbation:
        #optimizer = optim.SGD(model.parameters(), lr=lr)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=delta)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss = []
    test_acc = []
    iteration = 0
    while True:
        model.train()
        theta_prev = list(model.parameters())
        theta_prev = theta_prev[0].detach().clone().numpy()
        for X_batch, y_batch in train_loader:
            iteration += 1
            # Adaptive learning rate 
            if reg_lambda > 0:
                #print("Entered adaptive learning rate!")
                d_k = 2.0 / (reg_lambda * (iteration + 1))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = d_k

            optimizer.zero_grad()
            outputs = model(X_batch)
            params = list(model.parameters())
            W = params[0]
            #l2_reg = 0.0
            # Objective pertrubation
            W = list(model.parameters())[0]
            n = len(y_batch)
            if objective_perturbation and b is not None:
                #loss = criterion(outputs, y_batch) + 0.5 * delta * torch.sum(W**2)
                loss = criterion(outputs, y_batch) 
                noise = torch.tensor(0.0)
                for ii in range(W.shape[0]):
                    noise += torch.dot(b[ii], W[ii])
                #loss +=  (1/n) * noise 
                loss +=  (1/n) * torch.sum(b * W) 
            else:
                loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            
            theta = list(model.parameters())
            theta = theta[0].detach().cpu().numpy()
            #with torch.no_grad():
            #    for param in model.parameters():
            #        param.clamp_(0, 1)
            theta = list(model.parameters())
            theta = theta[0].detach().numpy()
            term_cond = np.linalg.norm(theta - theta_prev)/np.linalg.norm(theta)

        # Evaluate after each epoch
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                outputs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                acc = correct / total
                test_acc.append(acc)
        if iteration%100 == 0:
            print(f"Epoch [{iteration}] term_cond value: {term_cond:.6f} Loss: {loss.item():.4f}, Test Acc: {acc:.4f}")
        if term_cond < thres:
            print(f"Epoch [{iteration}] term_cond value: {term_cond:.6f} Loss: {loss.item():.4f}, Test Acc: {acc:.4f}")
            break
    if verbose==True:
        plt.plot(train_loss)
        plt.grid(True)
        plt.title("Train loss in each epoch")
        plt.yscale("log")
        plt.show()

    return train_loss, test_acc




def train_model_grad(model, train_loader, test_loader, epochs=20, lr=0.001, 
                weight_decay=0.0, epsilon=1
                ):

    '''
    Train the model using gradient perturbation.
    '''

    criterion = nn.CrossEntropyLoss(reduction="mean")
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss = []
    test_acc = []
    iteration = 0
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            n = X_batch.shape[0]
            if epoch==1:
                print('n =', n)
            iteration += 1
            # Adaptive learning rate 
            if weight_decay > 0:
                #print("Entered adaptive learning rate!")
                #d_k = 2.0 / (weight_decay * (epoch + 1))
                d_k = 1/np.sqrt(epoch+1)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = d_k

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            if epsilon > 0:
                W = list(model.parameters())[0]
                #W = W[0].detach().numpy()
                #sensetivity = 2 * d_k / n 
                #scale = sensetivity * epochs / epsilon
                #b_np = np.random.laplace(0, scale, size=W.shape)
                l = np.sqrt(2)
                T = epochs
                N = 1600
                delta = 1/(N**2)
                sigma =  (16.0 * l / (N * epsilon)) * np.sqrt( T * np.log(2.0/delta) * np.log(2.5 * T / (delta * N)) )
                #sigma = (4*np.sqrt(2)*l*n*np.sqrt(np.log10(n/delta)*np.log10(1/delta)))/epsilon
                if epoch==1:
                    print(f"GD adding noise with sigma = {sigma:.4f}, epsilon = {epsilon:.4f}, dk = {d_k:.4f}")
                b_np = np.random.normal(0, sigma, W.shape)
                b = torch.from_numpy(b_np).float()

            
                if W.grad is not None:
                    noise =  b
                    W.grad.data.add_(noise)

            optimizer.step()
            train_loss.append(loss.item())
            
        # Evaluate after each epoch
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                outputs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                acc = correct / total
                test_acc.append(acc)
        if iteration%10 == 0:
            print(f"Epoch [{epoch+1}] Loss: {loss.item():.4f}, Test Acc: {acc:.4f}")

    return train_loss, test_acc
