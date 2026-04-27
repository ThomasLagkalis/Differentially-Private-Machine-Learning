import torch.nn as nn
import torch

def grad_norm(model, norm_type=2):
    """
    Computes the norm of gradients of all parameters in a model.

    Args:
        model (nn.Module): PyTorch model with computed gradients
        norm_type (float): type of norm (2 for L2, 1 for L1, float('inf') for max)

    Returns:
        float: gradient norm
    """
    total_norm = 0.0

    if norm_type == float('inf'):
        total_norm = max(
            p.grad.abs().max().item()
            for p in model.parameters()
            if p.grad is not None
        )
        return total_norm

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type

    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm



def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(768, 10)
        )

    def forward(self, x):
        x = self.body(x)
        x = torch.flatten(x, 1)
        # print(out.size())
        x = self.fc(x)
        return x


class MLP(nn.Module):
    '''
    Multi-Layer perceptron
    '''
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        act = nn.Sigmoid
        #act = nn.ReLU
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return (self.layers(x))

def train(model, train_loader, test_loader, epochs, optimizer, loss_function):
    loss_per_epoch = []
    test_acc_per_epoch = []
    train_acc_per_epoch = []
    for epoch in range(epochs):
        model.train()
        tot_loss = 0
        correct, total = 0, 0
        for i, data in enumerate(train_loader, 0):
            x_batch, y_batch = data
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = loss_function(outputs, y_batch)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()

            outputs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        loss_per_epoch.append(tot_loss)
        train_acc_per_epoch.append(correct/total)
    
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
            test_acc_per_epoch.append(acc)
    
        print(f"Epoch {epoch+1}: Test acc: {test_acc_per_epoch[epoch]:.4f}, train acc: {train_acc_per_epoch[epoch]:.4f}, epoch loss: {loss_per_epoch[epoch]:.4f}")
    
    return test_acc_per_epoch, train_acc_per_epoch, loss_per_epoch, tot_loss
            


