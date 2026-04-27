# Source: https://www.codegenes.net/blog/softmax-regression-pytorch/
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class WeightClipper(object):

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            print("Entered")
            w = module.weight.data
            w = w.clamp(0,1)
            module.weight.data = w
            module.weight.data = w

# Softmax Regression Class with rounding
class SoftmaxRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SoftmaxRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.rounding_precision = None  # No rounding by default

    def set_rounding_precision(self, r):
        """Set rounding precision for confidence scores"""
        self.rounding_precision = r

    def forward(self, x):
        logits = self.linear(x)
        #probs = torch.softmax(logits, dim=1) 
        
        return logits


def load_orl_faces(dataset_path):
    images, labels = [], []
    avg_faces = []

    for person_id in range(40):  # 40 subjects
        folder = os.path.join(dataset_path, f"s{person_id+1}")
        person_images = []
        
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (92, 112))  # ensure consistent size
            img = img.astype(np.float32) / 255.0
            person_images.append(img)
            images.append(img.flatten())
            labels.append(person_id)
        
        avg_face = np.mean(person_images, axis=0)
        avg_faces.append(avg_face.flatten())
    
    return np.array(images), np.array(labels), np.array(avg_faces)

def tv_loss(img: np.ndarray) -> float:
    """
    Compute Total Variation (TV) loss for a 2D grayscale image.

    Parameters
    ----------
    img : np.ndarray
        Input image of shape (112, 92).

    Returns
    -------
    float
        The TV loss value.
    """
    # vertical differences (downward neighbors)
    diff_vertical = np.abs(img[1:, :] - img[:-1, :])

    # horizontal differences (rightward neighbors)
    diff_horizontal = np.abs(img[:, 1:] - img[:, :-1])

    # sum them up
    tv = np.sum(diff_vertical) + np.sum(diff_horizontal)
    return tv


# Training Function
# Cross entropy loss documentation:
# https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
def train_model(model, train_loader, test_loader,
                verbose=True, epochs=20, lr=0.001, 
                weight_decay=0.0, bloss_weight=0.0, 
                tv_weight=0.0, reg_lambda=0.0 
                ):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loss = []
    test_acc = []
    iteration = 0

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            iteration += 1
            # Adaptive learning rate 
            if reg_lambda > 0:
                d_k = 2.0 / (reg_lambda * (iteration + 5))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = d_k

            optimizer.zero_grad()
            outputs = model(X_batch)
            # Calculate TV loss of the weights
            params = list(model.parameters())
            W = params[0]
            tvl = 0.0
            black_loss = 0.0
            l2_reg = 0.0
            for i in range(40):
                img = W[i].detach().cpu().numpy()
                black_loss += sum(np.max(img) - img)
                l2_reg += sum(np.absolute(img)**2)
                img = img.reshape(112,92)
                tvl += tv_loss(img)
            if reg_lambda != 0:
                l2_reg = reg_lambda/2 * l2_reg
                loss = criterion(outputs, y_batch) + tv_weight * tvl + bloss_weight * black_loss * l2_reg
            else:
                loss = criterion(outputs, y_batch) + tv_weight * tvl + bloss_weight * black_loss 

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            with torch.no_grad():
                for param in model.parameters():
                    param.clamp_(0, 1)

        # Evaluate after each epoch
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                #outputs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                acc = correct / total
                test_acc.append(acc)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}, Test Acc: {acc:.4f}")
    if verbose==True:
        plt.plot(train_loss)
        plt.grid(True)
        plt.title("Train loss in each epoch")
        plt.yscale("log")
        plt.show()

    return train_loss, test_acc

# Prediction Helper
def predict_image(model, img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (92, 112)).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img.flatten(), dtype=torch.float32).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    return predicted_class
