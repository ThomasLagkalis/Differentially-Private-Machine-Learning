from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
import torch
import numpy as np
import os
import cv2


def sample_euclid_exp(shape, beta):
    """
    Sample from f(x) ~ exp(-beta * ||x||_2) in R^d.
    shape: the tensor shape (e.g. (40, 10304))
    """
    d = np.prod(shape)

    # 1. Sample radius: Gamma(shape=d, scale=1/beta)
    r = np.random.gamma(shape=d, scale=1.0/beta, size=1)  # scalar radius

    # 2. Sample random direction: normalize Gaussian vector
    x = np.random.normal(size=d)
    x /= np.linalg.norm(x)

    # 3. Construct noise vector and reshape
    return (r * x).reshape(shape)


def calculate_accuracy(model, dataset, labels):
    '''
    Function that calculates the accuracy on all the dataset.
    Inputs: - model: the model
            - dataset: the test dataset
            - labels: the labels
    Reutrns: a scalar that is the accuracy over the dataset
    '''
    j=0
    tot = len(dataset)
    correct = 0
    for x in dataset:
        x = torch.tensor(x)
        logits = model(x)
        probs = torch.nn.functional.softmax(logits)
        pred = torch.argmax(probs, dim=0)
        pred = pred.detach().numpy()
        correct += (pred == labels[j])
        j+=1
    
    return correct/tot


def prepare_data_loader(X, y):
    '''
    Prepares the data loader and returns it.
    Input: - X: the data
           - y: the labels of the data
    Returns: a DataLoader object ready for training and one for testing and the total number of training data.
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    total_samples = len(train_loader) * train_loader.batch_size

    return train_loader, test_loader, total_samples



def normalize_vector(vec, normalization_type='min_max'):
    '''
    Takes input vec: np.array: a vector and normalize it's values in the range [0,1]. 
    Inputs: - vec: np.array vector. The vector to normalize/
            - normalization_type: 'min_max' for min-max normalization (i.e (x_i - x_min)/(x_max - x_min) for all i) 
            or 'vector' for vector normalization (i.e u/||u||)
    Returns the normalized vector
    '''
    if normalization_type == 'min_max':
        vmin = np.min(vec)
        vmax = np.max(vec)
        return (vec-vmin)/(vmax - vmin)
    elif normalization_type == 'vector':
        return vec/np.linalg.norm(vec)
    else:
        raise ValueError("Invalid 'normalization_type' type. Use 'min_max' or 'vector'.")



def generate_data(M=2, N=100, d=2, sigma=0.1, mean_r=1, means="phasor"):
    """
    Generate random data.

    Args:
        M: number of classes
        N: Number of samples per class
        d: the dimension integer
        sigma: standard deviation of Gaussian noise
        mean_r: the redius of the mean of each class, no effect with "random" means
        means: "random" to generate means of the classes randomly with high variance
            or "phasor" to generate them on a circle

    Returns:
        data: samples (array of shape (M*N))
        labels: class indices (array of shape (M*N))
        means_vecs: a list of vectro each being the mean of the corresponding class
    """
    data = []
    labels = []
    means_vecs = []

    if means == "phasor" and d==2:
        for k in range(M):
            mean = mean_r * np.exp(1j * k * 2 * np.pi / M)
            means_vecs.append(mean)
            d = mean + sigma * (np.random.randn(N) + 1j * np.random.randn(N))
            # stack into (N, 2)
            points = np.column_stack((d.real, d.imag))
            data.append(points)
            labels.append(np.full(N, k))
        # concatenate into (M*N, 2) and (M*N,)
        data = np.vstack(data)
        labels = np.concatenate(labels)
        return data, labels 
    elif means == "random" and d==2:
        for k in range(M):
            mean = (np.random.randn() + 1j * np.random.randn()) * 2
            means_vecs.append(mean)
            d = mean + sigma * (np.random.rand(N) + 1j * np.random.randn(N))
            # stack into (N, 2)
            points = np.column_stack((d.real, d.imag))
            data.append(points)
            labels.append(np.full(N, k))
        # concatenate into (M*N, 2) and (M*N,)
        data = np.vstack(data)
        labels = np.concatenate(labels)
    elif d > 2:
        if means == "phasor":
            # Place means evenly on a hypersphere
            for k in range(M):
                vec = np.random.randn(d)  # random direction
                vec /= np.linalg.norm(vec)  # normalize
                means_vecs.append(mean_r * vec)
        elif means == "random":
            # Random means with higher variance
            means_vecs = [np.random.randn(d) * 2 for _ in range(M)]
        else:
            raise ValueError("Invalid 'means' type. Use 'phasor' or 'random'.")

        # Generate samples around each mean
        means_ret = []
        for k in range(M):
            mean = means_vecs[k]
            points = mean + sigma * np.random.randn(N, d)
            data.append(points)
            labels.append(np.full(N, k))

        data = np.vstack(data)
        labels = np.concatenate(labels)

    else:
        raise ValueError("Unsupported configuration: check d and means.")
    
    return data, labels, means_vecs

def angle_between(u, v, eps=1e-7):
    u = np.array(u, dtype=np.float64)
    v = np.array(v, dtype=np.float64)

    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    if norm_u < eps and norm_v < eps:
        return 0.0, 0.0
    elif norm_u < eps or norm_v < eps:
        return np.pi / 2, 90.0

    cos_theta = np.dot(u, v) / (norm_u * norm_v)

    # Numerical safety: clip before arccos
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    return angle_rad, angle_deg


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


def prepare_mnist(n, model_path = "./data/models/model_weights_LR_mnist.pth"):
    '''
    Loads and prepares the MNIST dataset. Takes as input the number of training points n and return X, y, means. 
    '''

    # Define transform: convert to tensor [0,1] and flatten to 784-dim vector
    transform = transforms.Compose([
        transforms.ToTensor(),                      # -> (1, 28, 28), values in [0,1]
        transforms.Lambda(lambda x: x.view(-1))     # -> (784,)
    ])

    # Load the training dataset
    train_dataset = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)

    # Load the test dataset
    test_dataset = datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)
    # Pick a smaller subset of training data (e.g. 5000 samples)
    subset_size = n
    indices = torch.randperm(len(train_dataset))[:subset_size]  # random subset
    train_dataset = Subset(train_dataset, indices)

    # Keep test dataset small as well if needed (e.g. 1000 samples)
    # Probably irrelevant.
    #test_subset_size = n/10
    #test_indices = torch.randperm(len(test_dataset))[:test_subset_size]
    #test_dataset = Subset(test_dataset, test_indices)

    # Convert subset to arrays X, y
    X = []
    y = []
    for i in range(len(train_dataset)):
        data, label = train_dataset[i]
        X.append(data.numpy())   # convert tensor -> numpy
        y.append(label)

    X = np.stack(X)   # shape (n, 784)

    y = np.array(y)   # shape (n,)

    means = []
    M=10
    for c in range(M):
        class_samples = X[y == c]
        class_mean = np.mean(class_samples, axis=0)
        means.append(class_mean)

    means = np.array(means)   # shape (10, 784)

    return X, y, means 
