import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
import math
from utils import angle_between, load_orl_faces, generate_data
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from model import SoftmaxRegression, MLPClassifier, train_model
from attack import reconstruct_face, save_reconstruction
import scipy.io as sio
import argparse




def main():
    # Parameters
    d = 50
    M = 10
    N = 50
    sigma = 0.1
    r = 1
    means="phasor"
    reg_l2_1 = 0.005
    reg_l2_2 = 0.01
    reg_l2_3 = 0.1
    thres = 1e-6
    iters = 1000
    verbocity=False
    plt.rcParams.update({'font.size': 20})

    # Command line arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mnist", action='store_true', help="Perform experiments using the MNIST dataset.")
    parser.add_argument("-o", "--orl", action='store_true', help="Perform experiments using the ORL dataset of faces.")
    parser.add_argument("-s","--simple", action='store_true', help="Performs simple 2d experiments for different lambdas (i.e. regularization weights)")
    parser.add_argument("-w","--matlab", action='store_true', help="Output the weights from training to use them.")
    parser.add_argument("-n","--mlp", action='store_true', help="Use MLP model for the orl dataset as described in the paper.")
    args = parser.parse_args()

    if args.mlp:
        dataset_path = "data/orl"  
        results_dir = "results/attack_results"
        reg_lambda = 1e-4

        # Load dataset 
        X, y, avg_faces = load_orl_faces(dataset_path)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.long))
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

        
        d, M = 92 * 112, 40
        
        model = MLPClassifier(input_dim=d, num_classes=M)
        train_model(model, train_loader, test_loader, verbose=verbocity, epochs=iters, lr=0.1, weight_decay=reg_lambda, reg_lambda=0)
        
        torch.save(model.state_dict(), f'data/models/model_weights_lambda_{reg_lambda}.pth')
        
        #model.load_state_dict(torch.load(f'./data/models/model_weights_lambda_{reg_lambda}.pth'))
        #model.eval()
        #logits = model(torch.tensor(X[0]))
        #probs = torch.nn.functional.softmax(logits, dim=0)
        #pred = torch.argmax(probs, dim=0)
        #print(probs)
        #print(pred)
        #target_label = 1  # example
        #rec_img, best_conf, loss_list = reconstruct_face(model, target_label,
        #                                          H=112, W=92,
        #                                          alpha=100000, step_size=0.1, init_mode="noise",
        #                                          gamma=0.99, beta=5000, tv_weight=0, rounding_precision=0.0,
        #                                                 device="cpu", show_cost=True, noise_in_weights=False, snr=1e3, interactive=False, reg_lambda2_weight=0, thres=1e-5)
        
        #rec_img2, best_conf2, loss_list2 = reconstruct_face(model, target_label,
        #                                          H=112, W=92,
        #                                          alpha=100000, step_size=0.1, init_mode="zeros",
        #                                          gamma=0.99, beta=5000, tv_weight=0, rounding_precision=0.0,
        #                                                 device="cpu", show_cost=True, noise_in_weights=False, snr=1e3, interactive=False, reg_lambda2_weight=0, thres=1e-5)
        
        #print(rec_img)
        #print(rec_img2)
        
        #rec_img = rec_img.detach().numpy()
        #rec_img2 = rec_img2.detach().numpy()
        #plt.plot(rec_img)
        #plt.plot(rec_img2)
        #plt.show()

        #rec_img_2d = rec_img.view(1, 1, 112, 92)
        #save_reconstruction(rec_img_2d, best_conf, target_label, results_dir="results/matlab_validation", mult_factor=1, rescale=False)

       

        return

    if args.matlab:
        dataset_path = "data/orl"  
        results_dir = "results/attack_results"

        # Load dataset 
        X, y, avg_faces = load_orl_faces(dataset_path)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.long))
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

        
        d, M = 92 * 112, 40
        
        print(f"Train model with reg_l2 = {reg_l2_1} ...")
        model = SoftmaxRegression(input_dim=d, num_classes=M)
        train_model(model, train_loader, test_loader, verbose=verbocity, epochs=iters, lr=0.1, weight_decay=0.001, reg_lambda=reg_l2_1)

        params = list(model.parameters())

        sio.savemat('weights.mat', dict(w=params[0].detach().numpy()))

        target_label = 1  # example
        rec_img, best_conf, loss_list = reconstruct_face(model, target_label,
                                                  H=112, W=92,
                                                  alpha=100000, step_size=0.1, init_mode="zeros",
                                                  gamma=0.99, beta=50000, tv_weight=0, rounding_precision=0.0,
                                                         device="cpu", show_cost=True, noise_in_weights=False, snr=1e3, interactive=False, reg_lambda2_weight = 0.01)
        
        rec_img_2d = rec_img.view(1, 1, 112, 92)

        save_reconstruction(rec_img_2d, best_conf, target_label, results_dir="results/matlab_validation", mult_factor=1, rescale=False)
        
        sio.savemat('weights.mat', dict(w=params[0].detach().numpy(), x=rec_img))

        return


    if args.simple:
        d, M = 2, 5
        data, labels = generate_data(M=M, N=N, d=d, sigma=sigma, mean_r=r, means=means)
        #data_mat = sio.loadmat('../M_class_LR/test_data.mat')
        #data = np.array(data_mat['X'].T)
        #labels = np.array(data_mat['y'])
        #theta_grad = np.array(data_mat['theta'])
        #labels = labels.reshape(100,)-1
        print(f"X shape = {data.shape}, y shape = {labels.shape}")

        #X_train, y_train = data, labels
        #X_test, y_test = torch.tensor([], dtype=torch.float32), torch.tensor([], dtype=torch.float32)
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                          torch.tensor(y_train, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                         torch.tensor(y_test, dtype=torch.long))
        
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)
        
        
        print(f"Train model with reg_l2 = {reg_l2_1} ...")
        model = SoftmaxRegression(input_dim=d, num_classes=M)
        train_model(model, train_loader, test_loader, verbose=verbocity, epochs=iters, lr=0.1, weight_decay=reg_l2_1, reg_lambda=reg_l2_1, thres=thres)
        
        params = list(model.parameters())
        W = params[0]
        W1 = W.detach().cpu().numpy()
        
        print(f"Train model with reg_l2 = {reg_l2_2} ...")
        model = SoftmaxRegression(input_dim=d, num_classes=M)
        #model.linear.weight.data.fill_(0)
        #model.linear.bias.data.fill_(0) 
        train_model(model, train_loader, test_loader, verbose=verbocity, epochs=iters, lr=0.1, weight_decay=reg_l2_2, reg_lambda=reg_l2_2, thres=thres)
        
        params = list(model.parameters())
        W = params[0]
        W2 = W.detach().cpu().numpy()

        
        print(f"Train model with reg_l2 = {reg_l2_3} ...")
        model = SoftmaxRegression(input_dim=d, num_classes=M)
        train_model(model, train_loader, test_loader, verbose=verbocity, epochs=iters, lr=0.1, weight_decay=reg_l2_3, reg_lambda=reg_l2_3, thres=thres)
        
        
        params = list(model.parameters())
        W = params[0]
        W3 = W.detach().cpu().numpy()

        plt.subplot(1, 3, 1)
        for k in range(M):
            class_points = data[labels == k]   # shape (N, 2) for class k
            plt.scatter(class_points[:, 0], class_points[:, 1], label=f"class {k}", alpha=0.6)
        
        #plt.legend()
        plt.grid(True)
        plt.gca().set_aspect("equal")  # keep circle shape
        #plt.show()
        
        for k in range(M):
            plt.arrow(0,0,W1[k][0], W1[k][1], width=0.025, label=f"W_{k}")
        
        plt.title(r'$\lambda=$' + f'{reg_l2_1}')
        
        plt.subplot(1, 3, 2)
        for k in range(M):
            class_points = data[labels == k]   # shape (N, 2) for class k
            plt.scatter(class_points[:, 0], class_points[:, 1], label=f"class {k}", alpha=0.6)
        
        #plt.legend()
        plt.grid(True)
        plt.gca().set_aspect("equal")  # keep circle shape
        #plt.show()
        
        for k in range(M):
            plt.arrow(0,0,W2[k][0], W2[k][1], width=0.025, label=f"W_{k}")
        
        plt.title(r'$\lambda=$' + f'{reg_l2_2}')
        
        plt.subplot(1, 3, 3)
        for k in range(M):
            class_points = data[labels == k]   # shape (N, 2) for class k
            plt.scatter(class_points[:, 0], class_points[:, 1], label=f"class {k}", alpha=0.6)
        
        #plt.legend()
        plt.grid(True)
        plt.gca().set_aspect("equal")  # keep circle shape
        #plt.show()
        
        for k in range(M):
            plt.arrow(0,0,W3[k][0], W3[k][1], width=0.025, label=f"W_{k}")
        
        plt.title(r'$\lambda=$' + f'{reg_l2_3}')

        plt.show()
        
        return
        
    if args.orl:
        dataset_path = "data/orl"  
        results_dir = "results/attack_results"

        # Load dataset 
        X, y, avg_faces = load_orl_faces(dataset_path)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.long))
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

        
        d, M = 92 * 112, 40
        
        print(f"Train model with reg_l2 = {reg_l2_1} ...")
        model = SoftmaxRegression(input_dim=d, num_classes=M)
        train_model(model, train_loader, test_loader, verbose=verbocity, epochs=iters, lr=0.1, weight_decay=reg_l2_1, reg_lambda=reg_l2_1)
        
        params = list(model.parameters())
        W = params[0]
        W1 = W.detach().cpu().numpy()
        
        print(f"Train model with reg_l2 = {reg_l2_2} ...")
        model = SoftmaxRegression(input_dim=d, num_classes=M)
        #model.linear.weight.data.fill_(0)
        #model.linear.bias.data.fill_(0) 
        train_model(model, train_loader, test_loader, verbose=verbocity, epochs=iters, lr=0.1, weight_decay=reg_l2_2, reg_lambda=reg_l2_2)
        
        params = list(model.parameters())
        W = params[0]
        W2 = W.detach().cpu().numpy()

        
        print(f"Train model with reg_l2 = {reg_l2_3} ...")
        model = SoftmaxRegression(input_dim=d, num_classes=M)
        train_model(model, train_loader, test_loader, verbose=verbocity, epochs=iters, lr=0.1, weight_decay=reg_l2_3, reg_lambda=reg_l2_3)
        
        
        params = list(model.parameters())
        W = params[0]
        W3 = W.detach().cpu().numpy()

        # Compute norms, ratios and angles
        angle12 = []
        angle23 = []
        angle13 = []
        w_norm_1 = []
        w_norm_2 = []
        w_norm_3 = []
        ratio12 = []
        ratio23 = []
        ratio13 = []
        for k in range(M):
            ang, _ = angle_between(W1[k], W2[k])
            angle12.append(math.sin(ang))
            ang, _ = angle_between(W2[k], W3[k])
            angle23.append(math.sin(ang))
            ang, _ = angle_between(W1[k], W3[k])
            angle13.append(math.sin(ang))
            
            w_norm_1.append(np.linalg.norm(W1[k]))
            w_norm_2.append(np.linalg.norm(W2[k]))
            w_norm_3.append(np.linalg.norm(W3[k]))
            
            ratio12.append(w_norm_1[k]/w_norm_2[k])
            ratio23.append(w_norm_2[k]/w_norm_3[k])
            ratio13.append(w_norm_1[k]/w_norm_3[k])
        
        if d == 2:
            plt.subplot(1, 4, 1)
            for k in range(M):
                class_points = data[labels == k]   # shape (N, 2) for class k
                plt.scatter(class_points[:, 0], class_points[:, 1], label=f"class {k}", alpha=0.6)
        
            #plt.legend()
            plt.grid(True)
            plt.gca().set_aspect("equal")  # keep circle shape
            #plt.show()
        
            for k in range(M):
                plt.arrow(0,0,W2[k][0], W2[k][1], width=0.05, label=f"W_{k}")
        
            plt.title(f"Multiclass classification via logistic regression on random data\n M={M}, N={N}, sigma={sigma}, reg_l2={reg_l2_2}")
        
        n_plots = 2 + (d==2)
        i_plot = (d==2)
        plt.subplot(1,n_plots,i_plot+1)
        plt.scatter(range(len(angle12)), angle12, label=fr"$\lambda_1$={reg_l2_1} and $\lambda_2$={reg_l2_2}")
        plt.scatter(range(len(angle23)), angle23, label=fr"$\lambda_1$={reg_l2_2} and $\lambda_2$={reg_l2_3}")
        plt.scatter(range(len(angle13)), angle13, label=fr"$\lambda_1$={reg_l2_1} and $\lambda_2$={reg_l2_3}")
        plt.title("Sine of angles between different weights")
        plt.grid(True)
        plt.xlabel("Class")
        plt.ylabel(r'Sin($\hat{W_{\lambda_1}}, \hat{W_{\lambda_2}}$)')
        plt.legend()
        plt.ylim(-1,1)

        plt.subplot(1, n_plots, i_plot+2)
        plt.scatter(range(len(ratio12)), ratio12, label=fr"$\lambda_1$={reg_l2_1} and $\lambda_2$={reg_l2_2}")
        plt.scatter(range(len(ratio23)), ratio23, label=fr"$\lambda_1$={reg_l2_2} and $\lambda_2$={reg_l2_3}")
        plt.scatter(range(len(ratio13)), ratio13, label=fr"$\lambda_1$={reg_l2_1} and $\lambda_2$={reg_l2_3}" )
        plt.title("Ratio of norms of weights")
        plt.grid(True)
        plt.xlabel("Class")
        plt.ylabel(r'$\|W_{\lambda_1}\|/\|W_{\lambda_2}\|$')
        plt.legend()
        

        plt.show()

        return 
        
    if args.mnist:
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
        subset_size = 1000
        indices = torch.randperm(len(train_dataset))[:subset_size]  # random subset
        train_dataset = Subset(train_dataset, indices)

        # Keep test dataset small as well if needed (e.g. 1000 samples)
        test_subset_size = 200
        test_indices = torch.randperm(len(test_dataset))[:test_subset_size]
        test_dataset = Subset(test_dataset, test_indices)
        

        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

        
        d = 28*28
        M = 10
         
        print(f"Train model with reg_l2 = {reg_l2_1} ...")
        model = SoftmaxRegression(input_dim=d, num_classes=M)
        train_model(model, train_loader, test_loader, verbose=verbocity, epochs=iters, lr=0.1, weight_decay=reg_l2_1, reg_lambda=reg_l2_1)
        
        params = list(model.parameters())
        W = params[0]
        W1 = W.detach().cpu().numpy()
        
        print(f"Train model with reg_l2 = {reg_l2_2} ...")
        model = SoftmaxRegression(input_dim=d, num_classes=M)
        #model.linear.weight.data.fill_(0)
        #model.linear.bias.data.fill_(0) 
        train_model(model, train_loader, test_loader, verbose=verbocity, epochs=iters, lr=0.1, weight_decay=reg_l2_2, reg_lambda=reg_l2_2)
        
        params = list(model.parameters())
        W = params[0]
        W2 = W.detach().cpu().numpy()

        
        print(f"Train model with reg_l2 = {reg_l2_3} ...")
        model = SoftmaxRegression(input_dim=d, num_classes=M)
        train_model(model, train_loader, test_loader, verbose=verbocity, epochs=iters, lr=0.1, weight_decay=reg_l2_3, reg_lambda=reg_l2_3)
        
        
        params = list(model.parameters())
        W = params[0]
        W3 = W.detach().cpu().numpy()

        # Compute norms, ratios and angles
        angle12 = []
        angle23 = []
        angle13 = []
        w_norm_1 = []
        w_norm_2 = []
        w_norm_3 = []
        ratio12 = []
        ratio23 = []
        ratio13 = []
        for k in range(M):
            ang, _ = angle_between(W1[k], W2[k])
            angle12.append(math.sin(ang))
            ang, _ = angle_between(W2[k], W3[k])
            angle23.append(math.sin(ang))
            ang, _ = angle_between(W1[k], W3[k])
            angle13.append(math.sin(ang))
            
            w_norm_1.append(np.linalg.norm(W1[k]))
            w_norm_2.append(np.linalg.norm(W2[k]))
            w_norm_3.append(np.linalg.norm(W3[k]))
            
            ratio12.append(w_norm_1[k]/w_norm_2[k])
            ratio23.append(w_norm_2[k]/w_norm_3[k])
            ratio13.append(w_norm_1[k]/w_norm_3[k])
        
        if d == 2:
            plt.subplot(1, 4, 1)
            for k in range(M):
                class_points = data[labels == k]   # shape (N, 2) for class k
                plt.scatter(class_points[:, 0], class_points[:, 1], label=f"class {k}", alpha=0.6)
        
            #plt.legend()
            plt.grid(True)
            plt.gca().set_aspect("equal")  # keep circle shape
            #plt.show()
        
            for k in range(M):
                plt.arrow(0,0,W2[k][0], W2[k][1], width=0.05, label=f"W_{k}")
        
            plt.title(f"Multiclass classification via logistic regression on random data\n M={M}, N={N}, sigma={sigma}, reg_l2={reg_l2_2}")
        
        n_plots = 2 + (d==2)
        i_plot = (d==2)
        plt.subplot(1,n_plots,i_plot+1)
        plt.scatter(range(len(angle12)), angle12, label=fr"$\lambda_1$={reg_l2_1} and $\lambda_2$={reg_l2_2}")
        plt.scatter(range(len(angle23)), angle23, label=fr"$\lambda_1$={reg_l2_2} and $\lambda_2$={reg_l2_3}")
        plt.scatter(range(len(angle13)), angle13, label=fr"$\lambda_1$={reg_l2_1} and $\lambda_2$={reg_l2_3}")
        plt.title("Sine of angles between different weights")
        plt.grid(True)
        plt.xlabel("Class")
        plt.ylabel(r'Sin($\hat{W_{\lambda_1}}, \hat{W_{\lambda_2}}$)')
        plt.legend()
        plt.ylim(-1,1)

        plt.subplot(1, n_plots, i_plot+2)
        plt.scatter(range(len(ratio12)), ratio12, label=fr"$\lambda_1$={reg_l2_1} and $\lambda_2$={reg_l2_2}")
        plt.scatter(range(len(ratio23)), ratio23, label=fr"$\lambda_1$={reg_l2_2} and $\lambda_2$={reg_l2_3}")
        plt.scatter(range(len(ratio13)), ratio13, label=fr"$\lambda_1$={reg_l2_1} and $\lambda_2$={reg_l2_3}" )
        plt.title("Ratio of norms of weights")
        plt.grid(True)
        plt.xlabel("Class")
        plt.ylabel(r'$\|W_{\lambda_1}\|/ \|W_{\lambda_2}\|$')
        plt.legend()
        

        plt.show()

        return
        
    else:
        data, labels, _ = generate_data(M=M, N=N, d=d, sigma=sigma, mean_r=r, means=means)
        #data_mat = sio.loadmat('../M_class_LR/test_data.mat')
        #data = np.array(data_mat['X'].T)
        #labels = np.array(data_mat['y'])
        #theta_grad = np.array(data_mat['theta'])
        #labels = labels.reshape(100,)-1
        print(f"X shape = {data.shape}, y shape = {labels.shape}")

        #X_train, y_train = data, labels
        #X_test, y_test = torch.tensor([], dtype=torch.float32), torch.tensor([], dtype=torch.float32)
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                          torch.tensor(y_train, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                         torch.tensor(y_test, dtype=torch.long))
        
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)
        
        
        print(f"Train model with reg_l2 = {reg_l2_1} ...")
        model = SoftmaxRegression(input_dim=d, num_classes=M)
        train_model(model, train_loader, test_loader, verbose=verbocity, epochs=iters, lr=0.1, weight_decay=reg_l2_1, reg_lambda=reg_l2_1)
        
        params = list(model.parameters())
        W = params[0]
        W1 = W.detach().cpu().numpy()
        
        print(f"Train model with reg_l2 = {reg_l2_2} ...")
        model = SoftmaxRegression(input_dim=d, num_classes=M)
        #model.linear.weight.data.fill_(0)
        #model.linear.bias.data.fill_(0) 
        train_model(model, train_loader, test_loader, verbose=verbocity, epochs=iters, lr=0.1, weight_decay=reg_l2_2, reg_lambda=reg_l2_2)
        
        params = list(model.parameters())
        W = params[0]
        W2 = W.detach().cpu().numpy()

        
        print(f"Train model with reg_l2 = {reg_l2_3} ...")
        model = SoftmaxRegression(input_dim=d, num_classes=M)
        train_model(model, train_loader, test_loader, verbose=verbocity, epochs=iters, lr=0.1, weight_decay=reg_l2_3, reg_lambda=reg_l2_3)
        
        
        params = list(model.parameters())
        W = params[0]
        W3 = W.detach().cpu().numpy()

        # Compute norms, ratios and angles
        angle12 = []
        angle23 = []
        angle13 = []
        w_norm_1 = []
        w_norm_2 = []
        w_norm_3 = []
        ratio12 = []
        ratio23 = []
        ratio13 = []
        for k in range(M):
            ang, _ = angle_between(W1[k], W2[k])
            angle12.append(math.sin(ang))
            ang, _ = angle_between(W2[k], W3[k])
            angle23.append(math.sin(ang))
            ang, _ = angle_between(W1[k], W3[k])
            angle13.append(math.sin(ang))
            
            w_norm_1.append(np.linalg.norm(W1[k]))
            w_norm_2.append(np.linalg.norm(W2[k]))
            w_norm_3.append(np.linalg.norm(W3[k]))
            
            ratio12.append(w_norm_1[k]/w_norm_2[k])
            ratio23.append(w_norm_2[k]/w_norm_3[k])
            ratio13.append(w_norm_1[k]/w_norm_3[k])
       
        if d == 2:
            plt.subplot(1, 4, 1)
            for k in range(M):
                class_points = data[labels == k]   # shape (N, 2) for class k
                plt.scatter(class_points[:, 0], class_points[:, 1], label=f"class {k}", alpha=0.6)
        
            #plt.legend()
            plt.grid(True)
            plt.gca().set_aspect("equal")  # keep circle shape
            #plt.show()
        
            for k in range(M):
                plt.arrow(0,0,W2[k][0], W2[k][1], width=0.05, label=f"W_{k}")
        
            plt.title(f"Multiclass classification via logistic regression on random data\n M={M}, N={N}, sigma={sigma}, reg_l2={reg_l2_2}")
        
        n_plots = 2 + (d==2)
        i_plot = (d==2)
        plt.subplot(1,n_plots,i_plot+1)
        plt.scatter(range(len(angle12)), angle12, label=fr"$\lambda_1$={reg_l2_1} and $\lambda_2$={reg_l2_2}")
        plt.scatter(range(len(angle23)), angle23, label=fr"$\lambda_1$={reg_l2_2} and $\lambda_2$={reg_l2_3}")
        plt.scatter(range(len(angle13)), angle13, label=fr"$\lambda_1$={reg_l2_1} and $\lambda_2$={reg_l2_3}")
        plt.title("Sine of angles between different weights")
        plt.grid(True)
        plt.xlabel("Class")
        plt.ylabel(r'Sin($\hat{W_{\lambda_1}}, \hat{W_{\lambda_2}}$)')
        plt.legend()
        plt.ylim(-1,1)

        plt.subplot(1, n_plots, i_plot+2)
        plt.scatter(range(len(ratio12)), ratio12, label=fr"$\lambda_1$={reg_l2_1} and $\lambda_2$={reg_l2_2}")
        plt.scatter(range(len(ratio23)), ratio23, label=fr"$\lambda_1$={reg_l2_2} and $\lambda_2$={reg_l2_3}")
        plt.scatter(range(len(ratio13)), ratio13, label=fr"$\lambda_1$={reg_l2_1} and $\lambda_2$={reg_l2_3}" )
        plt.title("Ratio of norms of weights")
        plt.grid(True)
        plt.xlabel("Class")
        plt.ylabel(r'$\|W_{\lambda_1}\|/ \|W_{\lambda_2}\|$')
        plt.legend()

        plt.show()


        
        #print(W2.flatten())
        #plt.figure()
        #plt.plot(W2.flatten(), label='My solution')
        #plt.plot(theta_grad, label='GD')
        #plt.legend()
        #plt.show()


if __name__ == '__main__':
    main()
