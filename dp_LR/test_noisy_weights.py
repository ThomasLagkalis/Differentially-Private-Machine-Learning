"""
    This driver code for the experiments on noisy weights and attack validation.
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
from utils import *
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from model import SoftmaxRegression, MLPClassifier, train_model
from attack import reconstruct_face, save_reconstruction, noisy_weights
import scipy.io as sio
import argparse
import math
import sys

# Helper functions
def generate_noisy_models(input_dim, num_classes, snr_arr, model_path, noise_type='laplace', verbose=False):
    '''
    Generate noisy models by adding noise to their weights.
    Inputs: - input_dim: the input dimension of all the models
            - num_classes: number of classes of the classification problem.
            - snr_arr: the array containing snr values for the additive noise. None value means no noise to be added.
            - model_path: the path of the base model.
    Returns: array of length len(snr_arr) containing the models and array length len(snr_arr) 
             containing the std of the additive noise for each SNR value.
    '''
    models = [SoftmaxRegression(input_dim=input_dim, num_classes=num_classes) for _ in snr_arr]
    std = []
    i = 0
    for snr in snr_arr:
        state_dict = torch.load(model_path)
        models[i].load_state_dict(state_dict)
        if snr != None:
            if noise_type=='gaussian':
                models[i], s = noisy_weights(models[i], snr, verbose=verbose)
            else:
                models[i], s = noisy_weights(models[i], snr, noise_type=noise_type, verbose=verbose) 
            std.append(s)
        else: 
            std.append(0)
        models[i].eval()
        i+=1

    return models, std




###################### Main Function ########################3
def main():
    # Setup parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--acc", action='store_true',
                        help="Perform prediction in the whole dataset and mesure the accuracy.")
    parser.add_argument("--plot_weights", action='store_true',
                        help="Plot the reshaped weights of the models for each snr and save them infile ./results/noisy_weights/noisy_weights_faces")
    parser.add_argument("--synthetic_data", action='store_true',
                        help="Do the experiments on synthetic (random) data generated around random means with high variance. Default is the ORL faces dataset.")
    parser.add_argument("--mnist", action='store_true',
                        help="Di the experiments on mnist dataset.")
    parser.add_argument("-v", "--verbose", action='store_true', help="Verbosity True. Default is False")
    args = parser.parse_args()

    #plt.rcParams.update({'font.size': 16})
    dataset_path = "data/orl" 
    results_dir = "results/attack_results"
    reg_lambda = 0.001
    snr_arr = [None]
    #snr_db = [i for i in range(-40,31,2)]
    #snr_arr = [10**(db/10) for db in snr_db]
    verbocity=False
    if args.synthetic_data:
        M, N, d = 10, 10, 10
        H, W = d, 1
    elif args.mnist:
        d, M = 28*28, 10
        H, W = 28, 28
    else:
        d, M, N = 92 * 112, 40, 10
        H, W = 112, 92

    # Load dataset 
    if args.synthetic_data:
        model_path = "./data/models/model_weights_LR_synthetic_data.pth"
        X, y, means = generate_data(M=M, N=N, d=d, sigma=13, means="random")
        # --- Normalize X and means ---
        #x_min = X.min(axis=0)
        #x_max = X.max(axis=0)
        #X = (X - x_min) / (x_max - x_min)
        #means = [(mean - x_min) / (x_max - x_min) for mean in means]
        #means = [mean - np.mean(mean) for mean in means]
        #means = np.array(means)
        train_loader, test_loader, _ = prepare_data_loader(X, y)
    elif args.mnist:
        model_path = "./data/models/model_weights_LR_mnist.pth"
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
        subset_size = 2000
        indices = torch.randperm(len(train_dataset))[:subset_size]  # random subset
        train_dataset = Subset(train_dataset, indices)

        # Keep test dataset small as well if needed (e.g. 1000 samples)
        test_subset_size = 400
        test_indices = torch.randperm(len(test_dataset))[:test_subset_size]
        test_dataset = Subset(test_dataset, test_indices)

        # Convert subset to arrays X, y
        X = []
        y = []
        for i in range(len(test_dataset)):
            data, label = train_dataset[i]
            X.append(data.numpy())   # convert tensor -> numpy
            y.append(label)

        X = np.stack(X)   # shape (1000, 784)

        y = np.array(y)   # shape (1000,)

        means = []
        for c in range(M):
            class_samples = X[y == c]
            class_mean = np.mean(class_samples, axis=0)
            means.append(class_mean)

        means = np.array(means)   # shape (10, 784)
        

        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)
    else:
        model_path = "./data/models/model_weights_LR.pth"
        X, y, means = load_orl_faces(dataset_path)
        #means = [mean - np.mean(mean) for mean in means]
        train_loader, test_loader, _ = prepare_data_loader(X, y)
    

        
        
    model = SoftmaxRegression(input_dim=d, num_classes=M)
    train_model(model, train_loader, test_loader,
                verbose=verbocity, lr=0.1,
                weight_decay=1e-1, reg_lambda=1e-1, thres=1e-5)
    torch.save(model.state_dict(), model_path)
    
    models, std = generate_noisy_models(d, M, snr_arr, model_path, noise_type='laplace', verbose=args.verbose)
   

    #########################Display and save the (noisy) weights#########################
    if not args.synthetic_data:
        # Plot the weights
        if args.plot_weights:
            plt.rcParams.update({'font.size': 20})
            plt.figure(figsize=(20, 4))
            for label in range(M):
                plt.clf()
                i=0
                for snr in snr_arr:
                    params = list(models[i].parameters())
                    params = params[0].detach().numpy()
                    im_prm = params[label].reshape(H, W)
                    plt.subplot(1, 4, i+1)
                    plt.imshow(im_prm, cmap='gray')
                    plt.colorbar()
                    if snr != None:
                        plt.title(f'SNR={snr}')
                    else:
                        plt.title(fr'SNR=$\infty$')
                    plt.xticks([])
                    plt.yticks([]) 
                    i+=1
                print(f"Saving figure for label {label}...")
                if args.mnist:
                    plt.savefig(f'results/noisy_weights/noisy_weights/label{label}_mnist.png')
                else:
                    plt.savefig(f'results/noisy_weights/noisy_weights/label{label}.png')
            return

    ###################### Test the predictions #######################
    if args.acc:
        acc = []
        angles = []
        for i in range(len(snr_arr)):
            acc_tmp = 0
            for _ in range(50):
                m, _ = generate_noisy_models(d, M, [snr_arr[i]], model_path, noise_type='laplace')
                ml = m[0]
                accuracy = calculate_accuracy(ml, X.astype(np.float32), y)  # calculate accuracy
                acc_tmp += accuracy
            acc.append(acc_tmp/50)
            #print(f'Accuracy: {acc_tmp/100}')
            target_label = 1
            angle_iter = 0
            for _ in range(5):
                models, _ = generate_noisy_models(d, M, snr_arr, model_path, noise_type='laplace')
                model = models[i]
                ang_l =0
                for target_label in range(M):
                    if args.verbose:
                        print(f"SNR: {snr_arr[i]}, Reconstruction for label: {target_label}") 
                    # For ORL reg_lambda = 1e-4
                    # MNIST reg_lambda = 
                    rec_img, best_conf, loss_list, _ = reconstruct_face(model, target_label, test_loader,
                                                                    H=H, W=W,
                                                                    alpha=100000, step_size=0.1, init_mode="weights",
                                                                    gamma=0.99, beta=5000, tv_weight=0.0, rounding_precision=0.0,
                                                                    device="cpu", show_cost=False, noise_in_weights=False,
                                                                    snr=1e3, interactive=False, reg_lambda2_weight=1e-4, thres=1e-5,
                                                                    verbose=args.verbose
                                                                    )
                    rec_img = rec_img.detach().numpy()
                    # Normalize image to [0,1]
                    original_face = normalize_vector(means[target_label], normalization_type='min_max')
                    rec_img = normalize_vector(rec_img, normalization_type='min_max')
                    ang, _ = angle_between(rec_img, original_face)
                    if not math.isnan(ang):
                        ang_l += math.sin(ang)
                angle_iter += ang_l/M
            angles.append(angle_iter/5)

            # Status bar print
            #sys.stdout.write('\r')
            sys.stdout.write( "\n[%-20s] %d%%, SNR=%.6f, Accuracy=%.6f, Sine(ang)=%.6f" % ('=' * (i * 20 // len(snr_arr)), i * 100 // len(snr_arr), snr_arr[i], acc[-1], angles[-1]))
            sys.stdout.flush()
        
        plt.rcParams.update({'font.size': 18})
        fig, ax1 = plt.subplots()
        fig.set_size_inches(8, 5)
        color = 'tab:red'
        ax1.set_xlabel('SNR (db)')
        ax1.set_ylabel('Accuracy', color=color)
        ax1.plot(snr_db, acc, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        
        color = 'tab:blue'
        ax2.set_ylabel(r'$Sine(\hat{x}, \mu)$', color=color)
        ax2.plot(snr_db, angles, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        if args.synthetic_data:
            plt.title(f"Angle and accuracy plots for different SNR\nM={M},N={N},d={d}")
        else:
            plt.title(f"Angle and accuracy plots for different SNR")

        plt.show()
        input('Press Enter to continue...')
    
    ###########################################
    # MI-ATTACK
    ##########################################

    models, std = generate_noisy_models(d, M, snr_arr, model_path, noise_type='laplace')
    angles = []
    diff = []
    recons = []
    i=0
    for snr in snr_arr:
        model = models[i]
        rec_imgs = []
        angles_tmp = []
        diff_tmp = []
        for target_label in range(M):
            class_mean = means[target_label]
            print(f"SNR: {snr}, Reconstruction for label: {target_label}") 
            rec_img, best_conf, loss_list, acc = reconstruct_face(model, target_label, class_mean, test_loader,
                                                            H=H, W=W,
                                                            alpha=100000, step_size=0.1, init_mode="weights",
                                                            gamma=0.99, beta=5000, tv_weight=0.001, rounding_precision=0.0,
                                                            device="cpu", show_cost=False, noise_in_weights=False,
                                                            snr=1e3, interactive=False, reg_lambda2_weight=1e-5, weight_decay=1e-5, thres=1e-5,
                                                            verbose=args.verbose
                                                            )
            rec_img = rec_img.detach().numpy()
            # Normalize image to [0,1]
            original_face = normalize_vector(class_mean, normalization_type='min_max')
            #original_face = class_mean
            rec_img = normalize_vector(rec_img, normalization_type='min_max')
            rec_imgs.append(rec_img)
            ang, _ = angle_between(rec_img, original_face)
            angles_tmp.append(math.sin(ang))
            print(math.sin(ang))
            diff_tmp.append(np.linalg.norm(rec_img - original_face))
            params = list(model.parameters())
            params = params[0].detach().numpy()
            params = params[target_label]
            params = normalize_vector(params[target_label], normalization_type='min_max')
             
        angles.append(angles_tmp)
        diff.append(diff_tmp)
        recons.append(rec_imgs)
        i+=1


        # Plot the results
        plt.rcParams.update({'font.size': 20})
        plt.subplot(1, 2, 1)
        if snr != None:
            plt.plot(range(len(angles_tmp)), angles_tmp, label=f'SNR=1e{i+1}')
        else: 
            plt.plot(range(len(angles_tmp)), angles_tmp, label=r'SNR=$\infty$')
        plt.title("Angles")
        plt.xlabel("Label", fontsize=20)
        plt.ylabel(r"Sin($\mu, \hat{x}$)", fontsize=20)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        #plt.rcParams.update({'font.size': 20})
        if snr != None:
            plt.plot(range(len(diff_tmp)), diff_tmp, label=f'SNR=1e{i+1}')
        else:
            plt.plot(range(len(diff_tmp)), diff_tmp, label=r'SNR=$\infty$')

        plt.title("Norm of the difference")
        plt.xlabel("Label")
        plt.ylabel(r'$\|\mu - \hat{x}\|$')
        plt.legend()

    plt.show()

    # Display the reconstructions 
    if not args.synthetic_data:
        for label in range(M):
            plt.rcParams.update({'font.size': 15})
            plt.figure(figsize=(20, 4))
            plt.clf()
            for i in range(len(snr_arr)):
                plt.subplot(1,3,i+1)
                im = recons[i][label]
                #im = means[label]
                vmax = np.max(im)
                vmin = np.min(im)
                im = im.reshape(H,W)
                plt.imshow(im, cmap='grey', vmax=vmax, vmin=vmin)
                plt.colorbar()
                plt.xticks([])
                plt.yticks([]) 
                if snr_arr[i] != None:
                    plt.title(f"Reconstruction SNR={snr_arr[i]}")
                else:
                    plt.title(fr"Reconstruction SNR=$\infty$")
                plt.subplot(1,3,i+2)
                im = means[label]
                vmax = np.max(im)
                vmin = np.min(im)
                im = im.reshape(H,W)
                plt.imshow(im, cmap='grey', vmax=vmax, vmin=vmin)
                plt.title('Mean')
                plt.colorbar()
                plt.xticks([])
                plt.yticks([]) 
                plt.subplot(1,3,i+3)
                params = list(model.parameters())
                params = params[0].detach().numpy()
                im = params[label]
                vmax = np.max(im)
                vmin = np.min(im)
                im = im.reshape(H,W)
                plt.imshow(im, cmap='grey', vmax=vmax, vmin=vmin)
                plt.title('Weight')
                plt.colorbar()
                plt.xticks([])
                plt.yticks([]) 
            
            if args.mnist:
                plt.savefig(f"results/noisy_weights/reconstructions/reoconstruction_label{label}_mnist.png")
            else:
                plt.savefig(f"results/noisy_weights/reconstructions/reoconstruction_label{label}.png")
    
    return 

if __name__ == '__main__':
    main()
