import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from utils import *
from model import SoftmaxRegression, train_model_grad
from attack import reconstruct_face, save_reconstruction, noisy_weights
import math
import sys
import scipy.io

def main():
    # Setup parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--mnist", action='store_true',
                        help="Do the experiments on mnist dataset.")
    parser.add_argument("-v", "--verbose", action='store_true',
                        help="Verbosity True. Default is False")
    args = parser.parse_args()

    epsilon_p_arr = np.load('results/dp_lr/data/epsilon_p_arr.npy') 
    #epsilon_p_arr = [0.1,0.5, 1, 1.5, 2]
    #epsilon_p_arr = [ 1]
    


    if args.mnist:
        d, M = 28*28, 10
        H, W = 28, 28
        model_path = "./data/models/model_weights_LR_mnist.pth"
        
        X, y, means = prepare_mnist(2000)
        # Normalize so ||X||=1
        norms = np.linalg.norm(X, axis=1, keepdims=True)  # shape (n, 1)
        X = X/norms
        mean_norms = np.linalg.norm(means, axis=1, keepdims=True)  # shape (n, 1)  
        means = means / mean_norms

        train_loader, test_loader, total_training_samples = prepare_data_loader(X, y)


    model = SoftmaxRegression(input_dim=d, num_classes=M)
    angles_grad = []
    acc_grad = []

    for epsilon in epsilon_p_arr:
        train_model_grad(model, train_loader, test_loader, epochs=2000, weight_decay=1e-3, epsilon=epsilon)
        acc_tmp = 0
        angle_iter = 0
        avg_iters = 200
        for i in range(avg_iters):
            accuracy = calculate_accuracy(model, X.astype(np.float32), y)  # calculate accuracy
            acc_tmp += accuracy
            ang_l=0
            for target_label in range(M):
                rec_img, best_conf, loss_list, _ = reconstruct_face(model, target_label, test_loader,
                                                                H=H, W=W,
                                                                alpha=100000, step_size=0.1, init_mode="weights",
                                                                gamma=0.99, beta=5000, tv_weight=0.0, rounding_precision=0.0,
                                                                device="cpu", show_cost=False, noise_in_weights=False,
                                                                snr=1e3, interactive=False, weight_decay=1e-2, reg_lambda2_weight=1, thres=1e-5,
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
        angles_grad.append(angle_iter/avg_iters)

        acc_grad.append(acc_tmp/avg_iters)
        print(f'epislon: {epsilon}, Model Accuracy: {acc_grad[-1]}, Rec. angle: {angles_grad[-1]}')


    plt.rcParams.update({'font.size': 20})
    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 9)
    color = 'tab:red'
    ax1.set_xlabel(r'$\epsilon_p$')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.semilogx(epsilon_p_arr, acc_grad, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel(r'$Sine(\hat{x}, \mu)$', color=color)
    ax2.semilogx(epsilon_p_arr, angles_grad, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title(fr"Angle and accuracy for different $\epsilon_p$")

    plt.show()

    np.save('results/dp_lr/data/acc_grad.npy', np.array(acc_grad))
    np.save('results/dp_lr/data/angles_grad.npy', np.array(angles_grad))


    return









if __name__ == '__main__':
    main()
