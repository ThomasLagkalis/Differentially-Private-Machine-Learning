'''
This is the code for experiments regarding Differential Private Logistic Regression
using output and objective perturbation.
'''
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from utils import *
from model import SoftmaxRegression, train_model
from attack import reconstruct_face, save_reconstruction, noisy_weights
import math
import sys
import scipy.io

# Helper functions
def generate_dp_models(input_dim, num_classes, number_of_data, epsilon_private_arr, 
                       reg_lambda, model_path, verbose=False, dp_type="epsilon", delta=1e-5):
    '''
    Generate differantial private models by adding noise to their weights (output perturbation).
    Inputs: - input_dim: integer. the input dimension of all the models
            - num_classes: integer. number of classes of the classification problem.
            - number_of_data: integer. the total number of input training data.
            - epsilon_private_arr: a list containing privacy budget values.
            - delta: is the (epsion, delta)-DP parameter the same for all models.
            - reg_lambda: scalar. L2-Regularization coefficient.
            - model_path: string. the path of the base model.
            - dp_type: string. 'epsilon' for epsilon-DP, 'epsilon_delta' for (epdilon, delta)-DP
    Returns: array of length len(epsilon_private_arr) containing the models and array length len(epsilon_private_arr)
             containing the std of the additive noise for each epsilon value.
    '''
    models = [SoftmaxRegression(input_dim=input_dim, num_classes=num_classes) for _ in epsilon_private_arr]
    std = []
    i = 0
    for epsilon in epsilon_private_arr:
        if dp_type == "epsilon":
            state_dict = torch.load(model_path)
            models[i].load_state_dict(state_dict)
            params = list(models[i].parameters())
            W, b = params[0], params[1]  # weight (40 x 10304), bias (40,)
            W_np = W.detach().numpy().astype(np.float64)
            b_np = b.detach().numpy().astype(np.float64)

            # Calculate beta (scale) parameter of laplace distribution.
            #scale = (number_of_data * reg_lambda * epsilon)/(2*input_dim)
            beta = (number_of_data * reg_lambda * epsilon)/(2*np.sqrt(2))
            scale = 1/beta
            std.append(np.sqrt(2 * scale**2 ))
    
            #w_noise = np.random.laplace(0, scale, size=W_np.shape).astype(np.float64)
            #b_noise = np.random.laplace(0, scale, size=b.shape).astype(np.float64)
            w_noise = sample_euclid_exp(W_np.shape, beta).astype(np.float64)
            b_noise = sample_euclid_exp(b_np.shape, beta).astype(np.float64)
        elif dp_type == "epsilon-delta":
            state_dict = torch.load(model_path)
            models[i].load_state_dict(state_dict)
            params = list(models[i].parameters())
            W, b = params[0], params[1]  # weight (40 x 10304), bias (40,)
            W_np = W.detach().numpy().astype(np.float64)
            b_np = b.detach().numpy().astype(np.float64)
            
            sensitivity = 2/(number_of_data * reg_lambda)
            c = np.sqrt(2*np.log(1.25/delta)) + 1e-7
            sigma = c * sensitivity / epsilon
            
            w_noise = np.random.normal(0, sigma, size=W_np.shape)
            b_noise = np.random.normal(0, sigma, size=b_np.shape)

            

            
        # Add noise in model's parameters (output perturbation)
        with torch.no_grad():
            W.add_(torch.from_numpy(w_noise.astype(np.float32)))
            b.add_(torch.from_numpy(b_noise.astype(np.float32)))

        models[i].eval()
        i+=1

    return models, std


def main():
    # Setup parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--acc", action='store_true',
                        help="Perform prediction in the whole dataset and mesure the accuracy.")
    parser.add_argument("--synthetic_data", action='store_true',
                        help="Do the experiments on synthetic (random) data generated around random means with high variance. Default is the ORL faces dataset.")
    parser.add_argument("--mnist", action='store_true',
                        help="Do the experiments on mnist dataset.")
    parser.add_argument("-v", "--verbose", action='store_true',
                        help="Verbosity True. Default is False")
    parser.add_argument("-o", "--objective", action='store_true',
                        help="Do the experiments for objective perturbation. Default is output perturbation.")
    parser.add_argument("-c", "--compare", action='store_true',
                        help="Do the experiments for all perturbations with different values of epsilon_p and compare the results.")
    parser.add_argument("-n", "--compare_n", action='store_true',
                        help="Do the experiments for all perturbations, with a fixed epsilon_p value and different n values (i.e. number of training data) and compare the results. This experiment uses the MNIST dataset")
    parser.add_argument("-s", "--save_results", action='store_true',
                        help="Save the results of the experiments in ./results/pd_lr/data/")

    args = parser.parse_args()

    #plt.rcParams.update({'font.size': 16})
    dataset_path = "data/orl" 
    results_dir = "results/attack_results"
    #epsilon_p_arr = [1e-6, 1e-5, 1e-4, 1e-3, 1e-1, 1, 1e1, 1e2, 1e3]
    #epsilon_p_arr = [1e-1, 1, 1e1, 1e2] 
    #epsilon_p_arr = [1e8]
    arr_db = [i for i in range(0,35,2)]
    epsilon_p_arr = [10**(db/10) for db in arr_db]
    reg_lambda = 1e-3
    c = 0.25 # For objective perturbation.
    
    if args.save_results:
        np.save('results/dp_lr/data/epsilon_p_arr_new_distr.npy', np.array(epsilon_p_arr))
    
    ##########################################################################################
    # Do the experiments for fixed epsilon_p and different n values.
    ##########################################################################################
    if args.compare_n:
        model_path = "./data/models/model_weights_LR_mnist.pth"
        N = [i for i in range(2000, 40000, 4000)]
        #N = [10000]
        epsilon_p = 5
        acc_obj = []
        angles_obj = []
        acc_output = []
        angles_output = []
        acc_non_private = []
        angles_non_private = []
        
        for n in N:
            d, M = 28*28, 10
            H, W = 28, 28
            avg_iters = 20
            print("Beginning experiment for n=", n)
            
            X, y, means = prepare_mnist(n)
            # Normalize so ||X||=1
            norms = np.linalg.norm(X, axis=1, keepdims=True)  # shape (n, 1)
            X = X/norms
            mean_norms = np.linalg.norm(means, axis=1, keepdims=True)  # shape (n, 1)  
            means = means / mean_norms
    
            train_loader, test_loader, total_training_samples = prepare_data_loader(X, y)

            # Calculate statistics for the non private model for the comparison.
            model = SoftmaxRegression(input_dim=d, num_classes=M)
            train_model(model, train_loader, test_loader,
                        verbose=False, lr=0.1,
                        weight_decay=reg_lambda, reg_lambda=reg_lambda, thres=1e-5)
            torch.save(model.state_dict(), model_path)
            
            angle_non_private_i = 0.0
            acc_non_private_i = 0.0
            for _ in range(avg_iters):
                acc_non_private_i += calculate_accuracy(model, X.astype(np.float32), y)  # calculate accuracy
                ang_l=0
                for target_label in range(M):
                    # For ORL reg_lambda = 1e-4
                    # MNIST reg_lambda = 
                    rec_img, best_conf, loss_list, _ = reconstruct_face(model, target_label, test_loader,
                                                                    H=H, W=W,
                                                                    alpha=100000, step_size=0.1, init_mode="weights",
                                                                    gamma=0.99, beta=5000, tv_weight=0.0, rounding_precision=0.0,
                                                                    device="cpu", show_cost=False, noise_in_weights=False,
                                                                    snr=1e3, interactive=False, weight_decay=1e-8, reg_lambda2_weight=1e-4, thres=1e-5,
                                                                    verbose=args.verbose
                                                                    )
                    rec_img = rec_img.detach().numpy()
                    # Normalize image to [0,1]
                    original_face = normalize_vector(means[target_label], normalization_type='min_max')
                    rec_img = normalize_vector(rec_img, normalization_type='min_max')
                    ang, _ = angle_between(rec_img, original_face)
                    if not math.isnan(ang):
                        ang_l += math.sin(ang)
                angle_non_private_i += ang_l/M
            acc_non_private.append(acc_non_private_i/avg_iters)
            angles_non_private.append(angle_non_private_i/avg_iters)
            
            # Output perturbation
            models, std = generate_dp_models(d, M, total_training_samples,
                                            [epsilon_p], reg_lambda,  model_path,
                                            verbose=args.verbose, dp_type='epsilon-delta', delta=1e-5)
            acc_tmp = 0
            for _ in range(avg_iters):
                m, _ = generate_dp_models(d, M, total_training_samples, [epsilon_p], reg_lambda, model_path)
                ml = m[0]
                accuracy = calculate_accuracy(ml, X.astype(np.float32), y)  # calculate accuracy
                acc_tmp += accuracy
            acc_output.append(acc_tmp/avg_iters)
            #print(f'Accuracy: {acc_tmp/100}')
            target_label = 1
            angle_iter = 0
            for _ in range(avg_iters):
                models, _ = generate_dp_models(d, M, total_training_samples, [epsilon_p], reg_lambda, model_path)
                model = models[0]
                ang_l =0
                for target_label in range(M):
                    if args.verbose:
                        print(f"epsilon: {epsilon_p}, Reconstruction for label: {target_label}") 
                    # For ORL reg_lambda = 1e-4
                    # MNIST reg_lambda = 
                    rec_img, best_conf, loss_list, _ = reconstruct_face(model, target_label, test_loader,
                                                                    H=H, W=W,
                                                                    alpha=100000, step_size=0.1, init_mode="weights",
                                                                    gamma=0.99, beta=5000, tv_weight=0.0, rounding_precision=0.0,
                                                                    device="cpu", show_cost=False, noise_in_weights=False,
                                                                    snr=1e3, interactive=False, weight_decay=1e-8, reg_lambda2_weight=1e-4, thres=1e-5,
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
            angles_output.append(angle_iter/avg_iters)



            # Objective perturbation
            model = SoftmaxRegression(input_dim=d, num_classes=M)
            train_model(model, train_loader, test_loader,
                        verbose=False, lr=0.1,
                        weight_decay=reg_lambda, reg_lambda=reg_lambda, 
                        thres=1e-4, objective_perturbation=True, epsilon=epsilon_p, c=1/4)
            acc_tmp=0
            angle_iter=0
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
                                                                    snr=1e3, interactive=False, weight_decay=1e-2, reg_lambda2_weight=1, thres=1e-4,
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
            angles_obj.append(angle_iter/avg_iters)

            acc_obj.append(acc_tmp/avg_iters)
        
        # Save the results if specified.
        
        if args.save_results:
            np.save('results/dp_lr/data/n_arr_gaussian.npy', np.array(N))
            np.save('results/dp_lr/data/acc_obj_n.npy', np.array(acc_obj))
            np.save('results/dp_lr/data/angle_obj_n.npy', np.array(angles_obj))
            np.save('results/dp_lr/data/acc_output_n_gaussian.npy', np.array(acc_output))
            np.save('results/dp_lr/data/angle_output_n_gaussian.npy', np.array(angles_output))
            np.save('results/dp_lr/data/acc_non_private_n.npy', np.array(acc_non_private))
            np.save('results/dp_lr/data/angle_non_private_n.npy', np.array(angles_non_private))
        

        # Plot the results.
        plt.rcParams.update({'font.size': 28})
        fig, ax1 = plt.subplots()
        fig.set_size_inches(16, 9)
        color = 'tab:red'
        ax1.set_xlabel('n')
        ax1.set_ylabel('Accuracy', color=color)
        #ax1.plot(N, acc_obj, color=color, linestyle='--', marker='x', linewidth=3, label='objective')
        ax1.plot(N, acc_output, color=color, linestyle='-.', marker='o',linewidth=3, label='output')
        ax1.plot(N, acc_non_private, color=color, linestyle='-',linewidth=3, label='non private')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        
        color = 'tab:blue'
        ax2.set_ylabel(r'$Sine(\hat{x}, \mu)$', color=color)
        #ax2.plot(N, angles_obj, color=color, linestyle='--', linewidth=3, marker='x', label='objective')
        ax2.plot(N, angles_output, color=color, linestyle='-.', linewidth=3, marker='o', label='output')
        ax2.plot(N, angles_non_private, color=color, linestyle='-', linewidth=3, label='non private')
        ax2.tick_params(axis='y', labelcolor=color)

        #fig.tight_layout()

        plt.title(fr"Accuracy for different $n$ and $\epsilon_p=${epsilon_p}")
        plt.legend(loc='center right')
        plt.show()
          

        return

    if args.synthetic_data:
        M, N, d = 10, 50, 3
        H, W = d, 1
        model_path = "./data/models/model_weights_LR_synthetic_data.pth"
        X, y, means = generate_data(M=M, N=N, d=d, sigma=0.001, means="random")
        # --- Normalize X and means ---
        norms = np.linalg.norm(X, axis=1, keepdims=True)  # shape (n, 1)
        X = X/norms
        mean_norms = np.linalg.norm(means, axis=1, keepdims=True)  # shape (n, 1)  
        means = means / mean_norms
        train_loader, test_loader, total_training_samples = prepare_data_loader(X, y)
        # --- Save as MATLAB .mat file ---
        train_data = []
        train_labels = []
        
        for batch_X, batch_y in train_loader:
            train_data.append(batch_X.numpy())
            train_labels.append(batch_y.numpy())
        
        train_data = np.concatenate(train_data, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        
    elif args.mnist:
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
    else: # ORL
        d, M, N = 92 * 112, 40, 10
        H, W = 112, 92

        model_path = "./data/models/model_weights_LR.pth"
        X, y, means = load_orl_faces(dataset_path)
        #means = [mean - np.mean(mean) for mean in means]
        if args.objective:
            norms = np.linalg.norm(X, axis=1, keepdims=True)  # shape (n, 1)
            X = X/norms
            mean_norms = np.linalg.norm(means, axis=1, keepdims=True)  # shape (n, 1)  
            means = means / mean_norms
        train_loader, test_loader, total_training_samples = prepare_data_loader(X, y)

    
    print(f"N={total_training_samples}, reg_lambda={reg_lambda}")
    
    #############################################################
    # This part of the code is about objective perturbation.
    ############################################################
    if args.objective and args.compare:
        model_acc_obj = []
        angles_obj = []
        for epsilon in epsilon_p_arr:
            model = SoftmaxRegression(input_dim=d, num_classes=M)
            train_model(model, train_loader, test_loader,
                        verbose=False, lr=0.1,
                        weight_decay=reg_lambda, reg_lambda=reg_lambda, 
                        thres=1e-4, objective_perturbation=True, epsilon=epsilon, c=1/4)
            w = list(model.parameters())[0]
            w = w.detach().numpy()
            if epsilon<=1:
                avg_iters = 10
            else:
                avg_iters = 10
            acc_tmp=0
            angle_iter=0
            for i in range(avg_iters):
                accuracy = calculate_accuracy(model, X.astype(np.float32), y)  # calculate accuracy
                acc_tmp += accuracy
                ang_l=0
                for target_label in range(M):
                    if args.verbose:
                        print("Iter: ", i)
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
            angles_obj.append(angle_iter/avg_iters)

            model_acc_obj.append(acc_tmp/avg_iters)
            if args.verbose:
                print(f'Model Accuracy: {model_acc_obj[-1]}, Rec. angle: {angles_obj[-1]}')
        if args.objective:
            plt.rcParams.update({'font.size': 20})
            fig, ax1 = plt.subplots()
            fig.set_size_inches(12, 9)
            color = 'tab:red'
            ax1.set_xlabel(r'$\epsilon_p$')
            ax1.set_ylabel('Accuracy', color=color)
            ax1.semilogx(epsilon_p_arr, model_acc_obj, color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            
            ax2 = ax1.twinx()
            
            color = 'tab:blue'
            ax2.set_ylabel(r'$Sine(\hat{x}, \mu)$', color=color)
            ax2.semilogx(epsilon_p_arr, angles_obj, color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            fig.tight_layout()
            if args.synthetic_data:
                plt.title(f"Angle and accuracy for different privacy budgets \nM={M},N={N},d={d}")
            else:
                plt.title(fr"Angle and accuracy for different $\epsilon_p$")

            plt.show()

            

            return



    # Calculate statistics for the non private model for the comparison.
    model = SoftmaxRegression(input_dim=d, num_classes=M)
    train_model(model, train_loader, test_loader,
                verbose=False, lr=0.1,
                weight_decay=reg_lambda, reg_lambda=reg_lambda, thres=1e-4)
    torch.save(model.state_dict(), model_path)
    if args.compare:
        acc_non_private = calculate_accuracy(model, X.astype(np.float32), y)  # calculate accuracy
        angle_non_private = -1.0
        ang_l=0
        for target_label in range(M):
            # For ORL reg_lambda = 1e-4
            # MNIST reg_lambda = 
            rec_img, best_conf, loss_list, _ = reconstruct_face(model, target_label, test_loader,
                                                            H=H, W=W,
                                                            alpha=100000, step_size=0.1, init_mode="weights",
                                                            gamma=0.99, beta=5000, tv_weight=0.0, rounding_precision=0.0,
                                                            device="cpu", show_cost=False, noise_in_weights=False,
                                                            snr=1e3, interactive=False, weight_decay=1e-8, reg_lambda2_weight=1e-4, thres=1e-5,
                                                            verbose=args.verbose
                                                            )
            rec_img = rec_img.detach().numpy()
            # Normalize image to [0,1]
            original_face = normalize_vector(means[target_label], normalization_type='min_max')
            rec_img = normalize_vector(rec_img, normalization_type='min_max')
            ang, _ = angle_between(rec_img, original_face)
            if not math.isnan(ang):
                ang_l += math.sin(ang)
        angle_non_private = ang_l/M

    models, std = generate_dp_models(d, M, total_training_samples,
                                    epsilon_p_arr, reg_lambda,  model_path,
                                    verbose=args.verbose)
   

    ###################### Test the predictions OUTPUT perturbation #######################
    if args.acc or args.compare:
        acc_output = []
        angles_output = []
        for i in range(len(epsilon_p_arr)):
            acc_tmp = 0
            avg_iters = 50
            for _ in range(avg_iters):
                m, _ = generate_dp_models(d, M, total_training_samples, [epsilon_p_arr[i]], reg_lambda, model_path, dp_type="epsilon-delta", delta=1e-5)
                ml = m[0]
                accuracy = calculate_accuracy(ml, X.astype(np.float32), y)  # calculate accuracy
                acc_tmp += accuracy
            acc_output.append(acc_tmp/avg_iters)
            #print(f'Accuracy: {acc_tmp/100}')
            target_label = 1
            angle_iter = 0
            for _ in range(avg_iters):
                models, _ = generate_dp_models(d, M, total_training_samples, epsilon_p_arr, reg_lambda, model_path)
                model = models[i]
                ang_l =0
                for target_label in range(M):
                    if args.verbose:
                        print(f"epsilon: {epsilon_p_arr[i]}, Reconstruction for label: {target_label}") 
                    # For ORL reg_lambda = 1e-4
                    # MNIST reg_lambda = 
                    rec_img, best_conf, loss_list, _ = reconstruct_face(model, target_label, test_loader,
                                                                    H=H, W=W,
                                                                    alpha=100000, step_size=0.1, init_mode="weights",
                                                                    gamma=0.99, beta=5000, tv_weight=0.0, rounding_precision=0.0,
                                                                    device="cpu", show_cost=False, noise_in_weights=False,
                                                                    snr=1e3, interactive=False, weight_decay=1e-8, reg_lambda2_weight=1e-4, thres=1e-4,
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
            angles_output.append(angle_iter/avg_iters)

            # Status bar print
            #sys.stdout.write('\r')
            sys.stdout.write( "\n[%-20s] %d%%, epsilon=%.6f, Accuracy=%.6f, Sine(ang)=%.6f" % ('=' * ((i+1) * 20 // len(epsilon_p_arr)), (i+1) * 100 // len(epsilon_p_arr), epsilon_p_arr[i], acc_output[-1], angles_output[-1]))
            sys.stdout.flush()
        
        plt.rcParams.update({'font.size': 23})
        fig, ax1 = plt.subplots()
        fig.set_size_inches(16, 9)
        color = 'tab:red'
        ax1.set_xlabel(r'$\epsilon_p$')
        ax1.set_ylabel('Accuracy', color=color)
        if args.compare:
            ax1.semilogx(epsilon_p_arr, acc_output, color=color, linestyle='--', marker='o')
            #ax1.semilogx(epsilon_p_arr, model_acc_obj, color=color, linestyle='-.', marker='x')
            ax1.axhline(y=acc_non_private, color=color, linestyle='-')
        else:
            ax1.semilogx(epsilon_p_arr, acc_output, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        
        color = 'tab:blue'
        ax2.set_ylabel(r'$Sine(\hat{x}, \mu)$', color=color)
        if args.compare:
            ax2.semilogx(epsilon_p_arr, angles_output, color=color, linestyle='--', marker='o', label='output')
            #ax2.semilogx(epsilon_p_arr, angles_obj, color=color, linestyle='-.', marker='x', label='objective')
            ax2.axhline(y=angle_non_private, color=color, linestyle='-', label='non private')
        else:
            ax2.semilogx(epsilon_p_arr, angles_output, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        if args.synthetic_data:
            plt.title(f"Angle and accuracy for different privacy budgets \nM={M},N={N},d={d}")
        else:
            plt.title(fr"Angle and accuracy for different $\epsilon_p$ and $\delta=10^{-5}$")
        
        plt.legend(loc='center right')
        plt.show()
        
        if args.save_results:
            #np.save('results/dp_lr/data/acc_obj_epsilon.npy', np.array(model_acc_obj))
            #np.save('results/dp_lr/data/angles_obj_epsilon.npy', np.array(angles_obj))
            np.save('results/dp_lr/data/acc_output_epsilon_gaussian_delta1e-5.npy', np.array(acc_output))
            np.save('results/dp_lr/data/angles_output_epsilon_gaussian_delta1e-5.npy', np.array(angles_output))
            np.save('results/dp_lr/data/epsilon_arr_gaussian_delta1e-5.npy', np.array(epsilon_p_arr))
        

        return 

if __name__ == '__main__':
    main()
