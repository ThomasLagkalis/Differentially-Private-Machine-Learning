from model import SoftmaxRegression, load_orl_faces, train_model, WeightClipper  
from attack import reconstruct_face, save_reconstruction

import argparse
import torch
import math
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from utils import angle_between
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Command line arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--lambda_experiments", action='store_true', help = "Perform accuracy expiriments with 20 reg lambda values")
    parser.add_argument("-w", "--weights_experiments", action='store_true', help = "Perform experiment with 3 different lambdas and plot the angle of the weights, the 2-norm of the weights and the ratio of succesive weights norms.")
    parser.add_argument("-c", "--attack_cost", action='store_true', help = "Perform an attack on one instance and plot the attack cost per iteration")
    parser.add_argument("-i", "--interactive", action='store_true', help="Perform an attack and plot the weights in each iteration.")
    parser.add_argument("-n","--mlp",action='store_true', help="Perform experiments using the MLP described in the paper.") 
    

    args = parser.parse_args()


    dataset_path = "data/orl"  
    results_dir = "results/attack_results"

    # Load dataset 
    X, y, avg_faces = load_orl_faces(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)
    
    if args.mlp:
        

        return
    if args.lambda_experiments:
        print("Performing 10 experiments for train loss and test accuracy results...")
        #lambdas = np.linspace(1e-3, 1, num=10)
        lambdas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        for l in lambdas:
            print(f"Experiment with lambda = {l}")
            input_dim, num_classes = 92 * 112, 40
            model = SoftmaxRegression(input_dim, num_classes)
            train_loss, test_acc = train_model(model, train_loader, test_loader, epochs=30, lr=0.1, weight_decay=0.0, tv_weight=1e-4, bloss_weight=0.0, reg_lambda=l, verbose=False)
            
            # plot the results
            plt.subplot(1, 2, 1)
            plt.plot(train_loss, label=f"lambda={l}")
            
            plt.subplot(1, 2, 2)
            plt.plot(test_acc, label=f"lambda={l}")
        plt.subplot(1, 2, 1)
        plt.xlabel("Iterations")
        plt.ylabel("Train loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.xlabel("Iterations")
        plt.ylabel("Test accuracy")
        plt.legend()
        plt.show()
    elif args.weights_experiments:
        print("Performing 3 experiments for the weights...")
        lambdas = [0.005, 0.05, 0.5]
        w_l = []
        weights_l = []
        for l in lambdas:
            print(f"Experiment with lambda = {l}")
            input_dim, num_classes = 92 * 112, 40
            model = SoftmaxRegression(input_dim, num_classes)
            _, _ = train_model(model, train_loader, test_loader, epochs=30, lr=0.1, weight_decay=l, tv_weight=1e-4, bloss_weight=0.0, reg_lambda=0, verbose=False)
            params = list(model.parameters())
            params = params[0]
            params = params.detach().numpy()
            iteration = 0
            fixed_vec = np.zeros(( len(params[0])))
            fixed_vec[0] = 1
            w_norms = []
            w_list = []
            for w in params:
                w_list.append(w)
                norm = np.linalg.norm(w)
                w_norms.append(norm)
                iteration += 1
            w_l.append(w_norms)
            weights_l.append(w_list)
        

            
            # plot the results
            plt.subplot(1, 3, 2)
            plt.scatter(range(len(w_norms)), w_norms, label=f"lambda={l}")
        
        ratios12 = np.zeros(len(w_norms))
        ratios23 = np.zeros(len(w_norms))
        ratios13 = np.zeros(len(w_norms))
        angles12 = np.zeros(len(w_list))
        angles23 = np.zeros(len(w_list))
        angles13 = np.zeros(len(w_list))
        #print(len(w_l[0]), len(w_l[1]), len(w_l[2]), len(w_l))
        for w_i in range(len(w_norms)):
            ratios12[w_i] = w_l[0][w_i] / w_l[1][w_i] 
            ratios23[w_i] = w_l[1][w_i] / w_l[2][w_i] 
            ratios13[w_i] = w_l[0][w_i] / w_l[2][w_i] 
            
            ang, _ = angle_between(weights_l[0][w_i], weights_l[0][w_i])
            angles12[w_i] = math.sin(ang)
            ang, _ = angle_between(weights_l[1][w_i], weights_l[2][w_i])
            angles23[w_i] = math.sin(ang)
            ang, _ = angle_between(weights_l[0][w_i], weights_l[2][w_i])
            angles13[w_i] = math.sin(ang)


        plt.subplot(1, 3, 3)
        plt.scatter(range(len(ratios12)), ratios12, label=f"l={lambdas[0]} amd l={lambdas[1]}")
        plt.scatter(range(len(ratios23)), ratios23, label=f"l={lambdas[1]} amd l={lambdas[2]}")
        plt.scatter(range(len(ratios13)), ratios13, label=f"l={lambdas[0]} amd l={lambdas[2]}")
        
        plt.subplot(1, 3, 1)
        plt.scatter(range(len(angles12)), angles12, label=f"l={lambdas[0]} amd l={lambdas[1]}")
        plt.scatter(range(len(angles23)), angles23, label=f"l={lambdas[1]} amd l={lambdas[2]}")
        plt.scatter(range(len(angles13)), angles13, label=f"l={lambdas[0]} amd l={lambdas[2]}")

        plt.subplot(1, 3, 1)
        plt.xlabel("Weight")
        plt.ylabel("sine")
        plt.ylim(-1.1, 1.1)
        plt.legend()
        plt.subplot(1, 3, 2)
        plt.xlabel("Weight")
        plt.ylabel("Norm Value")
        plt.legend()
        plt.subplot(1, 3, 3)
        plt.xlabel("Weight")
        plt.ylabel("Ratio Value")
        plt.legend()
        plt.show()  
    elif args.attack_cost:
        # Train model 
        input_dim, num_classes = 92 * 112, 40
        model = SoftmaxRegression(input_dim, num_classes)
        train_model(model, train_loader, test_loader, epochs=30, lr=0.1, weight_decay=0.0, tv_weight=1e-4, bloss_weight=0.0, reg_lambda=5*1e-3)
        target_label = 1  # example
        rec_img, best_conf, loss_list = reconstruct_face(model, target_label,
                                                  H=112, W=92,
                                                  alpha=3000, step_size=0.01, init_mode="noise",
                                                  gamma=0.999, beta=300, tv_weight=0.01, rounding_precision=0.0,
                                                         device="cpu", show_cost=True, noise_in_weights=True, snr=1e3, interactive=False)

        save_reconstruction(rec_img, best_conf, target_label, results_dir="results/", mult_factor=1, rescale=False)

    elif args.interactive:
        input_dim, num_classes = 92 * 112, 40
        model = SoftmaxRegression(input_dim, num_classes)
        train_model(model, train_loader, test_loader, epochs=30, lr=0.1, weight_decay=0.0, tv_weight=1e-4, bloss_weight=0.0, reg_lambda=5*1e-3)
        target_label = 1  # example
        rec_img, best_conf, loss_list = reconstruct_face(model, target_label,
                                                  H=112, W=92,
                                                  alpha=3000, step_size=0.01, init_mode="noise",
                                                  gamma=0.999, beta=300, tv_weight=5, rounding_precision=0.0,
                                                         device="cpu", show_cost=True, noise_in_weights=True, snr=1e3, interactive=True)

    else:
        # Train model 
        input_dim, num_classes = 92 * 112, 40
        model = SoftmaxRegression(input_dim, num_classes)
        train_model(model, train_loader, test_loader, epochs=30, lr=0.1, weight_decay=0.0, tv_weight=1e-4, bloss_weight=0.0, reg_lambda=5*1e-3)
        
        # Plot the weights for experimenting
        for person_id in range(40):
            params = list(model.parameters())
            W = params[0]
            img = W[person_id].detach().cpu().numpy()
            img_vec = img
            #print(f"weight before scale {np.min(img_vec)}, {np.max(img_vec)}")
            img = img.reshape(112,92)
            avg = avg_faces[person_id].reshape(112,92)
            avg_vec = avg_faces[person_id]
            a = np.dot(avg_vec.T, img_vec)/np.dot(img_vec.T, img_vec)
            W_scaled = a*img_vec
            #print(f"scaled vector: {np.min(W_scaled)}, {np.max(W_scaled)}")
            #print(f"averag vector: {np.min(avg_vec)}, {np.max(avg_vec)}")
            #print()
            vmin = min(np.min(img_vec), np.min(W_scaled), np.min(avg_vec))
            vmax = max(np.max(img_vec), np.max(W_scaled), np.max(avg_vec))
            img_scaled = W_scaled.reshape(112,92)
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            im1 = axes[0].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            axes[0].set_title("weight")
            fig.colorbar(im1, ax=axes[0])
            im2 = axes[1].imshow(img_scaled, cmap="gray", vmin=vmin, vmax=vmax)
            axes[1].set_title(f"a*weight (a={a:.2f})")
            fig.colorbar(im2, ax=axes[1])
            im3 = axes[2].imshow(avg, cmap="gray", vmin=vmin, vmax=vmax)
            axes[2].set_title("Average")
            fig.colorbar(im3, ax=axes[2])
            plt.savefig(f"results/faces_weights/face_{person_id}.png")
    
        # Run attack on a all labels 
        for i in range(40):
            target_label = i  # example
            rec_img, best_conf, loss_list = reconstruct_face(model, target_label,
                                                  H=112, W=92,
                                                  alpha=50000, step_size=0.01, init_mode="zeros",
                                                  gamma=0.99, beta=1000, tv_weight=0.01, rounding_precision=0.0,
                                                  device="cpu", show_cost=False, interactive=False)
            save_reconstruction(rec_img, best_conf, target_label, results_dir=results_dir, mult_factor=5, rescale=False)
            

if __name__ == "__main__":
    main()

