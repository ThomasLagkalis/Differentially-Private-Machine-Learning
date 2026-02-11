import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from utils import angle_between
import math


# Helpers
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def to_hw(x_flat, H, W):
    return x_flat.view(1, 1, H, W)


def to_flat(x_img):
    return x_img.view(1, -1)

def tv_loss(img, reduction="mean"):
    """Total variation loss for smoothness."""
    #print(img.shape)
    if img.dim() == 3:
        img = img.unsqueeze(0)  # [H,W] -> [1,H,W]
    n, c, h, w = img.shape
    dh = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
    dw = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
    return (dh.mean() + dw.mean()) if reduction == "mean" else (dh.sum() + dw.sum())


def apply_rounding(logits, rounding_precision, ste=True):
    """
    Apply rounding to confidences (probabilities).
    """
    if rounding_precision <= 0:
        return logits

    probs = torch.softmax(logits, dim=1)
    
    rounded_probs = torch.round(probs / rounding_precision) * rounding_precision
    rounded_probs = rounded_probs / rounded_probs.sum(dim=1, keepdim=True)
    
    rounded_logits = torch.log(rounded_probs + 1e-8)
    
    return rounded_logits

import torch
import numpy as np

def noisy_weights(model, snr, noise_type='gaussian', verbose=False):
    """
    Adds noise to model weights based on SNR value.
    Variance = ||w||^2 / (snr * |w|)

    Args:
        model: the PyTorch model 
        snr: signal-to-noise ratio (float)
        noise_type: 'gaussian' or 'laplace' — type of additive noise

    Returns:
        model: the model with noisy weights
        std: standard deviation (or scale parameter for Laplace noise)

    Note:
        This function was designed for the ORL dataset.
        Adjustments may be needed for other architectures.
    """
    params = list(model.parameters())
    W, b = params[0], params[1]  # weight (40 x 10304), bias (40,)
    W_np = W.detach().numpy().astype(np.float64)
    b_np = b.detach().numpy().astype(np.float64)

    # Compute norm over all weights (sum of L2 norms of class weights)
    w_norm2 = 0.0
    for i in range(W.shape[0]):
        w_norm2 += torch.norm(W[i]).item()**2

    sigma2 = w_norm2 / (snr * W.numel())
    std = np.sqrt(sigma2)

    with torch.no_grad():
        if noise_type.lower() == 'gaussian':
            W.add_(torch.normal(0, std, size=W.shape))
            b.add_(torch.normal(0, std, size=b.shape))
            if verbose:
                print(f"Added Gaussian noise to the weights. std={std:.4e}, sigma2={sigma2:.4e}")

        elif noise_type.lower() == 'laplace':
            # Laplace noise with mean=0, scale = std / sqrt(2)
            # (since Laplace variance = 2 * scale^2)
            scale = std / np.sqrt(2)
            signal_power = np.sum(W_np**2)

            w_noise = np.random.laplace(0, scale, size=W_np.shape).astype(np.float64)
            noise_power = np.sum(w_noise**2)
            
            b_noise = np.random.laplace(0, scale, size=b.shape).astype(np.float64)
            
            #W = W - 0.5
            #b = b - 0.5
            W.add_(torch.from_numpy(w_noise.astype(np.float32)))
            b.add_(torch.from_numpy(b_noise.astype(np.float32)))
            #W = W + 0.5
            #b = b + 0.5
            #for param in model.parameters():
            #    param.clamp_(0, 1)

            # Measure the SNR for validation (it should be the same as input snr).

            snr_experimental = signal_power / noise_power
            
            if verbose:
                print(f"Added Laplace noise to the weights. scale={scale:.4e}, variance={sigma2:.4e}, experimental snr = {snr_experimental}")

        else:
            raise ValueError("noise_type must be either 'gaussian' or 'laplace'.")

    return model, std

    

# -----------------------------------------------------------
# Attack core, basic, as described in paper
# -----------------------------------------------------------
def reconstruct_face(
    model,
    label,
    class_mean=None, # For interactive plots
    test_loader=None,
    H=112,
    W=92,
    alpha=5000,
    step_size=0.1,
    gamma=0.99,
    beta=100,
    tv_weight=0.01,
    rounding_precision=0.0,
    init_mode="zeros",
    prior_img=None,
    device="cpu",
    show_cost=False,
    verbose=False,
    noise_in_weights=False,
    snr=0,
    interactive = False,
    reg_lambda2_weight = 0.0,
    weight_decay=0.0,
    thres = 1e-5
):
    """
    MI-Face reconstruction for Softmax regression.
    """
    if verbose: 
        print(f"Beginning reconstruction attack, target label = {label}")
    if noise_in_weights==True:
        model=noisy_weights(model, snr)
    acc = -1.0	
    if test_loader!=None:	
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
        if verbose:
            print(f"Test Accuracy with noisy weights: {acc:.4f}")

    # Initialize image a.k.a x0 (default is zeros)
    if init_mode == "zeros":
        x_img = torch.zeros(1, 1, H, W, device=device)
    elif init_mode == "noise":
        x_img = torch.randn(1, 1, H, W, device=device)
        x_img.requires_grad_(True)
    elif init_mode == "ones":
        x_img = torch.ones(1, 1, H, W, device=device)
    elif init_mode == "weights":
        params = list(model.parameters())
        w = params[0]
        w = w[label].detach().cpu().numpy()
        w = w.reshape(H,W)
        w = torch.from_numpy(w)
        w = w.view(1, 1, H, W).to(device) 
        x_img = w.clone()
    else:
        x_img = torch.zeros(1, 1, H, W, device=device)
    
    x_img = to_flat(x_img)
    x_img = x_img.detach().requires_grad_()
    optimizer = torch.optim.SGD([x_img], lr=step_size, weight_decay=weight_decay)

    best_img, best_prob, no_improve = None, -1.0, 0
    start = time.time()
    iters = 0
    loss_list = []
    
    # Interactive plot
    #plt.ion()
    #x = x_img.detach().numpy()
    #graph = plt.plot(x[0][1:100], color='green')[0]
    #plt.pause(0.5)
    while(True) :
        
        x_prev = x_img.clone()
        iters += 1
        optimizer.zero_grad()
        x_flat = to_flat(x_img)
        logits = model(x_img)
        # Adaptive learning rate 
        if reg_lambda2_weight > 0:
            d_k = 2.0 / (reg_lambda2_weight * (iters + 1000))
            for param_group in optimizer.param_groups:
                param_group['lr'] = d_k

        log_probs = F.softmax(logits, dim=1)
        pred = torch.argmax(log_probs)
        
        x_img_2d = x_img.view(1, 1, H, W).to(device)
        tvl = tv_loss(x_img_2d)
        loss = 1-log_probs[0, label] + tv_weight * tvl   # maximize probability of label
        #loss = 1 - logits[0, label]
        loss.backward()
        #torch.nn.utils.clip_grad_norm_([x_img], max_norm=1.0)
        optimizer.step()
        loss_list.append(loss.item())
        
        # Interactive plot update
        #x = x_img.detach().numpy()
        #graph.remove()
        #plt.clf()
        #graph = plt.plot(x[0][1:100], color='green')[0]
        #graph = plt.plot(class_mean[1:100], label='class mean', color='red')[0]
        #plt.pause(0.000001)

        with torch.no_grad():
            #x_img.clamp_(0.0, 1.0)
            probs = F.softmax(logits, dim=1)
            #print(probs)
            p = probs[0, label].item()
            if p>best_prob:
                best_prob = p
                best_img = x_img
            thres_val = abs(np.linalg.norm(x_prev) - np.linalg.norm(x_img))/np.linalg.norm(x_img)
            if iters % 1000 == 0 and verbose:
                print(f"Iter.: {iters}, Threshold condition value: {thres_val:.7f}, best_prob: {p:.4f}")
            if ( thres_val < thres):
                break
            
    
    if show_cost==True:
        plt.close()
        plt.plot(loss_list)
        plt.title("Attack cost per iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.savefig("results/attack_cost.png")

    elapsed = time.time() - start
    return best_img if best_img is not None else x_img.detach(), best_prob, loss_list, acc 


# ----------------------------
# Save reconstruction
# TODO: add scale factor a 
# by solving the least squeres problem
# see main.py
# ----------------------------
def save_reconstruction(img_tensor, conf, label, mult_factor=1, rescale=False, results_dir="results/attack_results", tag="", avg_image_path="data/avg_images/"):
    ensure_dir(results_dir)
    fname = os.path.join(results_dir, f"mi_face_label{label}{('_' + tag) if tag else ''}")

    img_np = img_tensor.squeeze().detach().cpu().numpy()
    img_np *= mult_factor
    if np.max(img_np) > 1:
        img_np = img_np/255.0 # Ensure consistent range before do anything
    
    # Normalize from [0, 1] to [0, 255] if needed
    if np.max(img_np) <= 1.0 and rescale==True:
        #print("Entered")
        img_np = (img_np * 255).astype(np.uint8)

    avg_img_path = os.path.join(avg_image_path, f"average_face{label}.png")
    if os.path.exists(avg_img_path):
        avg_img_vec = cv2.imread(avg_img_path, cv2.IMREAD_GRAYSCALE)
        avg_img = cv2.resize(avg_img_vec, (92,112))
        avg_img = avg_img.astype(np.float32) / 255.0
       # print(np.max(avg_img).astype(np.float32))
        max_avg_image = np.max(avg_img).astype(np.float32)
        min_avg_image = np.min(avg_img).astype(np.float32)
    
    img_vec = img_np.flatten()
    avg_img_vec = avg_img.flatten()
    a = np.dot(avg_img_vec.T, img_vec)/np.dot(img_vec.T, img_vec)
    max_img = np.max(img_np)
    min_img = np.min(img_np)
    vmax = max(max_avg_image, max_img)
    vmin = min(min_avg_image, min_img)

    #img_np = img_np * (255/max_img)
    #print(max_img, max_avg_image, vmax)

    #if max_img > 0:  # Avoid division by zero
    #    scaling_factor =  max_img / max_avg_image
    #    img_np = (img_np * scaling_factor).astype(np.uint8)
    #    img_np = np.clip(img_np, 0, 255)
    
    scaled_img = a*img_np
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im1 = axes[0].imshow(img_np, cmap="grey")
    axes[0].set_title("Reconstruction (original color scale)")
    fig.colorbar(im1, ax=axes[0])
    im2 = axes[1].imshow(scaled_img, cmap="grey", vmax=vmax, vmin=vmin)
    axes[1].set_title("Reconstruction")
    fig.colorbar(im2, ax=axes[1])
    im3 = axes[2].imshow(avg_img, cmap="grey", vmax=vmax, vmin=vmin)
    axes[2].set_title("Average")
    fig.colorbar(im3, ax=axes[2])
    plt.savefig(f"{fname}.png")
    
    # Also save the numpy array
    np.save(fname + ".npy", img_np)

    with open(fname + ".txt", "w") as f:
        f.write(f"best_confidence={conf:.6f}\n")
    print(f"[saved] {fname}.png (conf={conf:.4f})")
