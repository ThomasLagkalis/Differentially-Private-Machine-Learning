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

def noisy_weights(model, snr):
    """
    Adds noise to weights base on snr value: variance = norm(w)^2/(snr*|w|)
    
    Input: model: the pytorch model 
            snr: the snr value
    Returns the model with the noisy_weights

    Note this function works only for the ORL dataset. Adjustments may needed in different
    datasets
    """
    params = list(model.parameters())
    W, b = params[0], params[1]  # weight (40 x 10304), bias (40,)

    # First calculate the norm over all weights
    w_norm = 0.0
    for i in range(W.shape[0]):   # loop over 40 classes
        w_norm += torch.norm(W[i]).item()

    # Then calculate the variance
    sigma2 = w_norm/(snr*40)
    std = np.sqrt(sigma2)

    with torch.no_grad():
        W.add_(torch.normal(0, std, size=W.shape))
        b.add_(torch.normal(0, std, size=b.shape))

    print("Added noise to the weights.")
    return model

    

# -----------------------------------------------------------
# Attack core, basic, as described in paper
# TODO:  check gradients calculations
# -----------------------------------------------------------
def reconstruct_face(
    model,
    label,
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
    verbose=True,
    noise_in_weights=False,
    snr=0,
    interactive = False
):
    """
    MI-Face reconstruction for Softmax regression.
    """
    if noise_in_weights==True:
        model=noisy_weights(model, snr)

    model.eval()

    # Initialize image a.k.a x0 (default is zeros)
    if init_mode == "zeros":
        x_img = torch.zeros(1, 1, H, W, device=device, requires_grad=True)
    elif init_mode == "noise":
        x_img = torch.randn(1, 1, H, W, device=device)
        x_img.requires_grad_(True)
    elif init_mode == "ones":
        x_img = torch.ones(1, 1, H, W, device=device, requires_grad=True)
    elif init_mode == "weights":
        params = list(model.parameters())
        w = params[0]
        w = w[label].detach().cpu().numpy()
        w = w.reshape(112,92)
        w = torch.from_numpy(w)
        w = w.view(1, 1, H, W).to(device) 
        x_img = w.clone().requires_grad_(True)
    else:
        x_img = torch.zeros(1, 1, H, W, device=device, requires_grad=True)
    
    optimizer = torch.optim.SGD([x_img], lr=step_size )

    best_img, best_prob, no_improve = None, -1.0, 0
    start = time.time()
    iters = 0
    loss_list = []

    #plt.figure()
    fig, ax = plt.subplots(1, 3, figsize=(10, 4))

    # Right subplot: static model weights
    params = list(model.parameters())
    w = params[0][label].detach().cpu().numpy()
    ax[2].scatter(range(len(w)), w)
    ax[2].set_title("Model Weights")
    #ax[1].set_ylim(-0.002, 0.05)
    
    plt.ion()  # interactive mode on
    angles = [] # angle per iteration
    # Left subplot: interactive updates
    for t in range(1, alpha + 1):
        if interactive == True :
            x = to_flat(x_img).detach().cpu().numpy()[0]
            a = np.dot(w.T, x)/np.dot(x.T, x)
            x = a*x
    
            ax[1].cla() 
            ax[1].scatter(range(len(x)), x, label="x_img")
            #ax[0].set_ylim(-0.002, 0.05)
            ax[1].set_title(f"Scaled (a = {a:.1f}) x_img (iter {t})")
            #ax[0].legend()

            ang, _ = angle_between(x, w)
            angles.append(math.sin(ang))
            ax[0].cla()
            ax[0].plot(angles)
            ax[0].set_title(f"Angle of x_img and w[label]")
    
            plt.pause(0.0001)       

        iters += 1
        optimizer.zero_grad()
        x_flat = to_flat(x_img)
        logits = model(x_flat)
        if rounding_precision > 0:
            logits = apply_rounding(logits, rounding_precision)

        log_probs = F.softmax(logits, dim=1)
        pred = torch.argmax(log_probs)
        
        tvl = tv_loss(x_img)
        #print(tv_weight, tvl) 
        loss = 1-log_probs[0, label] + tv_weight * tvl   # maximize probability of label
        #loss = 1 - logits[0, label]
        loss.backward()
        torch.nn.utils.clip_grad_norm_([x_img], max_norm=1.0)
        optimizer.step()
        loss_list.append(loss.item())

        with torch.no_grad():
            x_img.clamp_(0.0, 1.0)
            probs = F.softmax(logits, dim=1)
            p = probs[0, label].item()
            if p > best_prob:
                best_prob, best_img, no_improve = p, x_img.detach().clone(), 0
            else:
                no_improve += 1

        if best_prob >= gamma or no_improve >= beta:
            break
    
    if interactive == True:
        plt.ioff()  
        plt.show()  
    
    if show_cost==True:
        plt.close()
        plt.plot(loss_list)
        plt.title("Attack cost per iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.savefig("results/attack_cost.png")

    elapsed = time.time() - start
    if verbose:
        print(f"[label {label}] best_conf={best_prob:.4f}, iterations={iters}, time={elapsed}")
    return best_img if best_img is not None else x_img.detach(), best_prob, loss_list


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
