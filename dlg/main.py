import copy
from model import *
from utils import prepare_mnist, prepare_data_loader, moving_avg, show_reconstruction, load_mnist_lenet
import torch.nn as nn
import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

def add_noise_to_grads(model, snr):
    # Compute gradient norm and size
    grad_sq_sum = 0.0
    num_elems = 0
    grad_tuple = []

    for p in model.parameters():
        if p.grad is not None:
            grad_sq_sum += torch.sum(p.grad ** 2)
            num_elems += p.grad.numel()

    # Compute sigma from SNR
    rms_grad = torch.sqrt(grad_sq_sum / num_elems)
    sigma = rms_grad / torch.sqrt(torch.tensor(snr))
    sigma = torch.clamp(sigma, min=1e-8, max=100.0)

    # Add Gaussian noise to each gradient
    for p in model.parameters():
        if p.grad is not None:
            noise = torch.randn_like(p.grad) * sigma
            p.grad.add_(noise)
            grad_tuple.append(p.grad)
    
    grad_tuple = tuple(grad_tuple)
    return sigma, grad_tuple

    
def dlg_attack(original_dy_dx, model, criterion=cross_entropy_for_onehot):    
    # generate dummy data and label
    dummy_data = torch.randn((1,1,32,32)).requires_grad_(True)
    dummy_label = torch.randn((1,10), requires_grad=True)
    #optimizer = torch.optim.Adam([dummy_data, dummy_label], lr=0.1)
    optimizer = torch.optim.LBFGS([dummy_data,], lr=1)
    criterion_iDLG = nn.CrossEntropyLoss()

    history = []
    convergance = 1e10
    prev_loss = 1e10
    current_loss = 1e10
    iters = 0
    while (current_loss > 1e-4 and iters < 1000):
        if iters != 0:
            prev_loss = current_loss
        def closure():
            optimizer.zero_grad()
    
            pred = model(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            #dummy_loss = criterion(pred, dummy_onehot_label) 
            label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)
            dummy_loss = criterion_iDLG(pred, label_pred)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

    
            grad_diff = 0
            grad_count = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx): 
                grad_diff += ((gx - gy) ** 2).sum()
                grad_count += gx.nelement()
            # grad_diff = grad_diff / grad_count * 1000
            grad_diff.backward()
    
            return grad_diff
    
        current_loss = optimizer.step(closure)
        current_loss = current_loss.item()
        convergance = np.abs(current_loss - prev_loss)/np.abs(current_loss)
        #if iters % 10 == 0:
        #    print(iters, "%.4f" % current_loss)
        history.append(dummy_data[0].cpu())
        iters += 1
       
    print(f'Iters: {iters}, Loss: {current_loss:.4f}, Convergance value: {convergance:.6f}')
    return history

def noisy_train(model, train_loader, test_loader, epochs, optimizer, loss_function, snr):
    loss_per_epoch = []
    test_acc_per_epoch = []
    train_acc_per_epoch = []
    rec_x_per_epoch = []
    rec_error_per_epoch = []
    grad_norm_per_epoch = []
    
    
    for epoch in range(epochs):
        model.train()
        tot_loss = 0
        correct, total = 0, 0
        rec_error = 0.0

        rec_error_per_epoch.append([])
        grad_norm_per_epoch.append([])
        for i, data in enumerate(train_loader, 0):
            x_batch, y_batch = data
            optimizer.zero_grad()

            outputs = model(x_batch)
            loss = loss_function(outputs, y_batch)
            loss.backward()
            sigma, grad = add_noise_to_grads(model, snr) # Here we calculate std based on snr and add noise to gradients.
            

            # Perform attack. Note that in the first attack (i.e. epoch 1) the gradients are noiseless
            # because we first attack and then add noise.
            grad_norm_per_epoch[epoch].append(grad_norm(model))
            if (i%30 == 0) and (epoch < 2): 
                # Share the gradients with clients.
                # Make a deep copy of the model so we avoid conflicts with backpropagation.
                frozen_model = copy.deepcopy(model).eval()

                history = dlg_attack(grad, frozen_model)

                rec_x = history[-1].detach().numpy()
                x = x_batch.detach().numpy()
                rec_error = np.linalg.norm(rec_x - x)/np.linalg.norm(x)
                if i == 0:
                    rec_x_per_epoch.append(rec_x)
                rec_error_per_epoch[epoch].append(rec_error)
                print(f'Iter: {i+1}, rec_error: ', rec_error, f',grad L2 norm: {grad_norm_per_epoch[epoch][-1]}, Sigma: {sigma}')
            
             if i%1000 == 0:
                show_reconstruction(rec_x, epoch, i, title=fr"rec_error: {rec_error:.4f}, epoch: {epoch+1}, iter: {i+1}")
            
            optimizer.step()

            tot_loss += loss.item()

            outputs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        

        loss_per_epoch.append(tot_loss)
        train_acc_per_epoch.append(correct / total)

        # Evaluation
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
        
        print(f"Epoch {epoch+1}: Test acc: {acc:.4f}, train acc: {train_acc_per_epoch[epoch]:.4f}, epoch loss: {loss_per_epoch[epoch]:.4f}, reconstruction error: {rec_error:.4f}")
    
    error_grads = (rec_error_per_epoch, grad_norm_per_epoch)
    return test_acc_per_epoch, train_acc_per_epoch, loss_per_epoch, tot_loss, rec_x_per_epoch, error_grads



if __name__ == '__main__':
    #X, y = prepare_mnist(5000)
    #train_loader , test_loader, total_samples = prepare_data_loader(X, y)
    train_loader, test_loader = load_mnist_lenet(batch_size=1, n_train=4000, n_test=1000)
    
    ####################################################################################################################
    # First perform experiments without noise and then with addititve noise.
    ####################################################################################################################
    #model = MLP(X.shape[1], 10)
    model = LeNet()
    model.apply(weights_init)
    loss_function = nn.CrossEntropyLoss(reduction="mean")
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 7
    snr = 1e2

    #test_acc_per_epoch, train_acc_per_epoch, loss_per_epoch, cummulative_loss = train(model, train_loader, test_loader, epochs, optimizer, loss_function)

    #model = MLP(X.shape[1], 10)
    model = LeNet()
    model.apply(weights_init)
    loss_function = nn.CrossEntropyLoss(reduction="mean")
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    noisy_test_acc_per_epoch, noisy_train_acc_per_epoch, noisy_loss_per_epoch, noisy_cummulative_loss, noisy_rec_x_per_epoch, error_grads = noisy_train(model, train_loader, test_loader, 
                                                                                                                                           epochs, optimizer, loss_function, snr)
    errors = error_grads[0]
    grads = error_grads[1]


    plt.clf()
    plt.rcParams.update({'font.size': 28})
    plt.subplot(2,1,1)
    plt.semilogy(moving_avg(errors[0], 10), label="Epoch 1")
    plt.semilogy(moving_avg(errors[1], 10), label="Epoch 2")
    #plt.semilogy(moving_avg(errors[2], 10), label="Epoch 3")
    plt.title("Reconstruction error (moving avg, window=10)")
    plt.xlabel("Iteration")
    plt.ylabel(r'$\frac{\|x-\hat{x}\|_2}{\|x\|_2}$')
    plt.xticks(np.arange(0, 3000/30, 500))
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(moving_avg(grads[0], 1000), label="Epoch 1")
    plt.plot(moving_avg(grads[1], 1000), label="Epoch 2")
    #plt.plot(moving_avg(grads[2], 1000), label="Epoch 3")
    plt.xlabel("Iteration")
    plt.ylabel(r'$\|\nabla W\|_2$')
    plt.title(r"Gradients $L_2$ norm (moving avg, window=1000)")
    #plt.xticks([i for i in range(0, 4000*epochs+1, 4000*epochs//3)], ['0', '5', '10', '15'])
    plt.legend()

    plt.show()

