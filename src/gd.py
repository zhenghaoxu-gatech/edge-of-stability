from os import makedirs
from datetime import datetime
from tqdm import tqdm

import torch
from torch.nn.utils import parameters_to_vector

import argparse

from archs import load_architecture
from utilities import get_gd_optimizer, get_gd_directory, get_loss_and_acc, compute_losses, \
    save_files, save_files_final, get_hessian_eigenvalues, iterate_dataset
from data import load_dataset, take_first, DATASETS

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def main(dataset: str, arch_id: str, loss: str, opt: str, lr: float, max_steps: int, neigs: int = 0,
         physical_batch_size: int = 1000, eig_freq: int = -1, iterate_freq: int = -1, save_freq: int = -1,
         save_model: bool = False, beta: float = 0.0, nproj: int = 0,
         loss_goal: float = None, acc_goal: float = None, abridged_size: int = 5000, seed: int = 0, width: int = 200, bias: bool = True, init_bias: str = "b_init", init_weight: str = "w_init", batch_norm: bool = False):
    directory = get_gd_directory(dataset, lr, arch_id, seed, opt, loss, beta, 
                                 width=width, bias=bias, init_bias=init_bias, init_weight=init_weight, batch_norm=batch_norm)
    print(f"output directory: {directory}")
    makedirs(directory, exist_ok=True)

    train_dataset, test_dataset = load_dataset(dataset, loss)
    abridged_train = take_first(train_dataset, abridged_size)

    loss_fn, acc_fn = get_loss_and_acc(loss)

    torch.manual_seed(seed)
    network = load_architecture(arch_id, dataset, width=width, bias=bias, batch_norm=batch_norm)

    model_path = get_gd_directory(dataset, lr, arch_id, seed, opt+"_model", loss, beta, 
                                 width=width, bias=bias, init_bias=init_bias, init_weight=init_weight, batch_norm=batch_norm)+"/initial.pth"
    
    if not os.path.exists(model_path):
        torch.save(network.state_dict(), model_path)
    else: 
        network.load_state_dict(torch.load(model_path))

    for name, parameters in network.named_parameters():
        idx = name.split('.')[0]
        label = name.split('.')[1]
        if label == 'bias' and init_bias != "b_init":
            b1 = init_bias.split('_')[1] 
            b2 = init_bias.split('_')[3] 
            if idx == '1':
                parameters.data = parameters.data / parameters.norm().item() * float(b1)
            if idx == ('4' if batch_norm else '3'):
                parameters.data = parameters.data / parameters.norm().item() * float(b2)
        if label == 'weight' and init_weight != "w_init": 
            w1 = init_weight.split('_')[1] 
            w2 = init_weight.split('_')[3] 
            if idx == '1':
                parameters.data = parameters.data / parameters.norm().item() * float(w1)
            if idx == ('4' if batch_norm else '3'):
                parameters.data = parameters.data / parameters.norm().item() * float(w2)
        # print(name, ':', parameters.size(), ':', parameters.norm().item())
    # exit()
    network.cuda()

    torch.manual_seed(7)
    projectors = torch.randn(nproj, len(parameters_to_vector(network.parameters())))

    optimizer = get_gd_optimizer(network.parameters(), opt, lr, beta)

    train_loss, test_loss, train_acc, test_acc = \
        torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps)
    iterates = torch.zeros(max_steps // iterate_freq if iterate_freq > 0 else 0, len(projectors))
    eigs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs)
    w1_list, w2_list = \
        torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0), torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0)

    for step in tqdm(range(0, max_steps)):
        train_loss[step], train_acc[step] = compute_losses(network, [loss_fn, acc_fn], train_dataset,
                                                           physical_batch_size)
        test_loss[step], test_acc[step] = compute_losses(network, [loss_fn, acc_fn], test_dataset, physical_batch_size)

        if eig_freq != -1 and step % eig_freq == 0:
            eigs[step // eig_freq, :] = get_hessian_eigenvalues(network, loss_fn, abridged_train, neigs=neigs,
                                                                physical_batch_size=physical_batch_size)
            with torch.no_grad():
                for name, parameters in network.named_parameters():
                    idx = name.split('.')[0]
                    label = name.split('.')[1]
                    # print(name)
                    if label == 'weight': 
                        if idx == '1':
                            w1_list[step // eig_freq] = parameters.norm()
                        if idx == ('4' if batch_norm else '3'):
                            w2_list[step // eig_freq] = parameters.norm()
            # print("eigenvalues: ", eigs[step//eig_freq, :], "\t w1: ", w1_list[step // eig_freq], "\t w2: ", w2_list[step // eig_freq])
            # exit()

        if iterate_freq != -1 and step % iterate_freq == 0:
            iterates[step // iterate_freq, :] = projectors.mv(parameters_to_vector(network.parameters()).cpu().detach())

        if save_freq != -1 and step % save_freq == 0:
            save_files(directory, [("eigs", eigs[:step // eig_freq]), ("iterates", iterates[:step // iterate_freq]),
                                   ("w1", w1_list[:step // eig_freq]), ("w2", w2_list[:step // eig_freq]),
                                   ("train_loss", train_loss[:step]), ("test_loss", test_loss[:step]),
                                   ("train_acc", train_acc[:step]), ("test_acc", test_acc[:step])])

        # print(f"{step}\t{train_loss[step]:.3f}\t{train_acc[step]:.3f}\t{test_loss[step]:.3f}\t{test_acc[step]:.3f}")

        if (loss_goal != None and train_loss[step] < loss_goal) or (acc_goal != None and train_acc[step] > acc_goal):
            break

        optimizer.zero_grad()
        for (X, y) in iterate_dataset(train_dataset, physical_batch_size):
            loss = loss_fn(network(X.cuda()), y.cuda()) / len(train_dataset)
            loss.backward()
        optimizer.step()

    save_files_final(directory,
                     [("eigs", eigs[:(step + 1) // eig_freq]), ("iterates", iterates[:(step + 1) // iterate_freq]),
                      ("w1", w1_list[:(step+1) // eig_freq]), ("w2", w2_list[:(step+1) // eig_freq]),
                      ("train_loss", train_loss[:step + 1]), ("test_loss", test_loss[:step + 1]),
                      ("train_acc", train_acc[:step + 1]), ("test_acc", test_acc[:step + 1])])
    if save_model:
        torch.save(network.state_dict(), f"{directory}/snapshot_final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train using gradient descent.")
    parser.add_argument("dataset", type=str, choices=DATASETS, help="which dataset to train")
    parser.add_argument("arch_id", type=str, help="which network architectures to train")
    parser.add_argument("loss", type=str, choices=["ce", "mse", "huber", "sigmoid"], help="which loss function to use")
    parser.add_argument("lr", type=float, help="the learning rate")
    parser.add_argument("max_steps", type=int, help="the maximum number of gradient steps to train for")
    parser.add_argument("--opt", type=str, choices=["gd", "polyak", "nesterov"],
                        help="which optimization algorithm to use", default="gd")
    parser.add_argument("--seed", type=int, help="the random seed used when initializing the network weights",
                        default=0)
    parser.add_argument("--beta", type=float, help="momentum parameter (used if opt = polyak or nesterov)")
    parser.add_argument("--physical_batch_size", type=int,
                        help="the maximum number of examples that we try to fit on the GPU at once", default=1000)
    parser.add_argument("--acc_goal", type=float,
                        help="terminate training if the train accuracy ever crosses this value")
    parser.add_argument("--loss_goal", type=float, help="terminate training if the train loss ever crosses this value")
    parser.add_argument("--neigs", type=int, help="the number of top eigenvalues to compute")
    parser.add_argument("--eig_freq", type=int, default=-1,
                        help="the frequency at which we compute the top Hessian eigenvalues (-1 means never)")
    parser.add_argument("--nproj", type=int, default=0, help="the dimension of random projections")
    parser.add_argument("--iterate_freq", type=int, default=-1,
                        help="the frequency at which we save random projections of the iterates")
    parser.add_argument("--abridged_size", type=int, default=5000,
                        help="when computing top Hessian eigenvalues, use an abridged dataset of this size")
    parser.add_argument("--save_freq", type=int, default=-1,
                        help="the frequency at which we save resuls")
    parser.add_argument("--save_model", type=bool, default=False,
                        help="if 'true', save model weights at end of training")
    parser.add_argument("--batch_norm", action="store_true",
                        help="if 'true', use batch normalization")
    parser.add_argument("--width", type=int, default=200,
                        help="the width of the neural networks")
    parser.add_argument("--bias", action="store_true",
                        help="set bias=True")
    parser.add_argument("--init_bias", type=str, default="b_init",
                        help="if 'b_init', no changes to bias; if 'b1_x_b2_y', normalize the norm of bias to x and y for first and second layer")
    parser.add_argument("--init_weight", type=str, default="w_init",
                        help="if 'w_init', no changes to weight; if 'w1_x_w2_y', normalize the norm of weight to x and y for first and second layer")
    
    args = parser.parse_args()
    print(args.width, args.bias, args.init_bias, args.init_weight)

    main(dataset=args.dataset, arch_id=args.arch_id, loss=args.loss, opt=args.opt, lr=args.lr, max_steps=args.max_steps,
         neigs=args.neigs, physical_batch_size=args.physical_batch_size, eig_freq=args.eig_freq,
         iterate_freq=args.iterate_freq, save_freq=args.save_freq, save_model=args.save_model, beta=args.beta,
         nproj=args.nproj, loss_goal=args.loss_goal, acc_goal=args.acc_goal, abridged_size=args.abridged_size,
         seed=args.seed, 
         width=args.width, bias=args.bias, init_bias=args.init_bias, init_weight=args.init_weight, batch_norm=args.batch_norm)
