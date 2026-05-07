import argparse
import torch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import *
from utils import *


def run(argparser):
    
    args = argparser.parse_args()
    dataset = args.dataset
    nLayers_inf = args.nLayers_inf
    nLayers_gen = args.nLayers_gen
    nHeads = args.nHeads
    embedDim = args.embedDim
    Dr = args.Dr
    neigh_drop = args.neigh_drop
    neigh_drop_m = args.neigh_drop_m
    neigh_drop_n = args.neigh_drop_n
    layer_drop = args.layer_drop

    lr = args.lr
    niters = args.niters
    
    batch_size_train = args.batch_size_train
    batch_size_eval = args.batch_size_eval
    n_val_sweeps = args.n_val_sweeps
    
    delta = args.delta
    beta = args.beta
    
    lowmem = args.lowmem
    val_perc = args.val_perc
    subsamp = args.subsamp
    
    f = nn.ELU()
    
    #niters = 1000
    
    #nLayers_inf = 2
    #nLayers_gen = 2
    #nHeads = 10
    #embedDim = 100
    #f = nn.ELU()
    #Dr = 1
    #Dy = L + Dr
    #neigh_drop = 0#.2
    #neigh_drop_m = 0#.2
    #neigh_drop_n = 0#.2
    #layer_drop = 0#.2
    
    #batch_size = 1
    #subsamp = .5
    
    #batch_size_train = batch_size
    #batch_size_eval = 5
    #n_val_samps = 1
    
    #delta = .01
    #beta = 1
    
    #lowmem = False
    
    #val_perc = .1
    
    data, truths = get_data(dataset)
    L, M, N = data.shape
    
    train_data, val_data = train_val_split(data, val_perc)
    data_flattened = data.to(torch.float32).reshape(L,-1)[:,data.any(0).reshape(-1)].T.to(device)
    data_flattened_ = torch.cat([torch.zeros((1, L), device=device), data_flattened])
    
    train_data_flattened = train_data.to(torch.float32).reshape(L,-1)[:,train_data.any(0).reshape(-1)].T.to(device)
    train_data_flattened_ = torch.cat([torch.zeros((1, L), device=device), train_data_flattened])
    
    adj, adj_m, adj_n = get_adjs(data)
    
    lr = 1e-4
    
    model = VAE(M, nLayers_inf, nLayers_gen, f, embedDim, nHeads, L, Dr, layer_drop, neigh_drop, neigh_drop_m, neigh_drop_n, lowmem).to(device)
    
    counts = data.sum(1).T
    mv_logprobs = torch.log((counts + delta)**beta) - torch.log(((counts + delta)**beta).sum(-1))[:, None]
    
    #print('MV acc {:.4f}'.format((mv_logprobs.argmax(-1).detach()==truths).to(torch.float32).mean()))
    
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optim, niters)
    
    max_val_acc = 0
    max_val = np.inf
    
    
    for it in range(niters):
    
        all_data, all_data_flattened_, all_Ns, all_adj_subsamp, all_adj_m_subsamp, all_adj_n_subsamp = get_all_subsamps(train_data,
                                                                                                    batch_size_train, subsamp,True,False)
    
        model.train()
    
        edge_preds, r_mu, r_logsigma, theta_train = model.forward([all_data_flattened_,
                                                 all_adj_subsamp, all_adj_m_subsamp, all_adj_n_subsamp])
    
        theta_train = theta_train + 1e-8
    
        train_data_permed = train_data[:,:,all_Ns.reshape(-1)].permute((2,0,1)).reshape((batch_size_train,N,L,M)).permute((0,2,3,1))
    
        train_reconstruction = torch.log(edge_preds[train_data_permed] + 1e-8).sum()
        logp_t = (theta_train * mv_logprobs[all_Ns.reshape(-1)].reshape((batch_size_train, N, L))).sum()
    
        logp_r = -.5 * (r_mu**2 + torch.exp(r_logsigma)**2).sum()
        H_q_t = -(theta_train * torch.log(theta_train)).sum()
        H_q_r = r_logsigma.sum()
        loss = -(train_reconstruction + logp_r + logp_t + H_q_t + H_q_r)
        
        loss.backward()
        optim.step()
        optim.zero_grad()
        scheduler.step()
    
        model.eval()
        val_loss = 0
        theta_samps = torch.zeros((n_val_sweeps*batch_size_eval,N,L),device=device)
        with torch.no_grad():
            for i in range(n_val_sweeps):
    
                all_data, all_data_flattened_, all_Ns, all_adj_subsamp, all_adj_m_subsamp, all_adj_n_subsamp = get_all_subsamps(data,
                                                                batch_size_eval, subsamp,False,False)
    
                edge_preds, r_mu, r_logsigma, theta = model.forward([all_data_flattened_,
                                                 all_adj_subsamp, all_adj_m_subsamp, all_adj_n_subsamp])
    
                valid_reconstruction = torch.log(edge_preds[val_data[None,:,:,:].repeat(batch_size_eval, 1, 1, 1)] + 1e-10).mean()
                val_loss -= valid_reconstruction/n_val_sweeps
                theta_samps[i*batch_size_eval:(i+1)*batch_size_eval] = theta
    
            acc = (theta_samps.sum(0).argmax(-1) == truths).to(torch.float32).mean()
    
    
        if val_loss < max_val:
            max_val = val_loss
            max_val_acc = acc
    
        if it % 1 == 0:
            print(f'{it}, train loss: {loss.item():.4f}, val acc: {acc:.4f}, val loss: {val_loss:.4f}, best val {max_val:.4f} mv acc: {max_val_acc:.4f}')

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('-dataset',choices=['cf','face','dog','senti','prod','adult','rte','labelme'])
    p.add_argument('-nLayers_inf', dest='nLayers_inf', action='store',type=int, default=2)
    p.add_argument('-nLayers_gen', dest='nLayers_gen', action='store',type=int, default=2)
    p.add_argument('-nHeads', dest='nHeads', action='store',type=int, default=10)
    p.add_argument('-embedDim', dest='embedDim', action='store',type=int, default=100)
    p.add_argument('-Dr', dest='Dr', action='store',type=int, default=1)
    p.add_argument('-neigh_drop', dest='neigh_drop', action='store',type=float, default=.5)
    p.add_argument('-neigh_drop_m', dest='neigh_drop_m', action='store',type=float, default=.5)
    p.add_argument('-neigh_drop_n', dest='neigh_drop_n', action='store',type=float, default=.5)
    p.add_argument('-layer_drop', dest='layer_drop', action='store',type=float, default=.5)
    p.add_argument('-lr', dest='lr', action='store',type=float, default=1e-4)
    p.add_argument('-niters', dest='niters', action='store',type=int, default=1000)
    p.add_argument('-batch_size_train', dest='batch_size_train', action='store',type=int, default=1)
    p.add_argument('-batch_size_eval', dest='batch_size_eval', action='store',type=int, default=1)
    p.add_argument('-n_val_sweeps', dest='n_val_sweeps', action='store',type=int, default=1)
    p.add_argument('-delta', dest='delta', action='store',type=float, default=.01)
    p.add_argument('-beta', dest='beta', action='store',type=float, default=1.0)
    p.add_argument('-lowmem', dest='lowmem', action='store',type=bool, default=False)
    p.add_argument('-val_perc', dest='val_perc', action='store',type=float, default=.1)
    p.add_argument('-subsamp', dest='subsamp', action='store',type=float, default=.8)

    run(p)