import numpy as np
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
from torch_scatter import scatter_softmax, scatter_add, scatter_mean

def get_adjs(data):
    M,N = data.shape[1:]
    adj = torch.cat([
        torch.arange(M, device=device).repeat_interleave(data.sum(0).sum(-1))[:, None],
        torch.arange(N, device=device)[None, :].repeat((M,1))[data.any(0)][:, None]], 1)

    nEdges = data.sum()
    nrated = data.sum(0).sum(-1)
    nraters = data.sum(0).sum(0)
    item_idx = torch.arange(N, device=device)[None,:].repeat((M,1))[data.any(0)]
    
    
    adj_m = torch.cat([torch.cat([
        torch.arange(nEdges, device=device)[item_idx==n][:,None].repeat((1,nraters[n]))[~torch.eye(nraters[n],device=device,dtype=torch.bool)][:,None],
        torch.arange(nEdges, device=device)[item_idx==n][None,:].repeat((nraters[n],1))[~torch.eye(nraters[n],device=device,dtype=torch.bool)][:,None]],1)\
                          for n in range(N)])
    
    index = torch.sort(adj_m[:,0],dim=0).indices
    adj_m = adj_m[index]
    
    adj_n = torch.cat([torch.cat([
        torch.arange(nEdges, device=device)[adj[:,0]==m][:,None].repeat((1,nrated[m]))[~torch.eye(nrated[m],device=device,dtype=torch.bool)][:,None],
        torch.arange(nEdges, device=device)[adj[:,0]==m][None,:].repeat((nrated[m],1))[~torch.eye(nrated[m],device=device,dtype=torch.bool)][:,None]],1)\
                  for m in range(M)])
    
    index = torch.sort(adj_n[:,0],dim=0).indices
    adj_n_subsamp = adj_n[index]

    return adj, adj_m, adj_n

def get_adjs_subsamp(data, subsamp, resample=True, permute=False):
    L,M,N = data.shape
    npts = data.sum()
    npts_subsamp = (npts*subsamp).to(torch.int32)

    if resample:
        Ns = torch.randint(N, size=(N,), device=device)
        while data[:,:,Ns].sum() < npts_subsamp:
            Ns = torch.randint(N, size=(N,), device=device)
    elif permute:
        npts_subsamp = npts
        Ns = torch.randperm(N,device=device)
    else:
        Ns = torch.arange(N, device=device)
    
    data_ = data[:,:,Ns]
    
    adj = torch.cat([
        torch.arange(M, device=device).repeat_interleave(data_.sum(0).sum(-1))[:, None],
        torch.arange(N, device=device)[None, :].repeat((M,1))[data_.any(0)][:, None]], 1)

    subsamp = torch.sort(torch.randperm(data_.sum())[:npts_subsamp])[0]
    adj = adj[subsamp]
    

    while adj[:,1].max() + 1 < N:
        if resample:
            Ns = torch.randint(N, size=(N,), device=device)
            while data[:,:,Ns].sum() < npts_subsamp:
                Ns = torch.randint(N, size=(N,), device=device)
        elif permute:
            npts_subsamp = npts
            Ns = torch.randperm(N,device=device)
        else:
            Ns = torch.arange(N, device=device)
        
        data_ = data[:,:,Ns]
        
        adj = torch.cat([
            torch.arange(M, device=device).repeat_interleave(data_.sum(0).sum(-1))[:, None],
            torch.arange(N, device=device)[None, :].repeat((M,1))[data_.any(0)][:, None]], 1)

        subsamp = torch.sort(torch.randperm(data_.sum())[:npts_subsamp])[0]
        adj = adj[subsamp]

    
    dataa = torch.zeros((L, M, N), dtype=torch.bool, device=device)
    dataa[:,adj[:,0], adj[:,1]] = data_[:, adj[:,0], adj[:,1]]

    
    nEdges = dataa.sum()
    nrated = dataa.sum(0).sum(-1)
    nraters = dataa.sum(0).sum(0)

    
    adj_subsamp = torch.cat([
        torch.arange(M, device=device).repeat_interleave(dataa.sum(0).sum(-1))[:, None],
        torch.arange(N, device=device)[None, :].repeat((M,1))[dataa.any(0)][:, None]], 1)

    item_idx = torch.arange(N, device=device)[None,:].repeat((M,1))[dataa.any(0)]
    
    adj_m_subsamp = torch.cat([torch.cat([
        torch.arange(nEdges, device=device)[item_idx==n][:,None].repeat((1,nraters[n]))[~torch.eye(nraters[n],device=device,dtype=torch.bool)][:,None],
        torch.arange(nEdges, device=device)[item_idx==n][None,:].repeat((nraters[n],1))[~torch.eye(nraters[n],device=device,dtype=torch.bool)][:,None]],1)\
                          for n in range(N)])
    
    index = torch.sort(adj_m_subsamp[:,0],dim=0).indices
    adj_m_subsamp = adj_m_subsamp[index]
    
    adj_n_subsamp = torch.cat([torch.cat([
        torch.arange(nEdges, device=device)[adj_subsamp[:,0]==m][:,None].repeat((1,nrated[m]))[~torch.eye(nrated[m],device=device,dtype=torch.bool)][:,None],
        torch.arange(nEdges, device=device)[adj_subsamp[:,0]==m][None,:].repeat((nrated[m],1))[~torch.eye(nrated[m],device=device,dtype=torch.bool)][:,None]],1)\
                  for m in range(M)])
    
    index = torch.sort(adj_n_subsamp[:,0],dim=0).indices
    adj_n_subsamp = adj_n_subsamp[index]

    dataa_flattened = dataa.to(torch.float32).reshape(L,-1)[:,dataa.any(0).reshape(-1)].T.to(device)

    return dataa, dataa_flattened, Ns, adj_subsamp, adj_m_subsamp, adj_n_subsamp

def get_maxnneigh(adj):
    batch_size = adj.shape[0]
    foo = (adj[:,:,0] + (torch.arange(batch_size,device=device)*adj.shape[1])[:,None]).reshape(-1)
    skip_vals = adj.shape[1]*torch.arange(batch_size,device=device)
    return torch.bincount(foo[(foo[:,None] != skip_vals[None,:]).all(1)]).max()

def make_pre_mask(adj, nnodes, device):    
    foo = adj.clone().to(torch.int64)
    foo[:,1] = 1
    max_nneighs = torch.bincount(foo[:,0]).max()
    
    counts = scatter_add(src=foo[:,1], index=foo[:,0].long(), dim=0)
    offsets = torch.cat([torch.zeros(1,device=device), counts[:-1]]).cumsum(0)
    offsets_gather = offsets.gather(index=foo[:,0],dim=0)
    
    foo[:,1] = foo[:,1].cumsum(dim=0) - offsets_gather
    foo[:,1] -= 1 #back to 0 indexing
    
    pre_mask = torch.zeros((nnodes, max_nneighs),device=device)
    pre_mask[foo[:,0],foo[:,1]] = 1

    return pre_mask

def get_pre_mask_mn(adj, nnodes):
    foo = adj.clone().to(torch.int64)
    foo[:,:,1] = 1
    max_nneighs = get_maxnneigh(adj)
    batch_size = adj.shape[0]

    counts = scatter_add(src=foo[:,:,1], index=foo[:,:,0].long(), dim=1)
    offsets = torch.cat([torch.zeros((batch_size, 1),device=device), counts[:,:-1]],1).cumsum(1)
    offsets_gather = offsets.gather(index=foo[:,:,0],dim=1)

    foo[:,:,1] = foo[:,:,1].cumsum(dim=1) - offsets_gather
    foo[:,:,1] -= 1 #back to 0 indexing
        
    pre_mask = torch.zeros((batch_size, nnodes, foo[:,:,1].max()+1),device=device)
    pre_mask[torch.arange(batch_size, device=device).repeat_interleave(foo.shape[1]),
        foo[:,:,0].reshape(-1), foo[:,:,1].reshape(-1)] = 1

    return pre_mask

def train_val_split(data, val_perc):
    M,N = data.shape[1:]
    val_mask = torch.zeros(size=((M,N)), dtype=torch.bool, device=device)
    nval = data.sum()*val_perc
    count = 0
    while count < nval:
        train_mask = (data.any(0).to(int) - val_mask.to(int)).to(bool)
        score = torch.min((train_mask.sum(1)-1)[:,None], (train_mask.sum(0)-1)[None,:])*100
        med = score[train_mask.to(bool)][score[train_mask.to(bool)]>0].median()
        #score[score > sat_factor*med] = sat_factor*med
    
        draw = torch.multinomial(((score*train_mask)/(score*train_mask).sum()).reshape(-1), 1)[0]
        val_mask.reshape(-1)[draw] = 1
    
        count += 1
    
    val_data = torch.logical_and(data, val_mask[None,:,:])
    train_data = torch.logical_and(data, ~val_mask[None,:,:])
    return train_data, val_data

def get_subsample(data, subsample):
    Ns = torch.randint(N, (N,), device=device)
    data_mask = (torch.rand((L, M, N), device=device) < subsample).to(torch.float32)
    return Ns, (data[:, :, Ns] * data_mask).to(torch.bool)

def get_all_subsamps(data, batch_size, subsamp, resample=True, permute=False):

    L,M,N = data.shape
    npts_subsamp = (data.sum()*subsamp).to(torch.int32)
    
    all_Ns = torch.zeros((batch_size, N), device=device, dtype=torch.int32)
    all_adj_subsamp = torch.zeros((batch_size, npts_subsamp, 2), dtype=torch.int32, device=device)
    all_adj_m_subsamp = torch.zeros((batch_size, 0, 2), dtype=torch.int32, device=device)
    all_adj_n_subsamp = torch.zeros((batch_size, 0, 2), dtype=torch.int32, device=device)
    all_data_flattened = torch.zeros((batch_size, npts_subsamp, L), device=device)
    all_data = torch.zeros((batch_size, L, M, N),device=device,dtype=torch.bool)
    
    for b in range(batch_size):
        all_data[b], all_data_flattened[b], all_Ns[b], all_adj_subsamp[b], adj_m_subsamp, adj_n_subsamp = \
            get_adjs_subsamp(data, subsamp,resample,permute)
    
        if adj_m_subsamp.shape[0] > all_adj_m_subsamp.shape[1]:
            pad = adj_m_subsamp.shape[0] - all_adj_m_subsamp.shape[1]
            all_adj_m_subsamp = torch.cat([torch.zeros((batch_size, pad, 2), device=device, dtype=torch.int32),
                                           all_adj_m_subsamp], 1)
            all_adj_m_subsamp[b] = adj_m_subsamp + 1
        else:
            all_adj_m_subsamp[b, -adj_m_subsamp.shape[0]:] = adj_m_subsamp
    
        if adj_n_subsamp.shape[0] > all_adj_n_subsamp.shape[1]:
            pad = adj_n_subsamp.shape[0] - all_adj_n_subsamp.shape[1]
            all_adj_n_subsamp = torch.cat([torch.zeros((batch_size, pad, 2), device=device, dtype=torch.int32),
                                           all_adj_n_subsamp], 1)
            all_adj_n_subsamp[b] = adj_n_subsamp + 1
        else:
            all_adj_n_subsamp[b, -adj_n_subsamp.shape[0]:] = adj_n_subsamp
    
    all_data_flattened_ = torch.cat([torch.zeros((batch_size, 1, L), device=device),
                                     all_data_flattened],1)
    
    return all_data, all_data_flattened_, all_Ns, all_adj_subsamp, all_adj_m_subsamp, all_adj_n_subsamp

def get_data(dataset):
    data_ = np.loadtxt('datasets/' + dataset + '/label.csv',skiprows=1,delimiter=',').astype(int)
    truths = np.loadtxt('datasets/' + dataset + '/truth.csv',skiprows=1,delimiter=',').astype(int)[:,1]
    idx = np.loadtxt('datasets/' + dataset + '/truth.csv',skiprows=1,delimiter=',').astype(int)[:,0]
    
    truths = truths[np.argsort(idx)]
    idx = np.sort(idx)
    N = truths.size
    L = data_[:,-1].max()+1
    
    for i in np.arange(N):
        data_[data_[:,0]==idx[i],0] = -(i+1)
    
    data_ = data_[data_[:,0]<0]
    data_[:,0] = -data_[:,0]
    data_[:,0] = data_[:,0] - 1
    
    unique_m = np.unique(data_[:,1])
    M = unique_m.size
    for mm in np.arange(M):
        data_[data_[:,1] == unique_m[mm],1] = -(mm + 1)
    
    data_ = data_[data_[:,1] < 0]
    data_[:,1] *= -1
    data_[:,1] -= 1
    
    data = np.zeros((L,M,N)).astype(bool)
    for i in np.arange(data_.shape[0]):
        #ignore multiple labelings
        if (data[:, data_[i,1], data_[i,0]] == 0).all():
            data[data_[i,-1], data_[i,1], data_[i,0]] = 1
            
    data = torch.from_numpy(data).to(device)
    truths = torch.from_numpy(truths).to(device)
    return data, truths