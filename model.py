import torch
from torch import nn
from torch.autograd import grad
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_add, scatter_mean
from utils import *
from reinmax import reinmax

def lowmem_attention(x, y, y_neigh, rows, cols, attention_a, attention_f, neigh_drop, training):
    # x: [nx, nHeads, d], y and y_neigh: [ny, nHeads, d], rows/cols: flattened edges sorted by rows
    # neighbors are dropped before attention is computed, with one mask shared by all heads
    if training and neigh_drop > 0:
        keep = drop_neighbors(rows, x.shape[0], neigh_drop)
        rows, cols = rows[keep], cols[keep]

    log_alphas = torch.squeeze(attention_f(attention_a(torch.cat([x[rows], y[cols]],-1))),dim=-1)
    alphas = scatter_softmax(src=log_alphas, index=rows, dim=0)

    return scatter_add(src=alphas[:,:,None]*y_neigh[cols], index=rows, dim=0, dim_size=x.shape[0])

class BipartiteGraphAttentionModule(nn.Module):
    def __init__(self, f, embedDim, nHeads, layer_drop, neigh_drop, lowmem=False):
        super().__init__()
        self.f = f
        self.nHeads = nHeads
        self.embedDim = embedDim
        self.logit_lambda_self = nn.Parameter(torch.zeros(nHeads))
        self.logit_lambda_neigh = nn.Parameter(torch.zeros(nHeads))
        self.sigm = nn.Sigmoid()

        self.W_self = nn.Linear(embedDim, embedDim)
        self.W_neigh = nn.Linear(embedDim, embedDim)
        
        self.attention_a = nn.Linear(2 * embedDim // nHeads, 1)
        self.attention_f = nn.LeakyReLU()

        self.layer_drop = layer_drop
        self.neigh_drop = neigh_drop

        self.lowmem = lowmem

    def forward(self, x):
        x, y, adj = x

        M = x.shape[1]
        N = y.shape[1]
        batch_size = x.shape[0]
        nEdges = adj.shape[1]

        if self.training and self.layer_drop > 0:
            x = nn.Dropout(self.layer_drop)(x)
            y = nn.Dropout(self.layer_drop)(y)

        x_self = self.W_self(x).reshape((-1, M, self.nHeads, self.embedDim // self.nHeads)).permute((0,2,1,3))
        x_neigh = self.W_neigh(y).reshape((-1, N, self.nHeads, self.embedDim // self.nHeads)).permute((0,2,1,3))

        x_aligned = x.reshape((-1, M, self.nHeads, self.embedDim // self.nHeads))
        y_aligned = y.reshape((-1, N, self.nHeads, self.embedDim // self.nHeads))

        x = x.reshape((-1, M, self.nHeads, self.embedDim // self.nHeads)).permute((0,2,1,3))
        y = y.reshape((-1, N, self.nHeads, self.embedDim // self.nHeads)).permute((0,2,1,3))
        
        if not self.lowmem:

            x_neigh_aligned = x_neigh.permute((0,2,1,3))[torch.arange(batch_size).repeat_interleave(nEdges, dim=0),
                adj[:,:,1].reshape(-1)].reshape((batch_size, nEdges, self.nHeads, self.embedDim // self.nHeads)).permute((0,2,1,3))
            
            x_adj = x_aligned[torch.arange(batch_size).repeat_interleave(nEdges, dim=0),
                adj[:,:,0].reshape(-1)].reshape((batch_size, nEdges, self.nHeads, self.embedDim // self.nHeads)).permute((0,2,1,3))
    
            y_adj = y_aligned[torch.arange(batch_size).repeat_interleave(nEdges, dim=0),
                adj[:,:,1].reshape(-1)].reshape((batch_size, nEdges, self.nHeads, self.embedDim // self.nHeads)).permute((0,2,1,3))

            log_alphas = torch.squeeze(self.attention_f(self.attention_a(torch.cat([x_adj, y_adj],-1))),dim=-1)

            alphas = scatter_softmax(src=log_alphas, index=adj[:,None,:,0].long(), dim=2)
            
            if self.training and self.neigh_drop > 0:
                data_adj_subsamp = torch.zeros((batch_size, M, N), device=device, dtype=torch.int32)
                data_adj_subsamp[torch.arange(batch_size).repeat_interleave(nEdges), adj[:,:,0].reshape(-1), adj[:,:,1].reshape(-1)] = 1
                
                max_nneighs = torch.max(data_adj_subsamp.sum(-1))
                pre_mask = torch.sort(data_adj_subsamp, -1, descending=True)[0].to(torch.int32)[:,:,:max_nneighs]
                
                mask = (torch.rand((batch_size, self.nHeads, M, max_nneighs),device=device) > self.neigh_drop).to(torch.float32)
                add_to_mask = ((pre_mask[:,None,:,:]*mask).sum(-1)==0).to(torch.float32)
                rand_thing = torch.rand((batch_size, self.nHeads, M, max_nneighs),device=device)*(pre_mask -1*(1-pre_mask))[:,None,:,:]
                random_onehot = (rand_thing==torch.max(rand_thing, dim=-1).values[:,:,:,None]).to(torch.float32)
                final_mask = mask + add_to_mask[:,:,:,None] * random_onehot
                final_mask = final_mask.reshape(-1)[pre_mask[:,None,:,:].repeat((1, self.nHeads, 1, 1)).to(torch.bool).reshape(-1)].reshape((batch_size, self.nHeads, nEdges))

                alpha_new_sum = scatter_add(src=alphas*final_mask,index=adj[:,None,:,0].long(),dim=2)
                alpha_new_sum_gather = alpha_new_sum.gather(dim=2, index=adj[:,None,:,0].long().expand(-1, self.nHeads, -1))
                alphas_final = alphas*final_mask/alpha_new_sum_gather

                neigh_new = scatter_add(src=alphas_final[:,:,:,None]*x_neigh_aligned, index = adj[:,None,:,0].long(), dim=2)

                
            else:
                neigh_new = scatter_add(src=alphas[:,:,:,None]*x_neigh_aligned, index=adj[:,None,:,0].long(), dim=2)

        else:
            rows, cols = flatten_adj(adj, M, N)
            neigh_new = lowmem_attention(x_aligned.reshape((-1, self.nHeads, self.embedDim // self.nHeads)),
                y_aligned.reshape((-1, self.nHeads, self.embedDim // self.nHeads)),
                x_neigh.permute((0,2,1,3)).reshape((-1, self.nHeads, self.embedDim // self.nHeads)),
                rows, cols, self.attention_a, self.attention_f, self.neigh_drop, self.training)
            neigh_new = neigh_new.reshape((batch_size, M, self.nHeads, self.embedDim // self.nHeads)).permute((0,2,1,3))

        if neigh_new.shape[2] < M:
            neigh_new = torch.cat([neigh_new, 
                    torch.zeros((batch_size, self.nHeads, M - neigh_new.shape[2], self.embedDim//self.nHeads),device=device)],2)


        x = self.f(self.sigm(self.logit_lambda_self)[None,:,None,None] * x_self +\
           self.sigm(self.logit_lambda_neigh)[None, :, None, None] * neigh_new).permute((0,2,1,3)).reshape((-1, M, self.embedDim))
        
        return x

class BipartiteGraphAttentionInputLayer(nn.Module):
    def __init__(self, Dx, Dy, embedDim):
        super().__init__()
        self.linear_x = nn.Linear(Dx, embedDim)
        self.linear_y = nn.Linear(Dy, embedDim)

    def forward(self, x):
        X, Y, adj = x
        
        return [self.linear_x(X), self.linear_y(Y), adj]

class BipartiteGraphAttentionLayer(nn.Module):
    def __init__(self, f, embedDim, nHeads, layer_drop, neigh_drop, lowmem=False):
        super().__init__()
        self.passX = BipartiteGraphAttentionModule(f, embedDim, nHeads, layer_drop, neigh_drop, lowmem)
        self.passY = BipartiteGraphAttentionModule(f, embedDim, nHeads, layer_drop, neigh_drop, lowmem)

    def forward(self, x):
        X, Y, adj = x
        X_new = self.passX.forward(x)

        nEdges = adj.shape[1]
        batch_size = X.shape[0]
        
        index = torch.sort(adj[:,:,1],dim=1).indices
        adj_y = adj[torch.arange(batch_size,device=device).repeat_interleave(nEdges),
            index.reshape(-1)].reshape((batch_size,nEdges,2))
        Y_new = self.passY.forward([Y, X, adj_y.flip(-1)])
        
        return [X_new, Y_new, adj]

class BipartiteGraphAttentionOutputLayer(nn.Module):
    '''
    Outputs a distribution over E[(m,n)] for all m, n. [M, N, K]
    '''
    def __init__(self, embedDim, L):
        super().__init__()
        self.W = nn.Parameter(torch.randn(L, embedDim, embedDim), requires_grad=True)
    
    def forward(self, x):
        H, J, _ = x
        return nn.Softmax(dim=1)(torch.einsum("nab,kbc,ncd->nkad", H, self.W, J.mT))
    
class BipartiteGraphAttentionNetwork(nn.Module):
    def __init__(self, M, f, embedDim, nHeads, nLayers, L, Dr, layer_drop, neigh_drop, lowmem=False):
        super().__init__()
        layers = [BipartiteGraphAttentionInputLayer(M, Dr + L, embedDim)] + \
            [BipartiteGraphAttentionLayer(f, embedDim, nHeads, layer_drop, neigh_drop, lowmem) for _ in range(nLayers)] + \
            [BipartiteGraphAttentionOutputLayer(embedDim, L)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers.forward(x)


class EdgeModule(nn.Module):
    def __init__(self, f, embedDim, nHeads, layer_drop, neigh_drop_m, neigh_drop_n, lowmem=False):
        super().__init__()
        self.f = f
        self.nHeads = nHeads
        self.embedDim = embedDim
        self.logit_lambda_self = nn.Parameter(torch.zeros(nHeads))
        self.logit_lambda_n = nn.Parameter(torch.zeros(nHeads))
        self.logit_lambda_m = nn.Parameter(torch.zeros(nHeads))
        self.sigm = nn.Sigmoid()

        self.W_self = nn.Linear(embedDim, embedDim)
        self.W_n = nn.Linear(embedDim, embedDim)
        self.W_m = nn.Linear(embedDim, embedDim)
        self.attention_m = nn.Linear(2 * embedDim // nHeads, 1)
        self.attention_n = nn.Linear(2 * embedDim // nHeads, 1)
        self.attention_f = nn.LeakyReLU()

        self.layer_drop = layer_drop
        self.neigh_drop_m = neigh_drop_m
        self.neigh_drop_n = neigh_drop_n

        self.lowmem = lowmem

    def forward(self, x):
        x, adj, adj_m, adj_n = x

        batch_size, nEdges = x.shape[:2]

        nEdges_m = adj_m.shape[1]
        nEdges_n = adj_n.shape[1]

        N = adj[:,:,1].max() + 1

        if self.training and self.layer_drop > 0:
            x = nn.Dropout(self.layer_drop)(x)
        
        x_self = self.W_self(x).reshape((-1, nEdges, self.nHeads, self.embedDim // self.nHeads)).permute((0,2,1,3))

        x_aligned = x.reshape((-1, nEdges, self.nHeads, self.embedDim // self.nHeads))
        x_m = self.W_m(x).reshape((-1, nEdges, self.nHeads, self.embedDim // self.nHeads))
        x_n = self.W_n(x).reshape((-1, nEdges, self.nHeads, self.embedDim // self.nHeads))

        if not self.lowmem:

            x_m_aligned = x_m[torch.arange(batch_size).repeat_interleave(nEdges_m, dim=0),
                adj_m[:,:,1].reshape(-1)].reshape((batch_size, nEdges_m, self.nHeads, self.embedDim // self.nHeads)).permute((0,2,1,3))

            x_adj_m = x_aligned[torch.arange(batch_size).repeat_interleave(adj_m.shape[1], dim=0),
                adj_m[:,:,0].reshape(-1)].reshape((batch_size, adj_m.shape[1], self.nHeads, self.embedDim // self.nHeads)).permute((0,2,1,3))
    
            y_adj_m = x_aligned[torch.arange(batch_size).repeat_interleave(adj_m.shape[1], dim=0),
                adj_m[:,:,1].reshape(-1)].reshape((batch_size, adj_m.shape[1], self.nHeads, self.embedDim // self.nHeads)).permute((0,2,1,3))
    
            log_alphas = torch.squeeze(self.attention_f(self.attention_m(torch.cat([x_adj_m, y_adj_m],-1))),dim=-1)
            alphas = scatter_softmax(src=log_alphas, index=adj_m[:,None,:,0].long(), dim=2)

            if self.training and self.neigh_drop_m > 0:
                nnodes = adj.shape[1] + 1
                pre_mask = get_pre_mask_mn(adj_m, nnodes)
                max_nneighs = pre_mask.shape[-1]
                
                mask = (torch.rand((batch_size, self.nHeads, nnodes, max_nneighs),device=device) > self.neigh_drop_m).to(torch.float32)
                add_to_mask = ((pre_mask[:,None,:,:]*mask).sum(-1)==0).to(torch.float32)
                rand_thing = torch.rand((batch_size, self.nHeads, nnodes, max_nneighs),device=device)*(pre_mask -1*(1-pre_mask))[:,None,:,:]
                random_onehot = (rand_thing==torch.max(rand_thing, dim=-1).values[:,:,:,None]).to(torch.float32)
                final_mask = mask + add_to_mask[:,:,:,None] * random_onehot
                final_mask = final_mask.reshape(-1)[pre_mask[:,None,:,:].repeat((1, self.nHeads, 1, 1)).to(torch.bool).reshape(-1)].reshape((batch_size, self.nHeads, -1))

                alpha_new_sum = scatter_add(src=alphas*final_mask,index=adj_m[:,None,:,0].long(),dim=2)
                alpha_new_sum_gather = alpha_new_sum.gather(dim=2, index=adj_m[:,None,:,0].long().expand(-1, self.nHeads, -1))
                alphas_final = alphas*final_mask/alpha_new_sum_gather

                neigh_m_new = scatter_add(src=alphas_final[:,:,:,None]*x_m_aligned, index = adj_m[:,None,:,0].long(), dim=2)
                    
            else:
                neigh_m_new = scatter_add(src=alphas[:,:,:,None]*x_m_aligned, index = adj_m[:,None,:,0].long(), dim=2)
        else:
            x_flat = x_aligned.reshape((-1, self.nHeads, self.embedDim // self.nHeads))

            rows, cols = flatten_adj(adj_m, nEdges, nEdges)
            neigh_m_new = lowmem_attention(x_flat, x_flat, x_m.reshape((-1, self.nHeads, self.embedDim // self.nHeads)),
                rows, cols, self.attention_m, self.attention_f, self.neigh_drop_m, self.training)
            neigh_m_new = neigh_m_new.reshape((batch_size, nEdges, self.nHeads, self.embedDim // self.nHeads)).permute((0,2,1,3))

        # n
        if not self.lowmem:
            x_n_aligned = x_n[torch.arange(batch_size).repeat_interleave(nEdges_n, dim=0),
                adj_n[:,:,1].reshape(-1)].reshape((batch_size, nEdges_n, self.nHeads, self.embedDim // self.nHeads)).permute((0,2,1,3))

            x_adj_n = x_aligned[torch.arange(batch_size).repeat_interleave(adj_n.shape[1], dim=0),
                adj_n[:,:,0].reshape(-1)].reshape((batch_size, adj_n.shape[1], self.nHeads, self.embedDim // self.nHeads)).permute((0,2,1,3))
    
            y_adj_n = x_aligned[torch.arange(batch_size).repeat_interleave(adj_n.shape[1], dim=0),
                adj_n[:,:,1].reshape(-1)].reshape((batch_size, adj_n.shape[1], self.nHeads, self.embedDim // self.nHeads)).permute((0,2,1,3))    

            log_alphas = torch.squeeze(self.attention_f(self.attention_n(torch.cat([x_adj_n, y_adj_n],-1))),dim=-1)
            alphas = scatter_softmax(src=log_alphas, index=adj_n[:,None,:,0].long(), dim=2)
            
            if self.training and self.neigh_drop_n > 0:

                nnodes = adj.shape[1] + 1
                pre_mask = get_pre_mask_mn(adj_n, nnodes)
                max_nneighs = pre_mask.shape[-1]
                                
                mask = (torch.rand((batch_size, self.nHeads, nnodes, max_nneighs),device=device) > self.neigh_drop_n).to(torch.float32)
                add_to_mask = ((pre_mask[:,None,:,:]*mask).sum(-1)==0).to(torch.float32)
                rand_thing = torch.rand((batch_size, self.nHeads, nnodes, max_nneighs),device=device)*(pre_mask -1*(1-pre_mask))[:,None,:,:]
                random_onehot = (rand_thing==torch.max(rand_thing, dim=-1).values[:,:,:,None]).to(torch.float32)
                final_mask = mask + add_to_mask[:,:,:,None] * random_onehot
                final_mask = final_mask.reshape(-1)[pre_mask[:,None,:,:].repeat((1, self.nHeads, 1, 1)).to(torch.bool).reshape(-1)].reshape((batch_size, self.nHeads, -1))

                alpha_new_sum = scatter_add(src=alphas*final_mask,index=adj_n[:,None,:,0].long(),dim=2)
                alpha_new_sum_gather = alpha_new_sum.gather(dim=2, index=adj_n[:,None,:,0].long().expand(-1, self.nHeads, -1))
                alphas_final = alphas*final_mask/alpha_new_sum_gather

                neigh_n_new = scatter_add(src=alphas_final[:,:,:,None]*x_n_aligned, index = adj_n[:,None,:,0].long(), dim=2)
            else:
                neigh_n_new = scatter_add(src=alphas[:,:,:,None]*x_n_aligned, index = adj_n[:,None,:,0].long(), dim=2)
        else:
            rows, cols = flatten_adj(adj_n, nEdges, nEdges)
            neigh_n_new = lowmem_attention(x_flat, x_flat, x_n.reshape((-1, self.nHeads, self.embedDim // self.nHeads)),
                rows, cols, self.attention_n, self.attention_f, self.neigh_drop_n, self.training)
            neigh_n_new = neigh_n_new.reshape((batch_size, nEdges, self.nHeads, self.embedDim // self.nHeads)).permute((0,2,1,3))

        if neigh_m_new.shape[2] < nEdges:
            neigh_m_new = torch.cat([neigh_m_new, 
                torch.zeros((batch_size, self.nHeads, nEdges-neigh_m_new.shape[2], self.embedDim//self.nHeads),device=device)],2)

        if neigh_n_new.shape[2] < nEdges:
            neigh_n_new = torch.cat([neigh_n_new, 
                torch.zeros((batch_size, self.nHeads, nEdges-neigh_n_new.shape[2], self.embedDim//self.nHeads),device=device)],2)


        
        x = self.f(self.sigm(self.logit_lambda_self)[None, :, None, None] * x_self + \
           self.sigm(self.logit_lambda_m)[None, :, None, None] * neigh_m_new + \
           self.sigm(self.logit_lambda_n)[None, :, None, None] * neigh_n_new).permute((0,2,1,3)).reshape((-1, nEdges, self.embedDim))

        return [x, adj, adj_m, adj_n]

class EdgeInputLayer(nn.Module):
    def __init__(self, L, embedDim):
        super().__init__()
        self.linear = nn.Linear(L, embedDim)

    def forward(self, x):
        x, adj, adj_m, adj_n = x
        return [self.linear(x), adj, adj_m, adj_n]

class EdgeOutputLayer(nn.Module):
    def __init__(self, embedDim, L, Dr):
        super().__init__()
        self.L = L
        self.linear_t = nn.Linear(embedDim, L)
        self.linear_s_mean = nn.Linear(embedDim, Dr)
        self.linear_s_logstd = nn.Linear(embedDim, Dr)

        self.attention_a = nn.Linear(embedDim, 1)
        self.attention_f = nn.LeakyReLU()
        
    def forward(self, x):
        x, adj, _, _ = x

        batch_size = x.shape[0]

        log_alphas = torch.squeeze(self.attention_f(self.attention_a(x)),dim=-1)
        alphas = scatter_softmax(src=log_alphas, 
                     index=torch.cat([torch.zeros((batch_size, 1),device=device),adj[:,:,1]+1],1).long(), dim=1)
        x = scatter_add(src = alphas[:, :, None] * x, dim=1, 
                     index=torch.cat([torch.zeros((batch_size, 1),device=device),adj[:,:,1]+1],1).long())


        x = x[:,1:]
        
        return self.linear_t(x), self.linear_s_mean(x), self.linear_s_logstd(x)

class EdgeNetwork(nn.Module):
    def __init__(self, f, embedDim, nHeads, nLayers, L, Dr, layer_drop, neigh_drop_m, neigh_drop_n, lowmem=False):
        super().__init__()
        layers = [EdgeInputLayer(L, embedDim)] + \
            [EdgeModule(f, embedDim, nHeads, layer_drop, neigh_drop_m, neigh_drop_n, lowmem) for _ in range(nLayers)] + \
            [EdgeOutputLayer(embedDim, L, Dr)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers.forward(x)

class VAE(nn.Module):
    def __init__(self, M, nLayers_inf, nLayers_gen, f, embedDim, nHeads, L, Dr, 
                 layer_drop, neigh_drop, neigh_drop_m, neigh_drop_n, lowmem=False):
        super().__init__()

        self.generative_model = BipartiteGraphAttentionNetwork(M,f, embedDim, nHeads, nLayers_gen, L, Dr, layer_drop, neigh_drop, lowmem)
        self.inference_model = EdgeNetwork(f, embedDim, nHeads, nLayers_inf, L, Dr, layer_drop, neigh_drop_m, neigh_drop_n, lowmem)
        self.Dr = Dr
        self.M = M

    def forward(self, x):
        x, adj, adj_m, adj_n = x

        N = adj[:,:,1].max() + 1
        
        batch_size = x.shape[0]
        
        logits, r_mu, r_logsigma = self.inference_model.forward([x, adj, adj_m, adj_n])

        t_samps = reinmax(logits.reshape((N*batch_size,-1)),1)[0].reshape((batch_size,N,-1))
        
        r_samps = r_mu + torch.exp(r_logsigma) * torch.randn(batch_size, N, self.Dr, device=device)
        
        H = torch.tile(torch.eye(self.M, device=device)[None, :, :], (batch_size, 1, 1))
        J = torch.cat([t_samps, r_samps], -1)

        edge_preds = self.generative_model.forward([H, J, adj])

        return edge_preds, r_mu, r_logsigma, nn.Softmax(dim= -1)(logits)