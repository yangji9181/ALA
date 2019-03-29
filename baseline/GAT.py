import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, alpha, device):
        super(GATLayer, self).__init__()
        
        self.dropout = dropout
        self.device = device
        
        self.W = nn.Parameter(torch.zeros(size=(in_dim, out_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(alpha)
    
    def forward(self, features, adj, target_len, neighbor_len, target_index_out):
        h = torch.mm(features, self.W)
        
        compare = torch.cat([h[adj[0]], h[adj[1]]], dim=1)
        e = self.leakyrelu(torch.matmul(compare, self.a).squeeze(1))
        
        attention = torch.full((target_len, neighbor_len), -9e15).to(self.device)
        attention[target_index_out, adj[1]] = e
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, h)
        return F.elu(h_prime)
    
class GATModel(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout, alpha, device, nhead, nneighbor):
        super(GATModel, self).__init__()
        
        self.nneighbor = nneighbor
        self.dropout = dropout
        
        self.att_layer1 = [GATLayer(in_dim, hid_dim, dropout, alpha, device).to(device) for _ in range(nhead)]
        self.att_layer2 = GATLayer(hid_dim*nhead, out_dim, dropout, alpha, device).to(device)
        
        for i, att in enumerate(self.att_layer1):
            self.add_module('attention1_{}'.format(i), att)
        self.add_module('attention2', self.att_layer2)
        
    def sample(self, adj, samples):

        sample_list, adj_list = [samples], []
        for _ in range(2):
            
            new_samples, new_adjs = set(sample_list[-1]), []
            for sample in sample_list[-1]:
                neighbor_size = adj[1][sample]
                start = adj[1][:sample].sum()
                
                if neighbor_size<=self.nneighbor:
                    curr_new_samples = adj[0][start:start+neighbor_size]                    
                else:
                    curr_new_samples = random.sample(adj[0][start:start+neighbor_size].tolist(), self.nneighbor)
                new_samples = new_samples.union(set(curr_new_samples))
                curr_new_adjs = np.stack(([sample]*len(curr_new_samples), curr_new_samples), axis=-1).tolist()
                curr_new_adjs.append([sample, sample])
                new_adjs.append(curr_new_adjs)

            sample_list.append(np.array(list(new_samples)))
            adj_list.append(np.array([pair for chunk in new_adjs for pair in chunk]).T)
        
        return sample_list, adj_list
    
    def transform(self, sample_list, adj_list):
        
        trans_adj_list, target_index_outs = [], []
        for i, adjs in enumerate(adj_list):
            neighbor_index_dict = {k:v for v,k in enumerate(sample_list[i+1])}
            neighbor_index_out, neighbor_index_in = [neighbor_index_dict[k] for k in adjs[0]], [neighbor_index_dict[k] for k in adjs[1]]
            trans_adj_list.append([neighbor_index_out, neighbor_index_in])
            
            target_index_dict = {k:v for v,k in enumerate(sample_list[i])}
            target_index_out = [target_index_dict[k] for k in adjs[0]]
            target_index_outs.append(target_index_out)
            
        return target_index_outs, trans_adj_list
    
    def forward(self, feats, adj, samples):
        
        sample_list, adj_list = self.sample(adj, samples)
        target_index_outs, trans_adj_list = self.transform(sample_list, adj_list)
        
        x = feats[sample_list[-1]]
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, trans_adj_list[-1], len(sample_list[-2]), len(sample_list[-1]), target_index_outs[-1]) for att in self.att_layer1], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.att_layer2(x, trans_adj_list[-2], len(sample_list[-3]), len(sample_list[-2]), target_index_outs[-2])
        
        return x