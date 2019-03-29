import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(GraphSAGELayer, self).__init__()
        
        self.sage_layer = torch.nn.Sequential(nn.Linear(in_dim, out_dim, bias=False), nn.ReLU())
        self.add_module('sage_layer', self.sage_layer)     
        
    def forward(self, neighbors, emb_features):
        means = torch.stack([torch.mean(emb_features[neighbor], dim=0) for neighbor in neighbors])
        new_emb = self.sage_layer(means)        
        return F.normalize(new_emb, p=2, dim=-1)
    
class GraphSAGEModel(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout, device, nneighbor):
        super(GraphSAGEModel, self).__init__()
        
        self.nneighbor = nneighbor
        self.dropout = dropout
        
        self.sage_layer1 = GraphSAGELayer(in_dim, hid_dim, dropout).to(device)
        self.sage_layer2 = GraphSAGELayer(hid_dim, out_dim, dropout).to(device)
        
        self.add_module('sage1', self.sage_layer1)
        self.add_module('sage2', self.sage_layer2)
        
    def sample(self, adj, samples):

        sample_list, neighbor_list = [samples], []
        for _ in range(2):
            
            new_samples, new_neighbors = set(sample_list[-1]), []
            for sample in sample_list[-1]:
                neighbor_size = adj[1][sample]
                start = adj[1][:sample].sum()
                
                if neighbor_size<=self.nneighbor:
                    curr_new_samples = adj[0][start:start+neighbor_size]                    
                else:
                    curr_new_samples = random.sample(adj[0][start:start+neighbor_size].tolist(), self.nneighbor)
                new_samples = new_samples.union(set(curr_new_samples))
                curr_new_samples = list(curr_new_samples)
                curr_new_samples.append(sample)
                new_neighbors.append(curr_new_samples)
                
            sample_list.append(np.sort(list(new_samples)))
            neighbor_list.append(new_neighbors)
        
        return sample_list, neighbor_list
    
    def transform(self, sample_list, neighbor_list):
        
        trans_neighbor_list = []
        for i, adjs in enumerate(neighbor_list):
            neighbor_index_dict = {k:v for v,k in enumerate(sample_list[i+1])}
            trans_neighbors = []            
            for adj in adjs:
                trans_neighbors.append([neighbor_index_dict[each] for each in adj])
            trans_neighbor_list.append(trans_neighbors)
            
        return trans_neighbor_list
        
    def forward(self, feats, adj, samples):
        sample_list, neighbor_list = self.sample(adj, samples)
        trans_neighbor_list = self.transform(sample_list, neighbor_list)
        
        x = feats[sample_list[-1]]
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.sage_layer1(trans_neighbor_list[-1], x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.sage_layer2(trans_neighbor_list[-2], x)
        
        return x