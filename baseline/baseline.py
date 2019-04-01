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
    
    def __init__(self, dims, dropout, nlayer, device):
        super(GraphSAGEModel, self).__init__()
        
        self.nlayer = nlayer
        self.dropout = dropout
        
        self.layers = []
        for i in range(nlayer):
            self.layers.append(GraphSAGELayer(dims[i], dims[i+1], dropout).to(device))
            self.add_module('layer_{}'.format(i), self.layers[i])
            
        
    def forward(self, x, adj):
                
        for layer in self.layers:
            x = F.dropout(x, self.dropout, training=self.training)
            x = layer(adj, x)
        
        return x