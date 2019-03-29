import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(GraphSAGE, self).__init__()
        
        self.sage_layer = torch.nn.Sequential(nn.Linear(in_dim, out_dim, bias=False), nn.ReLU())
        self.add_module('sage_layer', self.sage_layer)
        
    def forward(self, neighbors, emb_features):
        means = torch.stack([torch.mean(emb_features[neighbor], dim=0) for neighbor in neighbors])
        new_emb = self.sage_layer(means)        
        return F.normalize(new_emb, p=2, dim=-1)