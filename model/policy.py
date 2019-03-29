import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, in_dim, mid_dim, p_space, q_space, c_space, dropout):
        super(Policy, self).__init__()
        
        self.p_space = p_space
        self.q_space = q_space
        self.c_space = c_space
        
        self.block_in = torch.nn.Sequential(
            nn.Linear(in_dim, mid_dim), 
            nn.Dropout(p=dropout),
            nn.ReLU())
        
        self.block_p = torch.nn.Sequential(
            nn.Linear(mid_dim, len(p_space)),
            nn.Softmax(dim=-1))
        
        self.block_q = torch.nn.Sequential(
            nn.Linear(mid_dim, len(q_space)),
            nn.Softmax(dim=-1))
        
        self.block_c = torch.nn.Sequential(
            nn.Linear(mid_dim, len(c_space)),
            nn.Softmax(dim=-1))
        
        self.add_module('block_in', self.block_in)
        self.add_module('block_p', self.block_p)
        self.add_module('block_q', self.block_q)
        self.add_module('block_c', self.block_c)
        
        self.log_probs_p, self.log_probs_q, self.log_probs_c = [], [], []
    
    def forward(self, in_features):
        mid_features = self.block_in(in_features)
        p_prob = self.block_p(mid_features)
        q_prob = self.block_q(mid_features)
        c_prob = self.block_c(mid_features)
        return p_prob, q_prob, c_prob
    
    def action(self, in_features):
        p_prob, q_prob, c_prob = self.forward(in_features)
        p_cat, q_cat, c_cat = Categorical(p_prob), Categorical(q_prob), Categorical(c_prob)
        p, q, c = p_cat.sample(), q_cat.sample(), c_cat.sample()
        self.log_probs_p.append(p_cat.log_prob(p))
        self.log_probs_q.append(q_cat.log_prob(q))
        self.log_probs_c.append(c_cat.log_prob(c))
        return self.p_space[p], self.q_space[q], self.c_space[c]