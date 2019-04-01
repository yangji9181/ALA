import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from storage import *
from policy import *
from randomwalk import *
from graphsage import *

class SmartSampling(nn.Module):
    def __init__(self, text_features, emb_features, p_space, q_space, c_space, 
                 nnodes, nsamplers, lr, weight_decay, dropout, device,
                 G):
        super(SmartSampling, self).__init__()
        
        self.nnodes = nnodes
        self.all_nodes = list(G.nodes())
        self.nsamplers = nsamplers
        self.device = device
        self.pdist = nn.PairwiseDistance()
        
        text_dim, emb_dim = len(text_features[0]), len(emb_features[0])
        self.Storage = Storage(nnodes=nnodes, text_features=text_features, emb_features=emb_features, device=device)
        self.Policy = Policy(in_dim=text_dim+emb_dim*2, mid_dim=int((text_dim+emb_dim*3)/2), 
                         p_space=p_space, q_space=q_space, c_space=c_space, dropout=dropout).to(device)
        self.RandomWalk = RandomWalk(G=G)        
        self.GraphSAGE = GraphSAGE(emb_dim, emb_dim, dropout).to(device)
        
        self.add_module('policy', self.Policy)
        self.add_module('graphsage', self.GraphSAGE)
        
        self.policy_optimizer = optim.Adam(self.Policy.parameters(), lr=lr, weight_decay=weight_decay)
        self.sage_optimizer = optim.Adam(self.GraphSAGE.parameters(), lr=lr, weight_decay=weight_decay)
    
    def forward(self):
        
        start_nodes = np.random.choice(self.all_nodes, self.nsamplers, replace=False)
        his_features = self.Storage.history(start_nodes)

        walks, distances = [], []
        
        for i, start in enumerate(start_nodes):
            
            in_features = torch.cat([self.Storage.text_features[start], his_features[i], self.Storage.emb_features[start]])
            p, q, c = self.Policy.action(in_features)
            
            walk, distance = self.RandomWalk.walk(start, p, q, c)            
            walks.append(set(walk))
            distances.append(distance)
        
        new_embs = self.GraphSAGE([list(walk) for walk in walks], self.Storage.emb_features)
        old_embs = self.Storage.emb_features[start_nodes]

        self.Storage.update(start_nodes, walks, new_embs.detach())        
        return new_embs, old_embs, distances
        
    def train(self, nepoch, penalty): 
        self.Policy.train()
        self.GraphSAGE.train()
        
        t = time.time()                 
        for epoch in range(1, nepoch+1):             
            
            self.policy_optimizer.zero_grad()
            self.sage_optimizer.zero_grad()            
            
            new_embs, old_embs, distances = self.forward()
            gains = self.pdist(new_embs, old_embs)

            sage_loss = -torch.mean(gains)
            sage_loss.backward()
            self.sage_optimizer.step()
            
            p_loss, q_loss, c_loss = [], [], []
            rewards = torch.stack([gain.detach()-penalty*distance for gain, distance in zip(gains, distances)])
            rewards = (rewards - rewards.mean()) / rewards.std()

            for reward, log_prob_p, log_prob_q, log_prob_c in zip(rewards, self.Policy.log_probs_p, self.Policy.log_probs_q, self.Policy.log_probs_c):
                p_loss.append(-log_prob_p*reward)
                q_loss.append(-log_prob_q*reward)
                c_loss.append(-log_prob_c*reward)
            p_loss, q_loss, c_loss = torch.stack(p_loss).mean(), torch.stack(q_loss).mean(), torch.stack(c_loss).mean()
            
            p_loss.backward(retain_graph=True)
            q_loss.backward(retain_graph=True)
            c_loss.backward()
            self.policy_optimizer.step()            
            
            del self.Policy.log_probs_p[:]
            del self.Policy.log_probs_q[:]
            del self.Policy.log_probs_c[:]
            
            if epoch%20==0:
                print("Epoch: {}, Gain: {:.4f}, Time: {:.4f}s".format(epoch, -sage_loss.item(), time.time()-t))
                t = time.time() 
                
        return self.Storage.emb_features