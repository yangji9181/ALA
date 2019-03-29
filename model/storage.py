import numpy as np

import torch

class Storage():
    def __init__(self, nnodes, text_features, emb_features, device):
        super(Storage, self).__init__()
        
        self.text_features = torch.from_numpy(text_features).float().to(device)
        self.emb_features = torch.from_numpy(emb_features).float().to(device)
        self.his_features = torch.from_numpy(np.zeros(emb_features.shape)).float().to(device)
        self.visited = [set([i]) for i in range(nnodes)]        
    
    def update(self, nodes, walks, new_embs):
        for node, walk in zip(nodes, walks):
            self.visited[node].union(walk)
        self.emb_features[nodes] = new_embs       
    
    def history(self, start_nodes):
        for node in start_nodes:
            self.his_features[node] = torch.mean(self.emb_features[list(self.visited[node])], dim=0).float()
        return self.his_features[start_nodes]