import numpy as np
import networkx as nx

class RandomWalk():
    def __init__(self, G):
        super(RandomWalk, self).__init__()
        
        self.G = G
        self.alias_nodes = self.get_alias_nodes()
        
    def walk(self, start, p, q, c):
        walk = [start]
        while len(walk)-1<c:
            cur = walk[-1]
            cur_nbrs = sorted(self.G.neighbors(cur))
            if len(cur_nbrs)>0:
                if len(walk)==1:
                    walk.append(cur_nbrs[self.alias_draw(self.alias_nodes[cur][0], self.alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    alias_edge = self.get_alias_edges(prev, cur, p, q)
                    walk.append(cur_nbrs[self.alias_draw(alias_edge[0], alias_edge[1])])
            else:
                break        
        
        distance = nx.shortest_path_length(self.G,source=start,target=walk[-1])
        return walk, distance
    
    def get_alias_nodes(self):        
        alias_nodes = {}
        for node in self.G.nodes():
            unnormalized_probs = [self.G[node][nbr]['weight'] for nbr in sorted(self.G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = self.alias_setup(normalized_probs)
        return alias_nodes
            
    def get_alias_edges(self, src, dst, p, q):
        unnormalized_probs = []
        for dst_nbr in sorted(self.G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(self.G[dst][dst_nbr]['weight']/p)
            elif self.G.has_edge(dst_nbr, src):
                unnormalized_probs.append(self.G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(self.G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
        return self.alias_setup(normalized_probs)
    
    def alias_setup(self, probs):
        K =  len(probs)
        q, J = np.zeros(K), np.zeros(K, dtype=np.int)
        smaller, larger = [], []

        for kk, prob in enumerate(probs):
            q[kk] = K*prob
            if q[kk] < 1.0: smaller.append(kk)
            else: larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small, large = smaller.pop(), larger.pop()
            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            if q[large] < 1.0: smaller.append(large)
            else: larger.append(large)

        return J, q
    
    def alias_draw(self, J, q):
        K = len(J)
        kk = int(np.floor(np.random.rand()*K))
        if np.random.rand() < q[kk]: return kk
        else: return J[kk]