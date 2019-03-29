import random
import numpy as np
import networkx as nx
from scipy.special import expit
from sklearn.metrics import f1_score, accuracy_score

def prepare(G, ratio):
    
    num_hidden = int(len(G.edges())*ratio)
    posi, nega = [], []
    
    all_edges, bridges = list(G.edges()), list(nx.bridges(G))

    flag = 1
    while flag:
        sampled_edges = np.random.choice(len(all_edges), size=num_hidden, replace=False)
        for edge in sampled_edges:
            edge = all_edges[edge]
            if edge in bridges or edge in posi:
                continue
        
            posi.append(edge)
            G.remove_edge(*edge)
            bridges = list(nx.bridges(G))
        
            if len(posi) == num_hidden:
                flag = 0
                break
                
    all_edges, all_nodes = list(G.edges()), list(G.nodes())
    while len(nega)<num_hidden:
        n1, n2 = random.choice(all_edges)
        n3, n4 = random.choice(all_nodes), random.choice(all_nodes)
        pair1, pair2 = tuple(np.sort([n1,n3])), tuple(np.sort([n2,n4]))
        if not (n1==n3 or pair1 in all_edges or pair1 in nega):
            nega.append(pair1)
        if not (n2==n4 or pair2 in all_edges or pair2 in nega):
            nega.append(pair2)
            
    posi = np.concatenate([np.array(posi), np.ones(len(posi),dtype=int).reshape(-1,1)], axis=1)
    nega = np.concatenate([np.array(nega), np.zeros(len(nega),dtype=int).reshape(-1,1)], axis=1)
    prediction_links = np.concatenate([posi,nega],axis=0)
    
    return G, prediction_links

def lp_evaluate(embeddings, prediction_links):
    
    pred = np.array([np.inner(embeddings[each[0]], embeddings[each[1]]) for each in prediction_links])
    pred = expit(pred)>0.5
    labels = prediction_links[:,2]
    f1, accuracy = f1_score(labels, pred), accuracy_score(labels, pred)
    
    return accuracy, f1