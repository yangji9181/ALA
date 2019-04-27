from __future__ import print_function
import numpy
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

import math
import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer, DecodeLink


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer=1):
        super(MLP, self).__init__()
        if layer==1:
            self.classifier = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=True),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(hidden_dim, output_dim, bias=True))
        elif layer==2:
            self.classifier = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=True),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(hidden_dim, hidden_dim, bias=True),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(hidden_dim, output_dim, bias=True))
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch, label):
        return self.loss_fn(self.classifier(batch), label)

    def predict(self, batch, label):
        self.eval()
        _, predicted = torch.max(self.classifier(batch), 1)
        c = (predicted == label).squeeze()
        precision = torch.sum(c).item() / float(c.size(0))
        self.train()
        return predicted.cpu().numpy(), precision


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return numpy.asarray(all_labels)


class Classifier(object):

    def __init__(self, clf):
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)
        Y = self.binarizer.transform(Y)
        self.clf.fit(X, Y)

    def evaluate(self, X, Y):
        top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)
        Y = self.binarizer.transform(Y)
        averages = ["micro", "macro", "samples", "weighted"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)
        # print('Results, using embeddings of dimensionality', len(self.embeddings[X[0]]))
        # print('-------------------')
        print(results)
        return results
        # print('-------------------')

    def predict(self, X, top_k_list):
        X_ = numpy.asarray(X)
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, X, Y, train_precent, seed=0):
        state = numpy.random.get_state()

        training_size = int(train_precent * len(X))
        numpy.random.seed(seed)
        shuffle_indices = numpy.random.permutation(numpy.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

        self.train(X_train, Y_train, Y)
        numpy.random.set_state(state)
        return self.evaluate(X_test, Y_test)


class GAT(nn.Module):
    def __init__(self, device, nfeat, nhid, output_dim, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.device = device
        self.dropout = dropout
        self.nhid = nhid
        self.nheads = nheads
        self.nembed = nhid*nheads

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.classifier = nn.Sequential(nn.Linear(64*2, 64, bias=True),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(64, output_dim, bias=True))
        self.loss_fn = nn.CrossEntropyLoss()
        # self.out_att = GraphAttentionLayer(nhid * nheads*2, 2, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, input, adj, L, R, label):
        input = input.to(self.device)
        L = L.to(self.device)
        R = R.to(self.device)
        adj = adj.to(self.device)
        label = label.to(self.device)

        emb = self.forward_encoder(input, adj)

        embL = emb[L]
        embR = emb[R]
        lr = torch.cat((embL, embR), dim=1)
        output = self.classifier(lr)

        loss = self.loss_fn(output, label)
        return loss

    def forward_encoder(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        return x

    def generate_embedding(self, features, adj):
        features = features.to(self.device)
        adj = adj.to(self.device)
        # embedding = torch.spmm(adj, features).data.cpu().numpy()
        embedding = self.forward_encoder(features, adj).data.cpu().numpy()
        return embedding

    def save_embedding(self, features, adj, path, binary=True):
        learned_embed = gensim.models.keyedvectors.Word2VecKeyedVectors(self.nembed)
        learned_embed.add(list(range(len(features))), self.generate_embedding(features, adj))
        learned_embed.save_word2vec_format(fname=path, binary=binary, total_vec=len(features))
