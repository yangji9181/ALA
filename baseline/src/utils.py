"""
Modified for DBLP data loading.
"""
import os
import numpy as np
import scipy.sparse as sp
import torch
import pickle as pkl
import random


def load_embeddings(filename):
    fin = open(filename, 'r')
    node_num, size = [int(x) for x in fin.readline().strip().split()]
    vectors = {}
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        assert len(vec) == size+1
        vectors[vec[0]] = [float(x) for x in vec[1:]]
    fin.close()
    assert len(vectors) == node_num
    return vectors



def print_config(config, logger=None):
    config = vars(config)
    info = "Running with the following configs:\n"
    for k, v in config.items():
        to_add = "\t{} : {}\n".format(k, str(v))
        if len(to_add) < 1000:
            info += to_add
    if not logger:
        print("\n" + info + "\n")
    else:
        logger.info("\n" + info + "\n")


def save_checkpoint(state, modelpath, modelname, logger=None, del_others=True):
    if del_others:
        for dirpath, dirnames, filenames in os.walk(modelpath):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                if path.endswith('pth.tar'):
                    if logger is None:
                        print(f'rm {path}')
                    else:
                        logger.warning(f'rm {path}')
                    os.system("rm -rf '{}'".format(path))
            break
    path = os.path.join(modelpath, modelname)
    if logger is None:
        print('saving model to {}...'.format(path))
    else:
        logger.warning('saving model to {}...'.format(path))
    torch.save(state, path)


def save_embedding(embedding, path, binary=True):
    embedding.save_word2vec_format(fname=path, binary=binary)


def construct_feature(data, w2v):
    Data = []
    labels = []
    if len(data[0]) == 3:
        for word1, word2, label in data:
            vector1 = w2v[word1]
            vector2 = w2v[word2]
            Data.append(np.concatenate([vector1, vector2]))
            labels.append(label)
    elif len(data[0]) == 2:
        for word, label in data:
            vector = w2v[word]
            Data.append(vector)
            labels.append(label)

    Data = np.concatenate((np.array(Data), np.array(labels)[:, np.newaxis]), axis=1)
    return Data


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/dblp/", dataset="dblp", from_text=True):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    """The folloing is for DBLP dataset."""
    if from_text:

        with open(path + 'node_features.csv', 'r') as feature_file:
            num_node = 0
            for feature in feature_file:
                nodeid = int(feature.rstrip().split(',')[0])
                if nodeid > num_node:
                    num_node = nodeid
            num_node += 1  # pad one because the nodeid start from 1 .....
            feature_len = len(feature.rstrip().split(',')[1].split())
            features = np.zeros((num_node, feature_len))

        with open(path + 'node_features.csv', 'r') as feature_file:
            for feature in feature_file:
                nodeid, node_feature = feature.rstrip().split(',')
                features[int(nodeid)] = [float(i) for i in node_feature.split()]

        del feature_file

        dim = num_node
        row_id_list = []
        col_id_list = []
        data_list = []
        max_row_id = 0
        max_col_id = 0
        graph = {}

        link_file = open(path + 'link.csv', 'r').readlines()[1:]
        links = [tuple(map(int, i.rstrip().split(','))) for i in link_file]
        link2weight = {}
        for pair in links:
            if pair not in link2weight:
                link2weight[pair] = 1
            else:
                link2weight[pair] += 1

        for pair in links:
            row_id_list.append(pair[0])
            max_row_id = max(max_row_id, pair[0])
            col_id_list.append(pair[1])
            max_col_id = max(max_col_id, pair[1])
            data_list.append(link2weight[pair])
            if pair[0] not in graph: graph[pair[0]] = []
            if pair[1] not in graph: graph[pair[1]] = []
            graph[pair[0]].append(pair[1])
            graph[pair[1]].append(pair[0])
        dim = max(max_row_id, max_col_id) + 1
        assert dim >= num_node
        matrix_dim = dim
        adj = sp.csr_matrix((data_list, (row_id_list, col_id_list)), shape=(dim, dim))
        for i in range(matrix_dim):
            if i in graph:
                graph[i].sort()

        del link_file, links, link2weight, row_id_list, col_id_list

    else:
        with open((path + "features.p"), 'rb') as feature_file:
            features = pkl.load(feature_file)
        # features = sp.csr_matrix(normalize(features))

        with open((path + "affinity_matrix.p"), 'rb') as adj_file:
            adj = pkl.load(adj_file)
        with open((path + "graph.p"), 'rb') as graph_file:
            graph = pkl.load(graph_file)

    with open((path + 'link.csv'), 'r') as link_file:
        next(link_file)
        links = []
        for line in link_file:
            d1, d2 = line.rstrip().split(',')
            links.append([int(d1), int(d2)])
            links.append([int(d2), int(d1)])
    links = np.array(links)

    feature_sum = np.sum(features, axis=1)
    nonzero_nodes = np.nonzero(feature_sum)[0]
    print("total nonzero features in utils:", len(nonzero_nodes))
    features = normalize(features)
    features = sp.csr_matrix(features)

    # labels = features  # labels will be recalculated in data sampling process
    # adj = sp.coo_matrix(adj) # change format from csr to coo
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # idx_train = range(1369946)
    # idx_val = range(1369946, 1541190)
    # idx_test = range(1541190, adj.shape[0])
    old_adj = adj

    features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    # labels = torch.FloatTensor(np.array(labels.todense()))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, graph, old_adj, links, nonzero_nodes


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)



def accuracy(output, labels):
    preds = output.max(1)[1].long()
    correct = preds.eq(labels.max(1)[1]).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def load_data_GAT(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    with open(path + 'node_features.csv', 'r') as feature_file:
        num_node = 0
        for feature in feature_file:
            nodeid = int(feature.rstrip().split(',')[0])
            if nodeid > num_node:
                num_node = nodeid
        num_node += 1  # pad one because the nodeid start from 1 .....
        feature_len = len(feature.rstrip().split(',')[1].split())
        features = np.zeros((num_node, feature_len))

    with open(path + 'node_features.csv', 'r') as feature_file:
        for feature in feature_file:
            nodeid, node_feature = feature.rstrip().split(',')
            features[int(nodeid)] = [float(i) for i in node_feature.split()]

    del feature_file

    dim = num_node
    row_id_list = []
    col_id_list = []
    data_list = []
    max_row_id = 0
    max_col_id = 0
    graph = {}

    link_file = open(path + 'link.csv', 'r').readlines()[1:]
    links = [tuple(map(int, i.rstrip().split(','))) for i in link_file]
    link2weight = {}
    for pair in links:
        if pair not in link2weight:
            link2weight[pair] = 1
        else:
            link2weight[pair] += 1

    for pair in links:
        row_id_list.append(pair[0])
        max_row_id = max(max_row_id, pair[0])
        col_id_list.append(pair[1])
        max_col_id = max(max_col_id, pair[1])
        data_list.append(link2weight[pair])
        if pair[0] not in graph: graph[pair[0]] = []
        if pair[1] not in graph: graph[pair[1]] = []
        graph[pair[0]].append(pair[1])
        graph[pair[1]].append(pair[0])
    dim = max(max_row_id, max_col_id) + 1
    assert dim >= num_node
    matrix_dim = dim
    for i in range(matrix_dim):
        if i in graph:
            graph[i].sort()
    del link_file, links, link2weight, row_id_list, col_id_list
    with open((path + 'link.csv'), 'r') as link_file:
        next(link_file)
        links = []
        for line in link_file:
            d1, d2 = line.rstrip().split(',')
            links.append([int(d1), int(d2)])
            links.append([int(d2), int(d1)])
    links = np.array(links)
    edges = links.copy()

    # build symmetric adjacency matrix
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(features.shape[0], features.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = normalize(features)
    features = torch.FloatTensor(features)

    return adj, features, graph, links


def generate_seq_data(graph, times=10):
    sampled = []
    nodes = list(graph.nodes())
    for e in range(times):
        for n in nodes:
            last = n
            seq = [last]
            for i in range(times-1):
                next = random.sample(list(graph[last].keys()), 1)[0]
                seq.append(next)
                last = next
            sampled.append(seq)