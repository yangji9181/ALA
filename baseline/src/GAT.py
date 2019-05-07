"""
Author: Xingyu Fu
Email: xingyuf2@illinois.edu
"""

from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
from datetime import datetime
from random import shuffle
from tqdm import trange
import time
import argparse
import pickle
import gensim
import glob
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

sys.path.append('../')
# os.chdir('..')

from src.models import GAT, SpGAT, MLP
from src.utils import print_config, save_checkpoint, save_embedding, construct_feature
from src.dataset import SupDataset, EvaDataset
from src.logger import myLogger
from sklearn.metrics import f1_score



def parse_args():
    # general settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='../data/dblp/',
                        help='dataset name.')
    parser.add_argument('--eval_file', type=str, default='../data/dblp/eval/rel.txt',
                        help='evaluation file path.')
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use")
    parser.add_argument('--log_level', default=20,
                        help='logger level.')
    parser.add_argument("--prefix", type=str, default='',
                        help="prefix use as addition directory")
    parser.add_argument('--suffix', default='', type=str,
                        help='suffix append to log dir')
    parser.add_argument('--log_every', type=int, default=100,
                        help='log results every epoch.')
    parser.add_argument('--eval_every', type=int, default=100,
                        help='save learned embedding every epoch.')

    parser.add_argument('--save_emb', type=int, default=1,
                        help='save learned embedding after training or not.')
    parser.add_argument('--save_emb_file', type=str, default='../data/cora/embed/graphsage_vec_nc.txt',
                        help='save learned embedding to this file path.')
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')


    # sample settings
    parser.add_argument('--neighbor_sample_size', default=30, type=int,
                        help='sample size for neighbor to be used in gcn')
    parser.add_argument('--sample_size', default=1000, type=int,
                        help='sample size for training data')
    parser.add_argument('--negative_sample_size', default=1, type=int,
                        help='negative sample / positive sample')
    parser.add_argument('--sample_embed', default=2500, type=int,
                        help='sample size for embedding generation')


    # training settings
    parser.add_argument('--sparse', type=int, default=0
                        , help='GAT with sparse version or not.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=8,
                        help='Number of hidden units.')
    parser.add_argument('--nb_heads', type=int, default=8,
                        help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.6,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Alpha for the leaky_relu.')
    parser.add_argument('-early_stop', type=int, default=1,
                        help='whether to use early stop')
    parser.add_argument('-patience', type=int, default=50,
                        help='used for early stop')

    # evluating settings
    parser.add_argument('--epochs_eval', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr_eval', type=float, default=0.0001,
                        help='Initial learning rate.')
    parser.add_argument('--batch_size_eval', type=float, default=10,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden_eval', type=int, default=100,
                        help='Number of hidden units.')
    parser.add_argument('-patience_eval', type=int, default=500,
                        help='used for early stop in evaluation')

    return parser.parse_args()


def evaluate(args, embedding, logger, repeat_times=5):
    best_train_accs, best_test_accs, best_test_f1s = [], [], []
    best_train_acc_epochs, best_test_acc_epochs, best_test_f1_epochs = [], [], []
    data = construct_feature(args.label_data, embedding)
    split = int(len(args.label_data) / repeat_times)

    for i in range(repeat_times):

        p1, p2 = i*split, (i+1)*split
        test = data[p1:p2, :]
        train1, train2 = data[:p1, :], data[p2:, :]
        train = np.concatenate([train1, train2])

        X_train, y_train = torch.FloatTensor(train[:, :-1]), torch.LongTensor(train[:, -1])
        X_test, y_test = torch.FloatTensor(test[:, :-1]), torch.LongTensor(test[:, -1])
        dataloader = DataLoader(EvaDataset(X_train, y_train), batch_size=args.batch_size_eval, shuffle=True)
        X_train = X_train.to(args.device)
        X_test = X_test.to(args.device)
        y_train = y_train.to(args.device)
        y_test = y_test.to(args.device)

        kwargs = {
            'input_dim': X_train.size(1),
            'hidden_dim': args.hidden_eval,
            'output_dim': args.output_dim
        }
        model = MLP(**kwargs).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_eval)
        best_test_acc, best_train_acc = 0, 0
        best_test_acc_epoch, best_train_acc_epoch = 0, 0
        best_test_f1 = 0
        best_test_f1_epoch = 0
        count = 0
        for epoch in range(args.epochs_eval):
            for i, (batch, label) in enumerate(dataloader):
                optimizer.zero_grad()
                loss = model(batch.to(args.device), label.to(args.device))
                loss.backward()
                optimizer.step()

            preds, test_acc = model.predict(X_test, y_test)
            f1 = f1_score(y_true=list(map(int, test[:, -1])), y_pred=preds, average='micro') * 100
            test_acc *= 100
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_acc_epoch = epoch + 1
                best_pred = preds
                count = 0
            else:
                count += 1
                if count >= args.patience_eval:
                    break
            if f1 > best_test_f1:
                best_test_f1 = f1
                best_test_f1_epoch = epoch + 1

            _, train_acc = model.predict(X_train, y_train)
            train_acc *= 100
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                best_train_acc_epoch = epoch + 1

            print('\repoch {}/{} train acc={:.4f}, test acc={:.4f}, test f1={:.4f}, best train acc={:.4f} @epoch:{:d}, best test acc={:.4f} @epoch:{:d}, best test f1={:.4f} @epoch:{:d}'.
                  format(epoch + 1, args.epochs_eval, train_acc, test_acc, f1, best_train_acc, best_train_acc_epoch, best_test_acc, best_test_acc_epoch, best_test_f1, best_test_f1_epoch), end='')
            sys.stdout.flush()

        print('')
        best_train_accs.append(best_train_acc)
        best_test_accs.append(best_test_acc)
        best_train_acc_epochs.append(best_train_acc_epoch)
        best_test_acc_epochs.append(best_test_acc_epoch)
        best_test_f1s.append(best_test_f1)
        best_test_f1_epochs.append(best_test_f1_epoch)

    best_train_acc, best_train_acc_epoch, best_test_acc, best_test_acc_epoch, best_test_f1, best_test_f1_epoch= \
        np.mean(best_train_accs), np.mean(best_train_acc_epochs), np.mean(best_test_accs), np.mean(best_test_acc_epochs), np.mean(best_test_f1s), np.mean(best_test_f1_epochs)
    std = np.std(best_test_accs)
    std_f1 = np.std(best_test_f1s)
    logger.info('{}: best train acc={:.2f} @epoch:{:d}, best test acc={:.2f} += {:.2f}, @epoch:{:d}, best test f1={:.2f} += {:.2f}, @epoch:{:d}'.
                format(args.eval_file, best_train_acc, int(best_train_acc_epoch), best_test_acc, std, int(best_test_acc_epoch), best_test_f1, std_f1, int(best_test_f1_epoch)))


    return best_train_acc, best_test_acc, std


def train(args, model, data, log_dir, logger, optimizer=None):
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    t = time.time()
    best_acc, best_epoch = 0, 0
    count = 0
    model.train()
    features = data.features
    if args.sparse:
        adjs = data.adj
    else:
        adjs = torch.from_numpy(data.old_adj.todense())

    for epoch in range(1, args.epochs+1):
        losses = []
        optimizer.zero_grad()

        # If data is too big, use sampled data to train: sampled_features, sampled_adjs
        (sampled_features, sampled_adjs), (sampled_link_featuresL, sampled_link_featuresR), sampled_labels = data.sample('link', adjs)
        loss = model(sampled_features, sampled_adjs, sampled_link_featuresL, sampled_link_featuresR, sampled_labels)

        # If data is not too big, use whole data to train: features, adjs
        # loss = model(features, adjs, sampled_link_featuresL, sampled_link_featuresR, sampled_labels)

        loss.backward()
        optimizer.step()

        if epoch % args.log_every == 0:
            losses.append(loss.item())

        if epoch % args.log_every == 0:
            duration = time.time() - t
            msg = 'Epoch: {:04d} '.format(epoch)
            msg += 'loss: {:.4f}\t'.format(loss)
            logger.info(msg+' time: {:d}s '.format(int(duration)))

        if epoch % args.eval_every == 0:
            learned_embed = gensim.models.keyedvectors.Word2VecKeyedVectors(model.nembed)

            # If data is not too big, use whole data to get embeddings
            embedding = model.generate_embedding(features, adjs)
            learned_embed.add([str(i) for i in range(embedding.shape[0])], embedding)

            # If data is too big, Sample data to get embedding
            # for i in range(0, len(args.nodes), args.sample_embed):
            #     nodes = args.nodes[i:i+args.sample_embed]
            #     test_features = features[nodes]
            #     test_adjs = torch.zeros((len(nodes), len(nodes)))
            #     for i, n in enumerate(nodes):
            #         test_adjs[i] = adjs[n][nodes]
            #     embedding = model.generate_embedding(test_features, test_adjs)
            #     learned_embed.add([str(node) for node in nodes], embedding)

            # If data is too big, use only test data to get embedding
            # test_features = features[args.nodes]
            # test_adjs = torch.zeros((len(args.nodes), len(args.nodes)))
            # for i, n in enumerate(args.nodes):
            #     test_adjs[i] = adjs[n][args.nodes]
            # embedding = model.generate_embedding(test_features, test_adjs)
            # learned_embed.add([str(i) for i in args.nodes], embedding)


            train_acc, test_acc, std = evaluate(args, learned_embed, logger)
            duration = time.time() - t
            logger.info('Epoch: {:04d} '.format(epoch)+
                        'train_acc: {:.2f} '.format(train_acc)+
                        'test_acc: {:.2f} '.format(test_acc)+
                        'std: {:.2f} '.format(std)+
                        'time: {:d}s'.format(int(duration)))
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                save_checkpoint({
                    'args': args,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, log_dir,
                    f'epoch{epoch}_time{int(duration):d}_trainacc{train_acc:.2f}_testacc{test_acc:.2f}_std{std:.2f}.pth.tar', logger, True)
                count = 0
            else:
                if args.early_stop:
                    count += args.eval_every
                if count >= args.patience:
                    logger.info('early stopped!')
                    break

    logger.info(f'best test acc={best_acc:.2f} @ epoch:{int(best_epoch):d}')
    if args.save_emb:
        learned_embed = gensim.models.keyedvectors.Word2VecKeyedVectors(model.nembed)
        # If data is not too big, use whole data to get embeddings
        embedding = model.generate_embedding(features, adjs)
        learned_embed.add([str(i) for i in range(embedding.shape[0])], embedding)

        # If data is too big, Sample data to get embedding
        # for i in range(0, len(args.nodes), args.sample_embed):
        #     nodes = args.nodes[i:i+args.sample_embed]
        #     test_features = features[nodes]
        #     test_adjs = torch.zeros((len(nodes), len(nodes)))
        #     for i, n in enumerate(nodes):
        #         test_adjs[i] = adjs[n][nodes]
        #     embedding = model.generate_embedding(test_features, test_adjs)
        #     learned_embed.add([str(node) for node in nodes], embedding)

        # If data is too big, use only test data to get embedding
        # test_features = features[args.nodes]
        # test_adjs = torch.zeros((len(args.nodes), len(args.nodes)))
        # for i, n in enumerate(args.nodes):
        #     test_adjs[i] = adjs[n][args.nodes]
        # embedding = model.generate_embedding(test_features, test_adjs)
        # learned_embed.add([str(i) for i in args.nodes], embedding)
        save_embedding(learned_embed, args.save_emb_file, binary=(os.path.splitext(args.save_emb_file)[1]== 'bin'))

    return best_acc


if __name__ == '__main__':
    # Initialize args and seed
    args = parse_args()
    # print('Number CUDA Devices:', torch.cuda.device_count())
    # call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    # torch.cuda.device(args.gpu)
    args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu > -1 else "cpu")
    # print('Active CUDA Device: GPU', torch.cuda.current_device())
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)

    if not args.eval_file:
        args.eval_file = f'../data/{os.path.basename(args.dataset)}/eval/rel.txt'
    labels, labeled_data = set(), []
    nodes = set()
    with open(args.eval_file, 'r') as lf:
        for line in lf:
            if line.rstrip() == 'test':
                continue
            line = line.rstrip().split('\t')
            if len(line) == 3:
                data1, data2, label = line[0], line[1], int(line[2])
                labeled_data.append((data1, data2, label))
                labels.add(label)
                nodes.update([int(data1), int(data2)])
            else:
                data, label = line[0], int(line[1])
                labeled_data.append((data, label))
                labels.add(label)
                nodes.update([int(data)])
    shuffle(labeled_data)
    args.nodes = list(nodes)
    args.label_data = labeled_data
    args.output_dim = len(labels)

    args.use_superv = 0
    Data = SupDataset(args, args.dataset)
    args.feature_len = Data.feature_len
    args.content_len = Data.content_len
    args.num_node, args.num_link = Data.num_node, Data.num_link


    # Initialize logger
    comment = f'_{os.path.basename(args.dataset)}_{args.suffix}'
    current_time = datetime.now().strftime('%b_%d_%H-%M-%S')
    if args.prefix:
        base = os.path.join('running_log', args.prefix)
        log_dir = os.path.join(base, args.suffix)
    else:
        log_dir = os.path.join('running_log', current_time + comment)
    args.model_path = log_dir

    if not os.path.exists(log_dir): os.makedirs(log_dir)

    args.log_dir = log_dir
    logger = myLogger(name='exp', log_path=os.path.join(log_dir, 'log.txt'))
    print_config(args, logger)
    logger.setLevel(args.log_level)

    # Model and optimizer
    if args.sparse:
        model = SpGAT(device=args.device,
                    nfeat=args.feature_len,
                    nhid=args.hidden,
                    output_dim=args.output_dim,
                    dropout=args.dropout,
                    nheads=args.nb_heads,
                    alpha=args.alpha)
    else:
        model = GAT(device=args.device,
                    nfeat=args.feature_len,
                    nhid=args.hidden,
                    output_dim=args.output_dim,
                    dropout=args.dropout,
                    nheads=args.nb_heads,
                    alpha=args.alpha)

    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train model
    t_total = time.time()
    train(args, model, Data, args.log_dir, logger, optimizer)
    logger.info("Optimization Finished!")
    logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
