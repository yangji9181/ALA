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

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append('../')
# os.chdir('..')

from src.models import MLP, Classifier
from src.utils import print_config, construct_feature
from src.dataset import EvaDataset
from src.logger import myLogger
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression


def parse_args():
    # general settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='lp', choices=['nc', 'lp'],
                        help='evaluation type, node classification or label prediction.')
    parser.add_argument('--dataset', default='../data/cora/',
                        help='dataset name.')
    parser.add_argument('--eval_file', type=str, default='../data/cora/eval/label.txt',
                        help='evaluation file path.')
    parser.add_argument('--embedding_file', type=str, default='../data/cora/embed/tadw_cora_vec.txt',
                        help='learned embedding file path.')
    parser.add_argument("--load_model", type=str, default=False,
                        help="whether to load model")
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
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')


    # evluating settings
    parser.add_argument('--clf-ratio', default=0.5, type=float,
                        help='The ratio of training data in the classification')
    parser.add_argument('--sample_embed', default=100, type=int,
                        help='sample size for embedding generation')
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


def evaluate_lp(args, data, logger, repeat_times=5):
    best_train_accs, best_test_accs, best_test_f1s = [], [], []
    best_train_acc_epochs, best_test_acc_epochs, best_test_f1_epochs = [], [], []

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
            f1 = f1_score(y_true=list(map(int, test[:, -1])), y_pred=preds, average='macro') * 100
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


def evaluate_nc(args, data, logger, repeat_times=5):
    X, Y_all = data[:, :-1], data[:, -1]

    best_train_accs, best_test_accs, best_test_f1s = [], [], []
    best_train_acc_epochs, best_test_acc_epochs, best_test_f1_epochs = [], [], []

    split = int(len(args.label_data) / repeat_times)

    for i in range(repeat_times):
        p1, p2 = i * split, (i + 1) * split
        test = data[p1:p2, :]
        train1, train2 = data[:p1, :], data[p2:, :]
        train = np.concatenate([train1, train2])

        X_train, y_train = train[:, :-1], train[:, -1]
        X_test, y_test = test[:, :-1], test[:, -1]
        dataloader = DataLoader(EvaDataset(X_train, y_train), batch_size=args.batch_size_eval, shuffle=True)

        kwargs = {
            'input_dim': X_train.size(1),
            'hidden_dim': args.hidden_eval,
            'output_dim': args.output_dim
        }
        model = Classifier(clf=LogisticRegression())

        best_test_acc, best_train_acc = 0, 0
        best_test_acc_epoch, best_train_acc_epoch = 0, 0
        best_test_f1 = 0
        best_test_f1_epoch = 0
        count = 0
        for epoch in range(args.epochs_eval):

            model.train(X_train, y_train, Y_all)
            test_acc, f1 = model.evaluate(X_test, y_test, average='micro')
            f1 *= 100
            test_acc *= 100
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_acc_epoch = epoch + 1
                count = 0
            else:
                count += 1
                if count >= args.patience_eval:
                    break
            if f1 > best_test_f1:
                best_test_f1 = f1
                best_test_f1_epoch = epoch + 1

            train_acc, _ = model.predict(X_train, y_train)
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

    best_train_acc, best_train_acc_epoch, best_test_acc, best_test_acc_epoch, best_test_f1, best_test_f1_epoch = \
        np.mean(best_train_accs), np.mean(best_train_acc_epochs), np.mean(best_test_accs), np.mean(
            best_test_acc_epochs), np.mean(best_test_f1s), np.mean(best_test_f1_epochs)
    std = np.std(best_test_accs)
    std_f1 = np.std(best_test_f1s)
    logger.info(
        '{}: best train acc={:.2f} @epoch:{:d}, best test acc={:.2f} += {:.2f}, @epoch:{:d}, best test f1={:.2f} += {:.2f}, @epoch:{:d}'.
        format(args.eval_file, best_train_acc, int(best_train_acc_epoch), best_test_acc, std, int(best_test_acc_epoch),
               best_test_f1, std_f1, int(best_test_f1_epoch)))

    return best_train_acc, best_test_acc, std


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

    # Load data
    labels, labeled_data = set(), []
    nodes = set()
    with open(args.eval_file, 'r') as lf:
        for line in lf:
            if line.rstrip() == 'test':
                continue
            line = line.rstrip().split()
            if args.type == 'lp':
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

    embedding = {}
    f = open(args.embedding_file, 'r').readlines()
    [node_num, embed_dim] = list(map(int, f[0].split()))
    for line in f[1:]:
        node, vec = line.split()[0], line.split()[1:]
        embedding[node] = np.array(vec, dtype=np.float32)
    assert len(embedding) == node_num


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

    # Evaluate model
    t_total = time.time()
    data = construct_feature(args.label_data, embedding)
    logger.info("Evaluation Finished!")
    if args.type == 'lp':
        evaluate_lp(args, data, logger)
    else:
        evaluate_nc(args, data, logger)
    logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
