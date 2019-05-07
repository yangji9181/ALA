from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.append('./')

from src.utils import print_config
from src.dataset import EvaDataset
from src.logger import myLogger


def parse_args():
    # general settings
    parser = argparse.ArgumentParser()

    parser.add_argument('--eval_file', type=str, default='',
                        help='evaluation file path.')
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use")
    parser.add_argument('--log_level', default=20,
                        help='logger level.')
    parser.add_argument('--log_every', type=int, default=200,
                        help='log results every epoch.')

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


def evaluate(args, data, logger, Model, repeat_times=5):
    """
    Node Classification.
    Example Use: train_acc, test_acc, std = evaluate(args, learned_embed, logger)
    :param args: Contain args.hidden, args.device, args.output_dim
    :param data: numpy array of shape: [number of sample] * [(dim of node1) + (dim of node2) + (1: label)]
    :param logger: A Logger
    :param Model: The model class passed in
    :param repeat_times: Fold number for cross validation.
    :return:
    """
    best_train_accs, best_test_accs = [], []
    best_train_acc_epochs, best_test_acc_epochs = [], []

    split = int(data.shape[0] / repeat_times)

    for i in range(repeat_times):
        p1, p2 = i * split, (i + 1) * split
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
        model = Model(**kwargs).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_eval)
        best_test_acc, best_train_acc = 0, 0
        best_test_acc_epoch, best_train_acc_epoch = 0, 0
        count = 0
        for epoch in range(args.epochs_eval):
            for i, (batch, label) in enumerate(dataloader):
                optimizer.zero_grad()
                loss = model(batch.to(args.device), label.to(args.device))
                loss.backward()
                optimizer.step()

            preds, test_acc = model.predict(X_test, y_test)
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

            _, train_acc = model.predict(X_train, y_train)
            train_acc *= 100
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                best_train_acc_epoch = epoch + 1

            print('\repoch {}/{} train acc={:.4f}, test acc={:.4f}, best train acc={:.4f} @epoch:{:d}, best test acc={:.4f} @epoch:{:d}'.
                format(epoch + 1, args.epochs_eval, train_acc, test_acc, best_train_acc, best_train_acc_epoch,
                       best_test_acc, best_test_acc_epoch), end='')

        print('')
        best_train_accs.append(best_train_acc)
        best_test_accs.append(best_test_acc)
        best_train_acc_epochs.append(best_train_acc_epoch)
        best_test_acc_epochs.append(best_test_acc_epoch)

    best_train_acc, best_train_acc_epoch, best_test_acc, best_test_acc_epoch = \
        np.mean(best_train_accs), np.mean(best_train_acc_epochs), np.mean(best_test_accs), np.mean(best_test_acc_epochs)
    std = np.std(best_test_accs)
    logger.info('{}: best train acc={:.2f} @epoch:{:d}, best test acc={:.2f} += {:.2f} @epoch:{:d}'.
                format(args.eval_file, best_train_acc, int(best_train_acc_epoch), best_test_acc, std,
                       int(best_test_acc_epoch)))

    return best_train_acc, best_test_acc, std


def evaluate_rel(args, data, logger, Model, repeat_times=5):
    """
    Link Prediction.
    Example use: train_acc, test_acc, std = evaluate_rel(args, data, logger)
    :param args: Contain args.hidden, args.device, args.output_dim
    :param data: numpy array of shape: [number of sample] * [(dim of node1) + (dim of node2) + (1: label)]
    :param logger:
    :param Model: The model class passed in
    :param repeat_times: Fold number for cross validation.
    :return:
    """
    best_train_accs, best_test_accs = [], []
    best_train_acc_epochs, best_test_acc_epochs = [], []

    split = int(len(args.label_data) / repeat_times)

    for i in range(repeat_times):
        p1, p2 = i * split, (i + 1) * split
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
            'output_dim': args.output_dim,
            'layer': 2
        }
        model = Model(**kwargs).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_eval)
        best_test_acc, best_train_acc = 0, 0
        best_test_acc_epoch, best_train_acc_epoch = 0, 0
        count = 0
        for epoch in range(args.epochs_eval):
            for i, (batch, label) in enumerate(dataloader):
                optimizer.zero_grad()
                loss = model(batch.to(args.device), label.to(args.device))
                loss.backward()
                optimizer.step()

            preds, test_acc = model.predict(X_test, y_test)
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

            _, train_acc = model.predict(X_train, y_train)
            train_acc *= 100
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                best_train_acc_epoch = epoch + 1
            print('\repoch {}/{} train acc={:.4f}, test acc={:.4f}, best train acc={:.4f} @epoch:{:d}, best test acc={:.4f} @epoch:{:d}'.
                format(epoch + 1, args.epochs_eval, train_acc, test_acc, best_train_acc, best_train_acc_epoch,
                       best_test_acc, best_test_acc_epoch), end='')

        print('')
        best_train_accs.append(best_train_acc)
        best_test_accs.append(best_test_acc)
        best_train_acc_epochs.append(best_train_acc_epoch)
        best_test_acc_epochs.append(best_test_acc_epoch)

    best_train_acc, best_train_acc_epoch, best_test_acc, best_test_acc_epoch = \
        np.mean(best_train_accs), np.mean(best_train_acc_epochs), np.mean(best_test_accs), np.mean(best_test_acc_epochs)
    std = np.std(best_test_accs)
    logger.info('{}: best train acc={:.2f} @epoch:{:d}, best test acc={:.2f} += {:.2f} @epoch:{:d}'.
                format(args.eval_file, best_train_acc, int(best_train_acc_epoch), best_test_acc, std,
                       int(best_test_acc_epoch)))

    return best_train_acc, best_test_acc, std


if __name__ == '__main__':
    # Initialize args and seed
    args = parse_args()
    args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu > -1 else "cpu")

    # initialize logger
    comment = f'_{args.dataset}_{args.mode}_{args.suffix}'
    log_dir = os.path.join('running_log', comment)
    args.log_dir = log_dir
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    logger = myLogger(name='exp', log_path=os.path.join(log_dir, 'log.txt'))
    print_config(args, logger)
    logger.setLevel(args.log_level)
