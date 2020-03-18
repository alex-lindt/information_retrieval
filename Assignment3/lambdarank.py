import os
import json
import argparse
import pickle as pkl

import numpy as np

import torch
import torch.nn as nn

import dataset
import ranking as rnk
import evaluate as evl

from pointwise_ltr import (
    evaluate_model,
    progress_over_last
)

import pandas as pd


class LambdaRank(nn.Module):
    def __init__(self, n_hidden):
        super(LambdaRank, self).__init__()

        self.output_size = 1
        self.input_size = 501

        self.n_hidden = n_hidden

        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.output_size)
        )

    def forward(self, x):
        return self.net(x)


def train_lambda_rank(ARGS, data, model):
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=ARGS.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(1500, 3500), gamma=0.1)

    # track loss and ndcg on validation set
    loss_curve = []
    ndcg_val_curve = []

    queries = np.arange(0, data.train.num_queries())

    print(f"Starting {ARGS.epochs} epochs: ")
    for epoch in range(ARGS.epochs):

        loss_epoch = []

        for batch in range(ARGS.bpe):
            X, y = sample_batch(data.train, queries, ARGS.device)

            scores = model(X)

            lambdas = lambda_rank_loss(scores, y, ARGS.irm_type)

            loss = (scores * lambdas.detach()).sum()

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()

        loss_curve.append(loss_epoch)

        # compute NDCG on validation set
        val_mean, _ = evaluate_model(model, data.validation, ARGS.device)
        ndcg_val_curve.append(val_mean)

        print(f"[Epoch {epoch}] val ndcg: {np.round(val_mean, 4)}, lr: {scheduler.get_lr()}")

        # early stopping using NDCG on validation set
        # if not progress_over_last(ndcg_val_curve):
        #     break

    return model, loss_curve, ndcg_val_curve


def lambda_rank_loss(scores, y, irm_type, gamma=1.0):
    _lamb = rank_net_loss(scores, y, gamma)
    _irm = irm_delta(scores, y, irm_type)
    lamb_irm = _lamb * _irm
    return lamb_irm.sum(dim=1)[:, None]


def rank_net_loss(scores, y, gamma):
    Sc, S = create_matrices(scores, gamma, y)
    _lambda = gamma * (0.5 * (1 - S) - Sc)
    return _lambda


def irm_delta(scores, y, irm_type):
    scores = scores.detach().numpy().flatten()
    labels = y.detach().numpy().flatten()

    if irm_type == "ndcg":

        k = scores.shape[0]

        sort_ind = np.argsort(scores)[::-1]

        sorted_labels = labels[sort_ind] + 1
        ideal_labels = np.sort(labels)[::-1] + 1

        _dcg = evl.dcg_at_k(sorted_labels, k)
        idcg = evl.dcg_at_k(ideal_labels, k)
        ref_ndcg = _dcg / idcg

        _ndcgs = np.zeros((scores.shape[0], scores.shape[0]))
        for i in range(scores.shape[0]):
            for j in range(scores.shape[0]):
                _scores = np.copy(scores)
                _labels = np.copy(labels)
                _scores[i], _scores[j] = _scores[j], _scores[i]
                # _labels[i], _labels[j] = _labels[j], _labels[i]

                # Taken from evaluation function
                sort_ind = np.argsort(_scores)[::-1]
                sorted_labels = _labels[sort_ind]
                _ndcg = evl.dcg_at_k(sorted_labels, k) / idcg

                _ndcgs[i, j] = np.abs(_ndcg - ref_ndcg)

        return torch.FloatTensor(_ndcgs)

    elif irm_type == "err":

        grades = (np.power(2.0, labels) - 1) / 16

        err = []
        past_p = []
        for k in range(labels.shape[0]):
            err_i = (1 / (k + 1)) * grades[k] * np.prod(past_p)
            err.append(err_i)
            past_p.append(1 - grades[k])
        ref_err = np.sum(err)

        _errs = np.zeros((labels.shape[0], labels.shape[0]))
        for i in range(labels.shape[0]):
            for j in range(labels.shape[0]):

                _grades = np.copy(grades)
                _grades[i], _grades[j] = _grades[j], _grades[i]

                err = []
                past_p = []
                for k in range(labels.shape[0]):
                    err_i = (1 / (k + 1)) * _grades[k] * np.prod(past_p)
                    err.append(err_i)
                    past_p.append(1 - _grades[k])

                _errs[i, j] = np.abs(np.sum(err) - ref_err)

        return torch.FloatTensor(_errs)


def create_matrices(scores, gamma, y):
    Sc = 1 / (1.0 + torch.exp(gamma * (scores - scores.t())))

    # Sc = torch.zeros(scores.shape[0], scores.shape[0])
    # for i, si in enumerate(scores):
    #     for j, sj in enumerate(scores):
    #         Sc[i, j] = gamma * (si - sj)
    # return 1 / (1 + torch.exp(Sc))

    # S = torch.zeros_like(Sc)
    # S[Sc > 0] = 1
    # S[Sc < 0] = -1
    # S[Sc == 0] = 0

    S = torch.zeros_like(Sc)
    for i, eli in enumerate(y):
        for j, elj in enumerate(y):
            if eli > elj:
                S[i, j] = 1
            elif eli < elj:
                S[i, j] = -1
            else:
                S[i, j] = 0
    return Sc, torch.FloatTensor(S)


def sample_batch(data_split, queries, device):
    """
    Randomly sample batch from data_split.

    Returns X,Y where X is a [batch_size, 501] feature vector and
    Y is a [batch_size] label vector.
    """
    # Note: following two lines taken from notebook given in canvas discussion

    qid = np.random.choice(queries, size=1)[0]

    qd_features = data_split.query_feat(qid)
    labels = data_split.query_labels(qid)

    return torch.Tensor(qd_features[:10, ...]).to(device), torch.Tensor(labels[:10, ...]).to(device)
    # return torch.Tensor(qd_features).to(device), torch.Tensor(labels).to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Training
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--n-hidden', type=int, default=256, help='number of hidden layer')
    parser.add_argument('--bpe', type=int, default=10, help='Batches per epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--irm-type', type=str, default="err", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")

    ARGS = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()

    model = LambdaRank(ARGS.n_hidden)

    train_lambda_rank(ARGS, data, model)
