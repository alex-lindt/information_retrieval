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
            nn.Linear(self.n_hidden, self.output_size)
        )

    def forward(self, x):
        return self.net(x)


def train_lambda_rank(ARGS, data, model):
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=ARGS.lr)

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

            with torch.no_grad():
                loss = lambda_rank_loss(scores, y, ARGS.irm_type)

            # loss_epoch.append(loss.item())

            # optimize
            optimizer.zero_grad()
            scores.backward(loss[:, None])
            optimizer.step()

        loss_curve.append(loss_epoch)

        # compute NDCG on validation set
        val_mean, _ = evaluate_model(model, data.validation, ARGS.device)
        ndcg_val_curve.append(val_mean)

        print(f"[Epoch {epoch}] loss: No idea validation ndcg: ({val_mean})")

        # early stopping using NDCG on validation set
        # if not progress_over_last(ndcg_val_curve):
        #     break

    return model, loss_curve, ndcg_val_curve


def lambda_rank_loss(scores, y, irm_type, gamma=1.0):
    return rank_net_loss(scores, y, gamma) * irm_delta(scores, y[:, None], irm_type)


def rank_net_loss(scores, y, gamma):
    Sc, S = create_matrices(scores, gamma, y)

    _lambda = gamma * (0.5 * (1 - S) - Sc)

    return _lambda.sum(dim=1)


def irm_delta(scores, y, irm_type):
    if irm_type == "ndcg":
        y_rels = torch.pow(2.0, y) - 1
        acc_gain = y_rels - y_rels.t()

        _idxs = y.argsort(dim=0, descending=True)

#         TODO. Finish up



def create_matrices(scores, gamma, y):
    Sc = 1 / (1.0 + torch.exp(gamma * (scores - scores.t())))
    # Sc = torch.zeros(scores.shape[0], scores.shape[0])
    # for i, si in enumerate(scores):
    #     for j, sj in enumerate(scores):
    #         Sc[i, j] = gamma * (si - sj)
    # return 1 / (1 + torch.exp(Sc))

    # S = torch.zeros_like(Sc)
    # S[Sc > 0] = 1
    # S[Sc == 0] = 0
    # S[Sc < 0] = -1

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

    return torch.Tensor(qd_features).to(device), torch.Tensor(labels).to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Training
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--n-hidden', type=int, default=256, help='number of hidden layer')
    parser.add_argument('--bpe', type=int, default=10, help='Batches per epoch')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--irm-type', type=str, default="ndcg", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")

    ARGS = parser.parse_args()

    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()

    model = LambdaRank(ARGS.n_hidden)

    train_lambda_rank(ARGS, data, model)
