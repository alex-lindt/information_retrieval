import os
import json
import argparse
import pickle as pkl

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import dataset
import evaluate as evl

from lambdarank import create_matrices, irm_delta, sample_batch, LambdaRank, lambda_rank_loss
from pairwise_ltr import RankNet
from pointwise_ltr import Pointwise_LTR_Model


def ranknet_loss(scores, labels, gamma=1.0, irm=False):
    _, S = create_matrices(scores, gamma, labels)
    score_diff = scores - scores.t()

    # assignment Equation (3)
    C = 0.5 * (1 - S) * gamma * score_diff + torch.log2(1 + torch.exp(-gamma * score_diff))

    # pairs on the diagonal are not valid so set diagonal to zero
    C_T = torch.sum(C * (torch.ones_like(C) - torch.eye(C.shape[0])))

    C_mean = C_T / (C.nelement() - C.shape[0])

    return C_mean


def sample_batch(qid, data_split, device):
    """
    Randomly sample batch from data_split.
    Returns X,Y where X is a [batch_size, d, 501] feature vector and
    Y is a [batch_size] label vector.
    """
    # Note: following two lines taken from notebook given in canvas discussion

    qd_features = data_split.query_feat(qid)
    labels = data_split.query_labels(qid)

    return torch.Tensor(qd_features).to(device), torch.Tensor(labels).to(device)


def evaluate_losses(model, data_split, device):
    losses = {"pointwise": [], "pairwise": [], "listwise": []}

    mse = torch.nn.MSELoss()

    no_queries = data_split.num_queries()

    for i, qid in enumerate(np.arange(data_split.num_queries())):
        if evl.included(qid, data_split):
            # print(f"Queries to evaluate: {i}/{no_queries}")

            X, y = sample_batch(qid, data_split, device)
            scores = model(X)

            loss_mse = mse(scores.reshape((-1)), y)
            loss_ranknet = ranknet_loss(scores, y)
            # Loss will be the same?
            loss_lambdarank = ranknet_loss(scores, y)

            if not torch.isnan(loss_mse):
                losses['pointwise'].append(loss_mse.item())
            if not torch.isnan(loss_ranknet):
                losses['pairwise'].append(loss_ranknet.item())
            if not torch.isnan(loss_lambdarank):
                losses['listwise'].append(loss_lambdarank.item())

    return np.mean(losses['pointwise']), np.mean(losses['pairwise']), np.mean(losses['listwise'])


def load_pickle(path_name):
    with open(path_name, "rb") as reader:
        data = pkl.load(reader)
    return data


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(0)
    np.random.seed(0)

    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()
    print("Data loaded ...")

    for model_name in ["pointwise", "pairwise", "listwise"]:
        print(f"Now eval: {model_name}")
        model = load_pickle(os.path.join("aq42", model_name + ".pkl"))
        l_point, l_pair, l_list = evaluate_losses(model, data.test, device)

        print(f"Results for {model_name}: Pointwise {l_point}, RankNet: {l_pair}, and LambdaRank: {l_list}")
