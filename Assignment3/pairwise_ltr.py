import time
import argparse
from itertools import permutations

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn

import dataset
import evaluate as evl
from lambdarank import create_matrices
from pointwise_ltr import progress_over_last


class RankNet(nn.Module):
    def __init__(self, n_hidden, gamma, batch_size, device):
        super().__init__()

        self.gamma = gamma
        self.batch_size = batch_size

        self.input_size = 501
        self.output_size = 1

        self.nn = torch.nn.Sequential(
            nn.Linear(self.input_size, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, self.output_size)
        )

        self.to(device)

    def forward(self, x):
        return self.nn(x)

    def loss(self, scores, labels):
        _, S = create_matrices(scores, self.gamma, labels)
        score_diff = scores - scores.t()

        # Assignment Equation (3)
        C = 0.5 * (1 - S) * self.gamma * score_diff + torch.log2(1 + torch.exp(-self.gamma * score_diff))

        # # WITH SAMPLING FIXED BATCH SIZE
        # B = np.ones(C.shape) - np.eye(C.shape[0])
        # idx = B.nonzero()
        # sample = [np.random.randint(0, len(idx[0])) for i in range(self.batch_size)]

        # # i = np.arange(len(idx[0]))
        # # np.random.shuffle(i)
        # # sample = i[:self.batch_size]
        # idx1, idx2 = idx[0][sample], idx[1][sample]

        # C_T = C[idx1, idx2].sum()

        # return C_T

        # WITH MEAN
        # pairs on the diagonal are not valid
        C_T = torch.sum(C * (torch.ones_like(C) - torch.eye(C.shape[0])))
        C_mean = C_T / (C.nelement() - C.shape[0])

        return C_mean

    def spedup_loss(self, scores, labels, ):
        Sc, S = create_matrices(scores, self.gamma, labels)
        _lambda = self.gamma * (0.5 * (1 - S) - Sc)

        # set diagonal of lambda to zero 
        lambda_0 = _lambda * (
                torch.ones_like(_lambda, dtype=torch.float) - torch.eye(_lambda.shape[0], dtype=torch.float))

        # average across i 
        lambdas = (lambda_0.sum(dim=1) / (lambda_0.shape[1] - 1))[:, None]

        return (scores * lambdas.detach()).mean()


def get_samples_by_qid(qid, data_split, device):
    docs = data_split.query_feat(qid)
    labels = data_split.query_labels(qid)

    return torch.Tensor(docs).to(device), torch.Tensor(labels).to(device)


def train_ranknet(ARGS, model, data, spedup=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=ARGS.lr)

    if spedup:
        loss_fn = model.spedup_loss
    else:
        loss_fn = model.loss

    # track loss and ndcg on validation set 
    loss_curve, ndcg_val_curve = [], []

    queries = np.arange(0, data.train.num_queries())

    print(f"Starting {ARGS.epochs} epochs: ")
    for epoch in range(ARGS.epochs):

        loss_epoch = []
        np.random.shuffle(queries)

        idx = 0
        for _ in range(ARGS.bpe):
            batch_loss, batch_loss_report = [], []
            current_bs = 0
            while current_bs <= ARGS.batch_size:

                X, y = get_samples_by_qid(qid=queries[idx], data_split=data.train, device=ARGS.device)

                if current_bs > ARGS.batch_size:
                    X = X[:current_bs - ARGS.batch_size]
                    y = y[:current_bs - ARGS.batch_size]

                idx += 1

                # if there is only one doc retrieved by the query
                if X.shape[0] == 1:
                    continue

                current_bs += X.size(0)

                scores = model(X)

                _loss = loss_fn(scores, y)
                batch_loss.append(_loss)

                if spedup:
                    batch_loss_report.append(model.loss(scores, y).item())
                else:
                    batch_loss_report.append(_loss.item())

            # if there is only one doc retrieved by the query, sample another query
            # if docs.shape[0] < 2:
            #     other_q = queries[100::]
            #     np.random.shuffle(other_q)
            #     for qid_new in other_q:
            #         docs, labels = get_samples_by_qid(qid=qid_new, data_split=data.train, device=device)
            #         if not docs.shape[0] < 2:
            #             break

            # optimize
            optimizer.zero_grad()
            for loss in batch_loss:
                loss.backward()
            # to prevent exploding gradient:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            loss_epoch.append(np.mean(batch_loss_report))

        loss_curve.append(loss_epoch)

        # compute NDCG on validation set
        val_mean, _ = evaluate_ranknet(model, data.validation, ARGS.device)
        ndcg_val_curve.append(val_mean)

        print(f"[Epoch {epoch}] loss: {loss_epoch[-1]} validation ndcg: ({val_mean})")

        # early stopping using NDCG on validation set
        if not progress_over_last(ndcg_val_curve):
            break

    return model, loss_curve, ndcg_val_curve


def evaluate_ranknet(model, data_split, device, metric='ndcg'):
    scores = model.forward(torch.Tensor(data_split.feature_matrix).to(device))
    scores = scores.cpu().detach().numpy().reshape(-1)
    if metric:
        return evl.evaluate(data_split, scores)[metric]
    return evl.evaluate(data_split, scores)


def plot_pairwise_ltr(model, loss_curve, ndcg_val_curve):
    x = np.linspace(0, len(ndcg_val_curve) + 1, len(ndcg_val_curve))

    loss_means = np.array([np.mean(l) for l in loss_curve])
    loss_stds = np.array([np.std(l) for l in loss_curve])

    # plt.title("Loss vs. NDCG on Validation Data")
    # plt.xlabel("Epochs")
    # plt.plot(x, loss_means, label='loss over epochs')
    # plt.fill_between(x, loss_means - loss_stds, loss_means + loss_stds, alpha=0.2)
    # plt.plot(x, ndcg_val_curve, label='NDCG on validation data')
    # plt.legend()

    plt.title("NDCG on Validation Data")
    plt.xlabel("Epochs")
    plt.plot(x, ndcg_val_curve, label='NDCG on validation data')
    plt.legend()
    plt.savefig('pairwise_NDCG')

    plt.clf()
    plt.title("Loss on Validation Data")
    plt.xlabel("Epochs")
    plt.plot(x, loss_means, label='loss over epochs')
    plt.fill_between(x, loss_means - loss_stds, loss_means + loss_stds, alpha=0.2)
    plt.legend()
    plt.savefig('pairwise_loss')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Training
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--n-hidden', type=int, default=256, help='number of hidden layer')
    parser.add_argument('--batch-size', type=int, default=64, help='number of hidden layer')
    parser.add_argument('--bpe', type=int, default=100, help='Batches per epoch')
    parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
    parser.add_argument('--gamma', type=float, default=1.0, help='learning rate')
    parser.add_argument('--irm-type', type=str, default="ndcg", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")

    ARGS = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()

    print('Number of features: %d' % data.num_features)
    print('Number of queries in training set: %d' % data.train.num_queries())
    print('Number of documents in training set: %d' % data.train.num_docs())
    print('Number of queries in validation set: %d' % data.validation.num_queries())
    print('Number of documents in validation set: %d' % data.validation.num_docs())
    print('Number of queries in test set: %d' % data.test.num_queries())
    print('Number of documents in test set: %d' % data.test.num_docs())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RankNet(n_hidden=ARGS.n_hidden, gamma=ARGS.gamma, batch_size=ARGS.batch_size, device=device)

    # tic = time.perf_counter()
    # TODO: Integrate parameters, no priority but looks nicer in the end
    model, loss_curve, ndcg_val_curve = train_ranknet(ARGS, model, data, spedup=False)

    # toc = time.perf_counter() 
    # print(f"Trained in {toc - tic:0.4f} seconds")

    test_mean = evaluate_ranknet(model, data.test, device, metric=None)

    print(test_mean)

    plot_pairwise_ltr(model, loss_curve, ndcg_val_curve)
