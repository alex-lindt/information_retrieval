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
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(3500, 4400), gamma=0.1)

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
                X, y = sample_batch(queries[idx], data.train, ARGS.device)
                if current_bs > ARGS.batch_size:
                    X = X[:current_bs - ARGS.batch_size]
                    y = y[:current_bs - ARGS.batch_size]

                idx += 1

                # if there is only one doc retrieved by the query
                if X.shape[0] == 1:
                    continue

                current_bs += X.size(0)

                scores = model(X)

                _loss = lambda_rank_loss(scores, y, ARGS.irm_type, ARGS.gamma)
                batch_loss.append(_loss)

                batch_loss_report.append(ranknet_loss(scores, y, ARGS.gamma).item())

            # optimize
            optimizer.zero_grad()
            for loss in batch_loss:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            scheduler.step()

            loss_epoch.append(np.mean(batch_loss_report))

        loss_curve.append(loss_epoch)

        # compute NDCG on validation set
        val_mean, _ = calculate_ndcg(model, data.validation, ARGS.device)
        ndcg_val_curve.append(val_mean)

        print(
            f"[Epoch {epoch}], loss: {np.round(loss_epoch[-1], 4)} val ndcg: {np.round(val_mean, 4)}, lr: {scheduler.get_lr()}")

        # early stopping using NDCG on validation set
        # if not progress_over_last(ndcg_val_curve):
        #     break

    return model, loss_curve, ndcg_val_curve


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


def ranknet_loss(scores, labels, gamma):
    _, S = create_matrices(scores, gamma, labels)
    score_diff = scores - scores.t()

    # Assignment Equation (3)
    C = 0.5 * (1 - S) * gamma * score_diff + torch.log2(1 + torch.exp(-gamma * score_diff))

    # WITH MEAN
    # pairs on the diagonal are not valid
    C_T = torch.sum(C * (torch.ones_like(C) - torch.eye(C.shape[0])))
    C_mean = C_T / (C.nelement() - C.shape[0])

    return C_mean


def lambda_rank_loss(scores, y, irm_type, gamma=1.0):
    Sc, S = create_matrices(scores, gamma, y)
    _lambdas = gamma * (0.5 * (1 - S) - Sc)

    # set diagonal of lambda to zero
    lambda_0 = _lambdas * (
            torch.ones_like(_lambdas, dtype=torch.float) - torch.eye(_lambdas.shape[0], dtype=torch.float))

    # Not sure whether lower triangular or full
    _irm = irm_delta(scores, y, irm_type)
    # _irm = torch.triu(irm_delta(scores, y, irm_type), diagonal=1)

    lamb_irm = (lambda_0 * _irm).sum(dim=1, keepdim=True)
    return (scores * lamb_irm.detach()).mean()


def irm_delta(scores, y, irm_type):
    scores = scores.detach().numpy().flatten()
    labels = y.detach().numpy().flatten()

    random_i = np.random.permutation(
        np.arange(scores.shape[0])
    )
    labels = labels[random_i]
    scores = scores[random_i]

    sort_ind = np.argsort(scores)[::-1]
    sorted_labels = labels[sort_ind] + 1

    if irm_type == "ndcg":
        ideal_labels = np.sort(labels)[::-1] + 1

        k = scores.shape[0]

        _dcg = evl.dcg_at_k(sorted_labels, k)
        idcg = evl.dcg_at_k(ideal_labels, k)
        ref_ndcg = _dcg / idcg

        _ndcgs = np.zeros((scores.shape[0], scores.shape[0]))
        for i in range(scores.shape[0]):
            for j in range(scores.shape[0]):
                _scores = np.copy(scores)
                _labels = np.copy(labels)
                _scores[i], _scores[j] = _scores[j], _scores[i]
                # TODO: also swap labels?
                # _labels[i], _labels[j] = _labels[j], _labels[i]

                # Taken from evaluation function
                sort_ind = np.argsort(_scores)[::-1]
                sorted_labels = _labels[sort_ind]
                _ndcg = evl.dcg_at_k(sorted_labels, k) / idcg

                _ndcgs[i, j] = np.abs(_ndcg - ref_ndcg)

        return torch.FloatTensor(_ndcgs)

    elif irm_type == "err":

        grades = (np.power(2.0, sorted_labels) - 1) / 16
        ref_err = calc_err(grades)

        _errs = np.zeros((labels.shape[0], labels.shape[0]))
        for i in range(labels.shape[0]):
            for j in range(labels.shape[0]):
                _grades = np.copy(grades)
                _grades[i], _grades[j] = _grades[j], _grades[i]

                _errs[i, j] = np.abs(calc_err(_grades) - ref_err)

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


def calculate_ndcg(model, data_split, device, print_results=False):
    """
    Function calculate ndcg
    """
    scores = model.forward(torch.Tensor(data_split.feature_matrix).to(device))
    scores = scores.cpu().detach().numpy().reshape(-1)
    return evl.evaluate(data_split, scores, print_results)["ndcg"]


def calculate_err(model, data_split, device, print_results=False):
    results = {}
    for qid in np.arange(data_split.num_queries()):
        if evl.included(qid, data_split):
            qd_features = data_split.query_feat(qid)
            scores = model(torch.Tensor(qd_features).to(device))
            labels = data_split.query_labels(qid)

            scores = scores.detach().numpy().flatten()
            labels = labels.flatten()

            random_i = np.random.permutation(
                np.arange(scores.shape[0])
            )
            labels = labels[random_i]
            scores = scores[random_i]

            sort_ind = np.argsort(scores)[::-1]
            sorted_labels = labels[sort_ind] + 1
            grades = (np.power(2.0, sorted_labels) - 1) / 16

            current_results = {
                'err': calc_err(grades, 0),
                'err@03': calc_err(grades, 3),
                'err@05': calc_err(grades, 5),
                'err@10': calc_err(grades, 10),
                'err@20': calc_err(grades, 20),
            }

            evl.add_to_results(results, current_results)

    print('"metric": "mean" ("standard deviation")')
    mean_results = {}
    for k in sorted(results.keys()):
        v = results[k]
        mean_v = np.mean(v)
        std_v = np.std(v)
        mean_results[k] = (mean_v, std_v)
        if print_results:
            print('%s: %0.04f (%0.05f)' % (k, mean_v, std_v))
    return mean_results["err"]


def calc_err(grades, k=0):
    if k > 0:
        k = min(grades.shape[0], k)
    else:
        k = grades.shape[0]
    grades = grades[: k]

    err = []
    past_p = []
    for k in range(grades.shape[0]):
        err_i = (1 / (k + 1)) * grades[k] * np.prod(past_p)
        err.append(err_i)
        past_p.append(1 - grades[k])

    return np.sum(err)


def plot_lambdarank(identifier, delta_weight, loss_curve, ndcg_val_curve):
    plt.clf()
    plt.cla()
    plt.close()

    x = np.linspace(0, len(ndcg_val_curve) + 1, len(ndcg_val_curve))

    loss_means = np.array([np.mean(l) for l in loss_curve])
    loss_stds = np.array([np.std(l) for l in loss_curve])

    plt.title("Loss vs. NDCG on Validation Data")
    plt.xlabel("Epochs")
    plt.plot(x, loss_means, label='loss over epochs')
    plt.fill_between(x, loss_means - loss_stds, loss_means + loss_stds, alpha=0.2)
    plt.plot(x, ndcg_val_curve, label='NDCG on validation data')
    plt.legend()
    plt.savefig(os.path.join(f"lambdarank", delta_weight, "figures", f'f_NDCGvsLoss_{identifier}.png'))

    plt.clf()
    plt.cla()
    plt.close()

    plt.title("NDCG on Validation Data")
    plt.xlabel("Epochs")
    plt.plot(x, ndcg_val_curve, label='NDCG on validation data')
    plt.legend()
    plt.savefig(os.path.join("lambdarank", delta_weight, "figures", f'f_NDCG_{identifier}.png'))

    plt.clf()
    plt.cla()
    plt.close()

    plt.title("Loss on Validation Data")
    plt.xlabel("Epochs")
    plt.plot(x, loss_means, label='loss over epochs')
    plt.fill_between(x, loss_means - loss_stds, loss_means + loss_stds, alpha=0.2)
    plt.legend()
    plt.savefig(os.path.join("lambdarank", delta_weight, "figures", f'f_loss_{identifier}.png'))


def save_thing(thing, name):
    with open(name, 'wb') as handle:
        pkl.dump(thing, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Training
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--n-hidden', type=int, default=256, help='Number of hidden layer')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--bpe', type=int, default=2, help='Batches per epoch')
    parser.add_argument('--gamma', type=float, default=1.0, help='Loss parameter gamma')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate')
    parser.add_argument('--irm-type', type=str, default="ndcg", help="Delta weight type")
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")

    ARGS = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()

    for delta_weight in ["ndcg", "err"]:
        ARGS.irm_type = delta_weight
        for lr in [0.02, 0.002]:
            ARGS.lr = lr
            for n_hidden in [32, 128, 256]:
                identifier = f"{ARGS.irm_type}_{ARGS.lr}_{ARGS.n_hidden}"

                ARGS.n_hidden = n_hidden

                model = LambdaRank(ARGS.n_hidden)
                model, loss_curve, ndcg_val_curve = train_lambda_rank(ARGS, data, model)

                save_thing(model, os.path.join(f"lambdarank", delta_weight, "models", f"m_{identifier}.pkl"))
                save_thing(loss_curve, os.path.join("lambdarank", delta_weight, "figures", f"loss_{identifier}.pkl"))
                save_thing(ndcg_val_curve,
                           os.path.join("lambdarank", delta_weight, "figures", f"ndcg_val_{identifier}.pkl"))

                print(f"Evaluation for model irm-type: {ARGS.irm_type}, lr: {ARGS.lr}, n-hidden: {ARGS.n_hidden}")
                mean_results_ndcg = calculate_ndcg(model, data.test, ARGS.device, True)
                mean_results_err = calculate_err(model, data.test, ARGS.device, True)

                print(f"Mean nDCG:{mean_results_ndcg}")
                print(f"Mean ERR:{mean_results_err}")

                plot_lambdarank(identifier, delta_weight, loss_curve, ndcg_val_curve)

                print("All saved and plotted, next model!")
