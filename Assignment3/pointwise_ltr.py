import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

import dataset
import ranking as rnk
import evaluate as evl
import numpy as np


class Pointwise_LTR_Model(nn.Module):

    def __init__(self, device, n_hidden):
        super().__init__()

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


def sample_batch(data_split, batch_size, device):
    """
    Randomly sample batch from data_split.

    Returns X,Y where X is a [batch_size, 501] feature vector and 
    Y is a [batch_size] label vector.
    """
    # Note: following two lines taken from notebook given in canvas discussion
    idx = np.random.permutation(np.arange(data_split.feature_matrix.shape[0]))[:batch_size]
    X = data_split.feature_matrix[idx, :]
    Y = data_split.label_vector[idx]
    return torch.Tensor(X).to(device), torch.Tensor(Y).to(device)


def progress_over_last(val_curve,n=10):
    """
    Early stopping using the validation set: Check if there is still progress.
    """
    if len(val_curve) < n:
        return True
    return any(val_curve[-1] - v > 1e-4 for v in val_curve[-n:-1])


def evaluate_model(model, data_split, device, metric='ndcg'):
    """
    Function to evaluate the pointwise model.
    """
    scores = model.forward(torch.Tensor(data_split.feature_matrix).to(device))
    scores = scores.cpu().detach().numpy().reshape(-1)
    if metric:
        return evl.evaluate(data_split, scores)[metric]
    return evl.evaluate(data_split, scores)


def train_pointwise_ltr(data, batches_per_epoch=100, n_hidden=512, lr=1e-5, batch_size=50):
    """
    Training a Pointwise LTR model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Pointwise_LTR_Model(device, n_hidden=n_hidden)
    model.train()

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # track loss and ndcg on validation set 
    loss_curve = []
    ndcg_val_curve = []

    epoch = 0
    while True:

        loss_epoch = []

        for _ in range(batches_per_epoch):
            # sample batch & pass through model          
            batch, true_labels = sample_batch(data.train, batch_size, device)
            estimated_labels = model.forward(batch)
            loss = loss_fn(estimated_labels.reshape((-1)), true_labels)
            loss_epoch.append(loss.item())

            # optimize
            optimizer.zero_grad()
            loss.backward()
            # to prevent exploding gradient:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

        loss_curve.append(loss_epoch)

        # compute NDCG on validation set
        val_mean, _ = evaluate_model(model, data.validation, device)
        ndcg_val_curve.append(val_mean)

        print(f"[Epoch {epoch}] loss: {loss.item()} validation ndcg: ({val_mean})")
        epoch += 1

        # early stopping using NDCG on validation set
        if not progress_over_last(ndcg_val_curve, n=3):
            break

    return model, loss_curve, ndcg_val_curve


def grid_search():
    """
    Performs grid search.
    """
    for n_hidden in [256, 512, 1024]:
        for lr in [1e-3, 1e-4, 1e-5]:
            for batch_size in [20, 50, 100]:
                _, loss_curve, ndcg_val_curve = train_pointwise_ltr(data,
                                                                    n_hidden=n_hidden,
                                                                    lr=lr,
                                                                    batch_size=batch_size)
                max_ndcg_val = max(ndcg_val_curve)
                print(f'Hidden {n_hidden} LR {lr} BS {batch_size} : {max_ndcg_val}')


def plot_pointwise_ltr(model, loss_curve, ndcg_val_curve):
    """
    Plot for AQ2.1.
    """
    x = np.linspace(0, len(ndcg_val_curve) + 1, len(ndcg_val_curve))

    loss_means = np.array([np.mean(l) for l in loss_curve])
    loss_stds = np.array([np.std(l) for l in loss_curve])

    plt.title("Loss vs. NDCG on Validation Data")
    plt.xlabel("Epochs")
    plt.plot(x, loss_means, label='Loss over epochs', color='cornflowerblue')
    plt.fill_between(x, loss_means - loss_stds, loss_means + loss_stds, alpha=0.2,  color='cornflowerblue')
    plt.plot(x, ndcg_val_curve, label='NDCG on validation data',  color='crimson')
    plt.legend()

    plt.savefig('pointwise_loss_NDCG')

    
def plot_distribution_of_scores(model, data):
    """
    Plot for AQ2.2.
    """
    model_scores_t = model.forward(torch.Tensor(data.test.feature_matrix).to(device))
    model_scores_t = model_scores_t.cpu().detach().numpy().reshape(-1)

    model_scores_v = model.forward(torch.Tensor(data.validation.feature_matrix).to(device))
    model_scores_v = model_scores_v.cpu().detach().numpy().reshape(-1)

    model_scores = list(np.round(model_scores_t))+list(np.round(model_scores_v))
    model_counts = [model_scores.count(i) for i in range(5)]
    P_model = model_counts / np.sum(model_counts)

    true_scores = list(data.test.label_vector)+list(data.validation.label_vector)
    true_counts = [true_scores.count(i) for i in range(5)]
    P_true = true_counts / np.sum(true_counts)

    x = np.arange(5)
    width = 0.4
    fig, ax = plt.subplots()
    ax.bar(x - width/2, P_model, width, label=f'Pointwise LTR Model', color='cornflowerblue')
    ax.bar(x + width/2, P_true, width, label=f'Groud Truth', color='midnightblue')

    ax.set_ylabel('Probability')
    ax.set_xlabel('Score')
    ax.set_title(f'Distributions of Scores')
    ax.set_xticks(x)
    ax.legend()
    fig.savefig(f'pointwise_distribution')


if __name__ == "__main__":
    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()

    # TRAIN BEST MODEL
    model, loss_curve, ndcg_val_curve = train_pointwise_ltr(data)

    # PLOT FOR AQ2.1
    # plot_pointwise_ltr(model, loss_curve, ndcg_val_curve)

    # PLOT FOR AQ2.1
    # plot_distribution_of_scores(model, data)
    
    # EVALUATE ON TEST SET
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluate_model(model, data.test, device, metric=None)
